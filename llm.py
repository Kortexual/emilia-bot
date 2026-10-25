# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Emilia: a Discord bot for LM Studio
# Copyright (C) 2026 Jamin Chen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Async client for LM Studio's OpenAI-compatible API, including the tool-calling loop.

How tools work: the model is offered tools such as `web_search`. If it decides to use one, LM Studio
answers with a `tool_calls` entry (possibly next to some text); we run the tool ourselves, hand the
result back as a `tool` message and ask again. The model decides when to use a tool, not a keyword list.
With multi-turn mode on, a tool call lets the model keep working over further turns until it calls
stop_talking (see LMStudioClient.chat).

Everything here is async (aiohttp), so a slow 27B model never freezes the Discord connection, and
cancelling a request really does stop LM Studio from generating.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Collection, Sequence

import aiohttp

from textutil import TOOL_BLOCKS, strip_reasoning, strip_tool_markup
from tooling import Tool, ToolResult, default_description

log = logging.getLogger("emilia.llm")

# Without multi-turn mode the model may ask for tools in at most this many consecutive rounds; after
# that it must answer. Every round is a full generation, which is slow on big models, so keep this small.
MAX_TOOL_ROUNDS = 2
MAX_CALLS_PER_ROUND = 2
_FINAL_ANSWER_HINT = "(No more tools are available. Write your final answer now.)"

# Multi-turn mode: after using a tool the model keeps getting turns until it calls stop_talking. While
# it is working every request requires a tool call, so it can't keep answering with plain text (which
# models tend to do by repeating themselves); it ends by calling stop_talking, with any last words as text.
STOP_TALKING = "stop_talking"
_STOP_TALKING_SCHEMA = {
    "name": STOP_TALKING,
    "description": (
        "Stop talking: your reply is finished and enough has been said. Call this as soon as you have answered "
        "and there is no more tool work to do. Anything you write alongside this call is posted as your last message."
    ),
    "parameters": {"type": "object", "properties": {}},
}
CONTINUE_HINT = (
    "(Note from the system, not from anyone in the chat: everything you wrote so far has already been posted. "
    "If the request is done, call stop_talking now; you don't need to say anything else. Only use another tool "
    "if there is more work to do. Never repeat or sum up what you already said.)"
)
_REPEATED_CALL = (
    "This is exactly the same as your previous tool call, so it was not run again: the result above still "
    "stands. If the request is done, call stop_talking now."
)
_LAST_TURN_HINT = "(This is your last turn for this message. No more tools are available, so write your final message now.)"


def _repeats(text: str, earlier: Sequence[str]) -> bool:
    """Whether `text` says again what the model already said during this reply. Models asked to carry
    on after they have finished tend to post their last message again instead of calling stop_talking."""
    def norm(value: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())

    new = norm(text)
    for old in map(norm, earlier):
        if new == old or (min(len(new), len(old)) >= 20 and (new in old or old in new)):
            return True
        if difflib.SequenceMatcher(None, new, old).ratio() >= 0.8:
            return True
    return False


class LLMError(Exception):
    """Talking to LM Studio failed. The message is meant for logs, not for chat users."""


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"LM Studio answered HTTP {status}: {body[:300]!r}")
        self.status = status
        self.body = body


@dataclass
class ChatResult:
    text: str  # the last thing the model wrote; earlier messages went to on_text
    sources: list[str] = field(default_factory=list)  # URLs returned by tools along the way
    used_tools: bool = False


@dataclass
class ToolUse:
    name: str
    arguments: dict[str, Any]
    description: str = ""  # the line shown in the chat, from the tool's describe()
    sources: list[str] = field(default_factory=list)  # filled in once the tool has run
    quiet: bool = False  # the tool's effect shows by itself (a reaction): no line is posted for it


TextHandler = Callable[[str], Awaitable[None]]
ToolHandler = Callable[[ToolUse], Awaitable[None]]


@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str = ""  # set when the model's arguments couldn't be understood


def _make_call(call_id: Any, name: Any, raw_args: Any, fallback_id: str) -> _ToolCall:
    """Normalise one tool call, whether it came from the structured field or from inline text."""
    error = ""
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except ValueError:
            parsed, error = {}, "the arguments were not valid JSON"
        if isinstance(parsed, dict):
            args = parsed
        else:
            args, error = {}, error or "the arguments must be a JSON object"
    else:
        args = {}
    return _ToolCall(id=str(call_id or fallback_id), name=str(name or ""), arguments=args, error=error)


def _structured_calls(message: dict[str, Any], round_no: int) -> list[_ToolCall]:
    calls = []
    for index, item in enumerate(message.get("tool_calls") or []):
        if isinstance(item, dict):
            function = item.get("function") or {}
            calls.append(
                _make_call(item.get("id"), function.get("name"), function.get("arguments"), f"call_{round_no}_{index}")
            )
    return calls


def parse_inline_calls(content: str, known_tools: set[str], round_no: int = 0) -> list[_ToolCall]:
    """Rescue tool calls a model printed as text (<tool_call>{...}</tool_call>, [TOOL_REQUEST]...).

    LM Studio normally converts these into `tool_calls`; when the model's formatting is slightly
    off it leaves them in `content`, and we'd otherwise post raw JSON into the chat.
    """
    candidates = [match for pattern in TOOL_BLOCKS for match in pattern.findall(content)]
    bare = content.strip()
    if not candidates and bare.startswith("{") and bare.endswith("}"):
        candidates.append(bare)

    calls = []
    for index, raw in enumerate(candidates):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        name = obj.get("name") if isinstance(obj, dict) else None
        if not isinstance(name, str) or name not in known_tools:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append(_make_call(None, name, args, f"call_{round_no}_{index}"))
    return calls


class LMStudioClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str = "",
        api_key: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300.0,
        max_concurrent: int = 1,
        tools: Sequence[Tool] = (),
        multi_turn: bool = False,
        max_turns: int = 8,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}
        self._multi_turn = multi_turn
        self._max_turns = max(2, max_turns)  # generations per message in multi-turn mode; the last one must answer
        self._tool_choice_ok = True  # cleared if LM Studio rejects tool_choice="required"
        self._slots = asyncio.Semaphore(max_concurrent)  # limits simultaneous generations
        self._session: aiohttp.ClientSession | None = None

    # --- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        """Open the HTTP session and check LM Studio is reachable. Never raises: if LM Studio isn't
        up yet, requests will simply retry the connection when messages arrive."""
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout), headers=headers)
        try:
            self._pick_model(await self.list_models())
        except LLMError as exc:
            log.warning("Could not reach LM Studio at %s yet (%s). Will retry when a message arrives.", self._base_url, exc)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/models", timeout=10)
        entries = data.get("data", []) if isinstance(data, dict) else []
        return [entry["id"] for entry in entries if isinstance(entry, dict) and entry.get("id")]

    def _pick_model(self, models: list[str]) -> None:
        if self.model:
            if models and self.model not in models:
                log.warning(
                    "Model %r is not in LM Studio's list (%s). Check LMSTUDIO_MODEL.", self.model, ", ".join(models[:8])
                )
            else:
                log.info("Using LM Studio model %s", self.model)
            return
        chat_models = [name for name in models if "embed" not in name.lower()]
        if not chat_models:
            raise LLMError("LM Studio lists no chat models; load one or set LMSTUDIO_MODEL")
        self.model = chat_models[0]
        log.info("LMSTUDIO_MODEL is not set; using %s", self.model)

    # --- chat ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        on_text: TextHandler | None = None,
        on_tool: ToolHandler | None = None,
        on_tool_done: ToolHandler | None = None,
        *,
        exclude: Collection[str] = (),
        tools: bool = True,
    ) -> ChatResult:
        """Run a conversation to completion, executing any tool calls the model makes on the way.
        With `tools=False` the model is offered none, however many are configured (used for the messages
        she starts herself, where nobody has asked for anything to be searched, read or run).

        As it goes, whatever the model writes before its last message is handed to `on_text`, and every
        tool call to `on_tool` just before the tool runs (and to `on_tool_done` after, with its sources),
        so the caller can post them as they happen. Without multi-turn mode the model gets MAX_TOOL_ROUNDS rounds of tools and then
        must answer. In multi-turn mode a tool call starts "working": the model is prompted again after
        every generation (text or tools) until it calls stop_talking or max_turns runs out. stop_talking is
        only offered while working, so it can't end a reply before it has started. Tools named in
        `exclude` are neither offered nor run for this conversation.
        """
        available = {name: tool for name, tool in self._tools.items() if name not in exclude} if tools else {}
        convo = list(messages)
        sources: list[str] = []
        used_ids: set[str] = set()
        used_tools = False
        working = False
        said: list[str] = []  # what the model wrote so far in this reply, to catch it repeating itself
        previous_calls: list[str] = []  # the last round's tool calls, to catch it repeating them
        repeated_rounds = 0
        tools_enabled = bool(available)
        limit = self._max_turns if self._multi_turn else MAX_TOOL_ROUNDS + 1

        for round_no in range(limit):
            last = round_no + 1 >= limit
            offer = tools_enabled and not last  # the last round must be an answer
            offered = (set(available) | ({STOP_TALKING} if working else set())) if offer else set()
            try:
                message = await self._complete(convo, offer, finish_option=working, require_tool=working, tools=available)
            except LLMHTTPError as exc:
                if not (offer and exc.status in (400, 422)):
                    raise
                log.warning("LM Studio rejected a request that offered tools (%s); retrying without tools.", exc)
                tools_enabled = offer = False
                offered = set()
                message = await self._complete(convo, False)

            content = message.get("content")
            content = content if isinstance(content, str) else ""
            text = strip_tool_markup(strip_reasoning(content))
            echoed = working and bool(text) and _repeats(text, said)
            if echoed:
                log.info("The model repeated an earlier message; not posting it again")
                text = ""

            calls: list[_ToolCall] = []
            if offer:
                calls = _structured_calls(message, round_no) or parse_inline_calls(content, offered, round_no)
            if not calls:
                # Without tools the reply ends here, unless it is working and has something new to say;
                # nothing new, or no turn left after this one, means it has finished.
                if not working or not text or round_no + 2 >= limit:
                    if working:
                        log.info("Turn over: %s", "out of turns (MAX_TOOL_TURNS)" if text else "the model had nothing new to say")
                    return ChatResult(text, sources, used_tools)
                if on_text is not None:
                    await on_text(text)
                said.append(text)
                convo.append({"role": "assistant", "content": text})
                convo.append({"role": "user", "content": CONTINUE_HINT})
                continue

            calls = calls[:MAX_CALLS_PER_ROUND]
            for index, call in enumerate(calls):
                if call.id in used_ids:  # some servers restart their numbering with every response
                    call.id = f"call_{round_no}_{index}"
                used_ids.add(call.id)
            convo.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in calls
                    ],
                }
            )
            if round_no + 2 >= limit:
                hint = _LAST_TURN_HINT if self._multi_turn else _FINAL_ANSWER_HINT
            else:
                hint = CONTINUE_HINT if self._multi_turn else ""
            if working and all(call.name == STOP_TALKING for call in calls):
                log.info("Model called stop_talking: turn over")
                return ChatResult(text, sources, used_tools)  # what it wrote alongside stop_talking is its last message
            keys = [f"{c.name}:{json.dumps(c.arguments, sort_keys=True)}" for c in calls if not (working and c.name == STOP_TALKING)]
            repeated = bool(keys) and keys == previous_calls  # back to back only: "run tests, edit, run tests" is fine
            previous_calls = keys
            # A repeated tool call, or a repeated message next to a new one, is a strike; two in a row end
            # the turn. Models stuck in a loop repeat their message and only reword their tool calls slightly.
            repeated_rounds = repeated_rounds + 1 if repeated or echoed else 0
            if repeated_rounds >= 2:
                log.info("Turn over: the model kept repeating itself")
                return ChatResult("", sources, used_tools)
            if text:
                said.append(text)
                if on_text is not None:
                    await on_text(text)
            finished = False
            for call in calls:
                log.info("Model called %s(%s)", call.name, json.dumps(call.arguments)[:200])
                if call.name == STOP_TALKING and working:
                    finished = True
                    result = ToolResult("Done.")
                elif repeated:
                    log.info("Not running %s again: same call as the round before", call.name)
                    result = ToolResult(_REPEATED_CALL)
                elif call.name not in offered:  # unknown or excluded: refused, and not shown in the chat as if it ran
                    result = await self._run_tool(call, offered)
                else:
                    use = ToolUse(call.name, call.arguments, self._describe(call), quiet=bool(getattr(self._tools[call.name], "quiet", False)))
                    if on_tool is not None:
                        await on_tool(use)
                    result = await self._run_tool(call, offered)
                    use.sources = list(result.sources)
                    if on_tool_done is not None:
                        await on_tool_done(use)
                    used_tools = True
                sources.extend(result.sources)
                note = f"\n\n{hint}" if hint and call is calls[-1] else ""
                convo.append({"role": "tool", "tool_call_id": call.id, "content": result.text + note})
            working = working or (self._multi_turn and any(call.name in available for call in calls))
            if finished:
                log.info("Model called stop_talking: turn over")
                return ChatResult("", sources, used_tools)

        raise LLMError("the tool loop ended without an answer")  # unreachable: the last round offers no tools

    def _describe(self, call: _ToolCall) -> str:
        describe = getattr(self._tools.get(call.name), "describe", None)
        if describe is not None:
            try:
                return describe(call.arguments)
            except Exception:
                log.exception("Could not describe a call to %s", call.name)
        return default_description(call.name, call.arguments)

    async def _run_tool(self, call: _ToolCall, offered: set[str]) -> ToolResult:
        tool = self._tools.get(call.name) if call.name in offered else None
        if tool is None:
            return ToolResult(f"Error: there is no tool called {call.name!r}.")
        if call.error:
            return ToolResult(f"Error: could not read the arguments for {call.name} ({call.error}).")
        try:
            return await tool.run(call.arguments)
        except Exception:
            log.exception("Tool %s crashed", call.name)
            return ToolResult(f"Error: the {call.name} tool failed.")

    async def _complete(
        self,
        convo: list[dict[str, Any]],
        offer_tools: bool,
        finish_option: bool = False,
        require_tool: bool = False,
        tools: dict[str, Tool] | None = None,
    ) -> dict[str, Any]:
        """One request to /chat/completions; returns the assistant message. With require_tool the model
        must answer with a tool call (tool_choice="required"), if the server supports it. `tools` are the
        ones to offer (all of them by default)."""
        tools = self._tools if tools is None else tools
        if not self.model:  # LM Studio wasn't reachable at startup
            self._pick_model(await self.list_models())

        payload: dict[str, Any] = {"model": self.model, "messages": convo, "stream": False}
        # Only sent when configured, so LM Studio's own UI settings apply otherwise.
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if offer_tools and tools:
            schemas = [tool.schema for tool in tools.values()] + ([_STOP_TALKING_SCHEMA] if finish_option else [])
            payload["tools"] = [{"type": "function", "function": schema} for schema in schemas]
            if require_tool and self._tool_choice_ok:
                payload["tool_choice"] = "required"

        async with self._slots:
            try:
                data = await self._request("POST", "/chat/completions", payload)
            except LLMHTTPError as exc:
                if "tool_choice" not in payload or exc.status not in (400, 422):
                    raise
                log.warning("LM Studio rejected tool_choice=required (%s); not sending it again.", exc)
                self._tool_choice_ok = False
                del payload["tool_choice"]
                data = await self._request("POST", "/chat/completions", payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response from LM Studio: {str(data)[:300]}") from None
        if not isinstance(message, dict):
            raise LLMError(f"unexpected message from LM Studio: {str(message)[:300]}")
        return message

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        if self._session is None:
            raise LLMError("the LM Studio client has not been started")
        url = f"{self._base_url}{path}"
        options: dict[str, Any] = {}
        if timeout is not None:  # never pass timeout=None: aiohttp reads that as "no timeout at all"
            options["timeout"] = aiohttp.ClientTimeout(total=timeout)
        try:
            async with self._session.request(method, url, json=payload, **options) as response:
                body = (await response.read()).decode("utf-8", errors="replace")
                status = response.status
        except asyncio.TimeoutError:
            raise LLMError(f"timed out waiting for LM Studio ({url})") from None
        except aiohttp.ClientError as exc:
            raise LLMError(f"cannot reach LM Studio at {url}: {exc}") from None
        if status >= 400:
            raise LLMHTTPError(status, body)
        try:
            return json.loads(body)
        except ValueError:
            raise LLMError(f"LM Studio sent invalid JSON: {body[:200]!r}") from None
