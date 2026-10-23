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
do_nothing (see LMStudioClient.chat).

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
from typing import Any, Awaitable, Callable, Sequence

import aiohttp

from textutil import TOOL_BLOCKS, strip_reasoning, strip_tool_markup
from tooling import Tool, ToolResult

log = logging.getLogger("emilia.llm")

# Without multi-turn mode the model may ask for tools in at most this many consecutive rounds; after
# that it must answer. Every round is a full generation, which is slow on big models, so keep this small.
MAX_TOOL_ROUNDS = 2
MAX_CALLS_PER_ROUND = 2
_FINAL_ANSWER_HINT = "(No more tools are available. Write your final answer now.)"

# Multi-turn mode: after using a tool the model keeps getting turns until it calls do_nothing.
DO_NOTHING = "do_nothing"
_DO_NOTHING_SCHEMA = {
    "name": DO_NOTHING,
    "description": "End your turn. Call this when you have finished working and have nothing more to say.",
    "parameters": {"type": "object", "properties": {}},
}
CONTINUE_HINT = (
    "(Note from the system, not from anyone in the chat: your turn is still going. If you have finished, "
    "call do_nothing now and write nothing. Only write if you have something new to say; never repeat or "
    "sum up what you already said. Anything you write is posted in the chat right away.)"
)
_LAST_TURN_HINT = "(This is your last turn for this message. No more tools are available, so write your final message now.)"


def _repeats(text: str, earlier: Sequence[str]) -> bool:
    """Whether `text` says again what the model already said during this reply. Models asked to carry
    on after they have finished tend to post their last message again instead of calling do_nothing."""
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
    text: str  # the last thing the model wrote; earlier messages went to on_step
    sources: list[str] = field(default_factory=list)  # URLs returned by tools along the way
    used_tools: bool = False


@dataclass
class ToolUse:
    name: str
    arguments: dict[str, Any]


@dataclass
class Step:
    """One generation that was not the last: what the model wrote, and the tools it ran (already finished)."""

    text: str
    tools: list[ToolUse] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


StepHandler = Callable[[Step], Awaitable[None]]


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
    async def chat(self, messages: list[dict[str, Any]], on_step: StepHandler | None = None) -> ChatResult:
        """Run a conversation to completion, executing any tool calls the model makes on the way.

        Every generation except the last is handed to `on_step` once its tools have run, so the caller
        can post it. Without multi-turn mode the model gets MAX_TOOL_ROUNDS rounds of tools and then
        must answer. In multi-turn mode a tool call starts "working": the model is prompted again after
        every generation (text or tools) until it calls do_nothing or max_turns runs out. do_nothing is
        only offered while working, so it can't end a reply before it has started.
        """
        convo = list(messages)
        sources: list[str] = []
        used_ids: set[str] = set()
        used_tools = False
        working = False
        said: list[str] = []  # what the model wrote so far in this reply, to catch it repeating itself
        tools_enabled = bool(self._tools)
        limit = self._max_turns if self._multi_turn else MAX_TOOL_ROUNDS + 1

        for round_no in range(limit):
            last = round_no + 1 >= limit
            offer = tools_enabled and not last  # the last round must be an answer
            offered = (set(self._tools) | ({DO_NOTHING} if working else set())) if offer else set()
            try:
                message = await self._complete(convo, offer, finish_option=working)
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
            if working and text and _repeats(text, said):
                log.info("The model repeated itself; not posting it again")
                text = ""

            calls: list[_ToolCall] = []
            if offer:
                calls = _structured_calls(message, round_no) or parse_inline_calls(content, offered, round_no)
            if not calls:
                # Without tools the reply ends here, unless it is working and has something new to say;
                # nothing new, or no turn left after this one, means it has finished.
                if not working or not text or round_no + 2 >= limit:
                    return ChatResult(text, sources, used_tools)
                if on_step is not None:
                    await on_step(Step(text))
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
            step = Step(text)
            if text:
                said.append(text)
            finished = False
            for call in calls:
                log.info("Model called %s(%s)", call.name, json.dumps(call.arguments)[:200])
                if call.name == DO_NOTHING and working:
                    finished = True
                    result = ToolResult("Done.")
                else:
                    result = await self._run_tool(call, offered)
                    step.tools.append(ToolUse(call.name, call.arguments))
                    step.sources.extend(result.sources)
                sources.extend(result.sources)
                note = f"\n\n{hint}" if hint and call is calls[-1] else ""
                convo.append({"role": "tool", "tool_call_id": call.id, "content": result.text + note})
            used_tools = used_tools or bool(step.tools)
            working = working or (self._multi_turn and any(call.name in self._tools for call in calls))
            if finished and not step.tools:
                return ChatResult(text, sources, used_tools)  # what it wrote alongside do_nothing is its last message
            if on_step is not None:
                await on_step(step)
            if finished:
                return ChatResult("", sources, used_tools)

        raise LLMError("the tool loop ended without an answer")  # unreachable: the last round offers no tools

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

    async def _complete(self, convo: list[dict[str, Any]], offer_tools: bool, finish_option: bool = False) -> dict[str, Any]:
        """One request to /chat/completions; returns the assistant message."""
        if not self.model:  # LM Studio wasn't reachable at startup
            self._pick_model(await self.list_models())

        payload: dict[str, Any] = {"model": self.model, "messages": convo, "stream": False}
        # Only sent when configured, so LM Studio's own UI settings apply otherwise.
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if offer_tools and self._tools:
            schemas = [tool.schema for tool in self._tools.values()] + ([_DO_NOTHING_SCHEMA] if finish_option else [])
            payload["tools"] = [{"type": "function", "function": schema} for schema in schemas]

        async with self._slots:
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
