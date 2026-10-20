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

How web search works: the model is offered a `web_search` tool. If it decides it is unsure, LM
Studio answers with a `tool_calls` entry instead of text; we run the search ourselves, hand the
results back as a `tool` message and ask again. The model decides when to search, not a keyword list.

Everything here is async (aiohttp), so a slow 27B model never freezes the Discord connection, and
cancelling a request really does stop LM Studio from generating.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import aiohttp

from textutil import TOOL_BLOCKS, strip_reasoning, strip_tool_markup
from tooling import Tool, ToolResult

log = logging.getLogger("emilia.llm")

# The model may ask for tools in at most this many consecutive rounds; after that it must answer.
# Every round is a full generation, which is slow on big models, so keep this small.
MAX_TOOL_ROUNDS = 2
MAX_CALLS_PER_ROUND = 2
_FINAL_ANSWER_HINT = "\n\n(No more searches are available. Write your final answer now.)"


class LLMError(Exception):
    """Talking to LM Studio failed. The message is meant for logs, not for chat users."""


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"LM Studio answered HTTP {status}: {body[:300]!r}")
        self.status = status
        self.body = body


@dataclass
class ChatResult:
    text: str
    sources: list[str] = field(default_factory=list)  # URLs returned by tools along the way
    used_tools: bool = False


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}
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
    async def chat(self, messages: list[dict[str, Any]]) -> ChatResult:
        """Run a conversation to completion, executing any tool calls the model makes on the way."""
        convo = list(messages)
        sources: list[str] = []
        used_ids: set[str] = set()
        used_tools = False
        tools_enabled = bool(self._tools)

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            offer = tools_enabled and round_no < MAX_TOOL_ROUNDS  # the last round must be an answer
            try:
                message = await self._complete(convo, offer)
            except LLMHTTPError as exc:
                if not (offer and exc.status in (400, 422)):
                    raise
                log.warning("LM Studio rejected a request that offered tools (%s); retrying without web search.", exc)
                tools_enabled = offer = False
                message = await self._complete(convo, False)

            content = message.get("content")
            content = content if isinstance(content, str) else ""

            calls: list[_ToolCall] = []
            if offer:
                calls = _structured_calls(message, round_no) or parse_inline_calls(content, set(self._tools), round_no)
            if not calls:
                return ChatResult(strip_tool_markup(strip_reasoning(content)), sources, used_tools)

            calls = calls[:MAX_CALLS_PER_ROUND]
            for index, call in enumerate(calls):
                if call.id in used_ids:  # some servers restart their numbering with every response
                    call.id = f"call_{round_no}_{index}"
                used_ids.add(call.id)
            convo.append(
                {
                    "role": "assistant",
                    "content": strip_tool_markup(strip_reasoning(content)),
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in calls
                    ],
                }
            )
            next_round_is_final = round_no + 1 >= MAX_TOOL_ROUNDS
            for call in calls:
                log.info("Model called %s(%s)", call.name, json.dumps(call.arguments)[:200])
                result = await self._run_tool(call)
                sources.extend(result.sources)
                text = result.text + (_FINAL_ANSWER_HINT if next_round_is_final else "")
                convo.append({"role": "tool", "tool_call_id": call.id, "content": text})
            used_tools = True

        raise LLMError("the tool loop ended without an answer")  # unreachable: the last round offers no tools

    async def _run_tool(self, call: _ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"Error: there is no tool called {call.name!r}.")
        if call.error:
            return ToolResult(f"Error: could not read the arguments for {call.name} ({call.error}).")
        try:
            return await tool.run(call.arguments)
        except Exception:
            log.exception("Tool %s crashed", call.name)
            return ToolResult(f"Error: the {call.name} tool failed.")

    async def _complete(self, convo: list[dict[str, Any]], offer_tools: bool) -> dict[str, Any]:
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
            payload["tools"] = [{"type": "function", "function": tool.schema} for tool in self._tools.values()]

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
