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

"""The minimal tool interface shared by the LLM client and the tools themselves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    """What a tool hands back: text for the model, plus URLs worth showing to the user."""

    text: str
    sources: list[str] = field(default_factory=list)


class Tool(Protocol):
    """Anything the model may ask for. `schema` is the OpenAI "function" object.

    A tool may also define `describe(arguments) -> str`: the short line posted in the chat when the
    model uses it, e.g. "💻 Running `ls -la`". Tools without one get default_description()."""

    name: str
    schema: dict[str, Any]  # {"name": ..., "description": ..., "parameters": {JSON schema}}

    async def run(self, arguments: dict[str, Any]) -> ToolResult: ...


def code_span(value: Any, max_chars: int = 60) -> str:
    """`value` as one short line of inline code. Backticks are swapped out so model-written arguments
    can't break out of the span; "" for nothing."""
    text = " ".join(str(value).split()).replace("`", "'")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return f"`{text}`" if text else ""


def default_description(name: str, arguments: dict[str, Any]) -> str:
    """For tools without describe(): the name and its first argument, e.g. "🛠️ `lookup: tea`"."""
    detail = next((v for k, v in arguments.items() if k != "confirm" and isinstance(v, str) and v.strip()), "")
    return "🛠️ " + code_span(f"{name}: {detail}" if detail.strip() else name)
