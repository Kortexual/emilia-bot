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
    """Anything the model may ask for. `schema` is the OpenAI "function" object."""

    name: str
    schema: dict[str, Any]  # {"name": ..., "description": ..., "parameters": {JSON schema}}

    async def run(self, arguments: dict[str, Any]) -> ToolResult: ...
