# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Emilia: a Discord bot for LM Studio
# Copyright (C) 2026 YOUR NAME
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

"""Text helpers: cleaning what goes into the model, what comes out, and fitting Discord's limits."""
from __future__ import annotations

import re

# Control characters (except tab and newline), plus invisible characters that can hide text from
# human readers: zero-width space, word joiner, BOM, bidi overrides/isolates and Unicode "tag"
# characters. Zero-width joiners are deliberately kept so multi-part emoji still render.
#
# (The original sanitize() used `c not in "\x00-\x1f\x7f"`, which is a *substring* test, not a
# range: it deleted every "-" from messages and let most control characters through.)
_INVISIBLE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f\u200b\u2060\ufeff\u202a-\u202e\u2066-\u2069\U000e0000-\U000e007f]"
)

# Tags the prompt format relies on. Users (and web pages) must not be able to forge them.
_RESERVED_TAGS = re.compile(r"</?\s*(?:USER_INPUT|SYSTEM_PROMPT)\b[^>]*>", re.IGNORECASE)

# Tool-call text that some models print instead of using the structured tool_calls field.
# Group 1 is the JSON payload.
TOOL_BLOCKS = (
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[TOOL_REQUEST\]\s*(.*?)\s*\[END_TOOL_REQUEST\]", re.DOTALL | re.IGNORECASE),
)

# [label](https://url): a masked link hides where it really goes. Web snippets are untrusted, so a
# prompt-injected page could try to smuggle a deceptive one into the bot's reply.
_MASKED_LINK = re.compile(r"\[([^\]\n]{1,200})\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_UNFINISHED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


# --- Input ------------------------------------------------------------------
def strip_invisible(text: str) -> str:
    return _INVISIBLE.sub("", text)


def strip_reserved_tags(text: str) -> str:
    return _RESERVED_TAGS.sub("", text)


def sanitize_user_text(text: str, max_chars: int = 2048) -> str:
    """Make untrusted chat text safe to put in a prompt.

    Removes control/invisible characters and forged prompt tags, tidies runs of spaces inside each
    line (keeping indentation, so pasted code survives) and caps the length.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_reserved_tags(strip_invisible(text))

    lines = []
    for line in text.split("\n"):
        body = line.lstrip(" \t")
        indent = line[: len(line) - len(body)].expandtabs(4)
        lines.append(indent + " ".join(body.split()))
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars].rstrip()


# --- Output -----------------------------------------------------------------
def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models leave in the reply."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_CLOSE.split(text)[-1]  # a lone </think>: everything before it was reasoning
    text = _THINK_UNFINISHED.sub("", text)  # reasoning that never finished (ran out of tokens)
    return text.strip()


def strip_tool_markup(text: str) -> str:
    for pattern in TOOL_BLOCKS:
        text = pattern.sub("", text)
    return text.strip()


def clean_reply(text: str, speaker_names: tuple[str, ...] = ()) -> str:
    """Tidy a model reply before it is posted or stored."""
    text = strip_reserved_tags(strip_tool_markup(strip_reasoning(text)))
    text = _MASKED_LINK.sub(r"\1 (<\2>)", text)  # show the real URL; <> also stops a big link preview
    for name in filter(None, speaker_names):
        # Models sometimes imitate the "Name: message" layout of the chat history.
        text = re.sub(rf"^\s*{re.escape(name)}\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


_WORD = re.compile(r"[a-z0-9']+")


class LeakDetector:
    """Spots replies that repeat a long stretch of the system prompt word for word.

    Looking for verbatim runs (rather than keywords like "system prompt") means normal replies that
    merely mention such phrases are never blocked, while an actual dump of the prompt is caught.
    The window is long enough that a character describing itself in similar words doesn't trip it.
    """

    def __init__(self, secret_text: str, window: int = 12) -> None:
        words = _WORD.findall(secret_text.lower())
        self._window = window
        self._grams = {tuple(words[i : i + window]) for i in range(len(words) - window + 1)}

    def leaks(self, reply: str) -> bool:
        words = _WORD.findall(reply.lower())
        size = self._window
        return any(tuple(words[i : i + size]) in self._grams for i in range(len(words) - size + 1))


# --- Discord message limits ---------------------------------------------------
_FENCE_CLOSE = "\n```"


def _find_cut(text: str, budget: int) -> tuple[int, int]:
    """Pick where to break `text`: returns (end, start) so the chunk is text[:end] and the next one
    begins at text[start:]. Prefers paragraph breaks, then lines, then sentences, then words."""
    window = text[:budget]
    for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
        idx = window.rfind(sep)
        if idx > budget // 2:  # don't produce a tiny first chunk
            keep = len(sep.rstrip())  # keep the full stop, drop the whitespace
            return idx + keep, idx + len(sep)
    return budget, budget


def split_message(text: str, limit: int = 1900) -> list[str]:
    """Split text into chunks of at most `limit` characters (Discord's cap is 2000).

    A ``` code block that straddles a boundary is closed at the end of one chunk and re-opened at
    the start of the next, so formatting doesn't leak across messages.
    """
    text = text.strip()
    chunks: list[str] = []
    fence: str | None = None  # opening line of a code block still open at the start of the next chunk
    rest = text
    while rest:
        prefix = f"{fence}\n" if fence else ""
        if len(prefix) + len(rest) <= limit:
            chunks.append(prefix + rest)
            break
        budget = max(limit - len(prefix) - len(_FENCE_CLOSE), 1)
        end, start = _find_cut(rest, budget)
        piece, rest = rest[:end].rstrip(), rest[start:].lstrip("\n")

        state = fence
        for line in piece.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                state = None if state else stripped
        if piece.strip():
            chunks.append(prefix + piece + (_FENCE_CLOSE if state else ""))
        fence = state
    return chunks
