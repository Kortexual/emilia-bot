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

"""Images and files attached to Discord messages, turned into something the model can use.

Every attachment becomes a short note in the sender's message ("[image: cat.png]"), so the model knows
it is there. With VISION on, images also go to the model as OpenAI-style `image_url` parts (this needs a
vision model). With READ_FILES on, text files (code, logs, notes...) are downloaded and their contents
put into the message under their note. Anything else (PDFs, audio, archives) is only named.

The contents are for the request at hand only: memory keeps just the notes, so a picture or a long file
doesn't crowd the conversation out of the prompt. Downloads come from Discord's own CDN through
discord.py's session, so they never block the event loop.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Iterable

import aiohttp
import discord

from textutil import sanitize_user_text, strip_cipher_script, strip_invisible

log = logging.getLogger("emilia.attachments")

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}  # what vision models served by LM Studio accept
MAX_IMAGES = 4  # per message, across the message itself, the one it replies to and a forward
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024  # a text file bigger than this isn't downloaded at all
NAME_CHARS = 80

# Text files that don't announce themselves as text/*.
TEXT_TYPES = {
    "application/json", "application/xml", "application/javascript", "application/x-javascript",
    "application/yaml", "application/x-yaml", "application/toml", "application/x-sh", "application/sql",
    "application/x-python", "application/ld+json",
}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env", ".xml", ".html", ".htm", ".css", ".scss", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".py",
    ".rb", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".swift", ".php", ".lua",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".sql", ".r", ".tex", ".diff", ".patch", ".srt", ".vtt",
}


def safe_filename(name: str) -> str:
    """A file name that can't break out of its note or forge markup."""
    cleaned = strip_cipher_script(strip_invisible(name or ""))
    for bad in '<>"[]\\`\n\r\t':
        cleaned = cleaned.replace(bad, "")
    return " ".join(cleaned.split())[:NAME_CHARS] or "file"


def content_type(attachment: Any) -> str:
    """The attachment's media type without parameters, guessed from its name when Discord doesn't say."""
    raw = getattr(attachment, "content_type", None) or mimetypes.guess_type(attachment.filename)[0] or ""
    return raw.split(";")[0].strip().lower()


def is_text(attachment: Any) -> bool:
    kind = content_type(attachment)
    extension = os.path.splitext(attachment.filename)[1].lower()
    return kind.startswith("text/") or kind in TEXT_TYPES or extension in TEXT_EXTENSIONS


def human_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size // 1024} KB"
    return f"{size} bytes"


class AttachmentReader:
    """Collects the attachments for one reply: notes to remember, what to show in the prompt, and images.

    `add` can be called for several messages (the one being answered, the one it replies to, a forward);
    the image count and the text budget are shared between them."""

    def __init__(self, *, vision: bool, read_files: bool, max_chars: int) -> None:
        self.vision = vision
        self.read_files = read_files
        self.notes: list[str] = []  # one per attachment, e.g. "[image: cat.png]"; this is what memory keeps
        self.shown: list[str] = []  # the same, with a text file's contents under its note; this is what the model reads
        self.images: list[str] = []  # data: URLs, sent as image parts
        self._chars_left = max_chars

    def __bool__(self) -> bool:
        return bool(self.notes)

    def stored_text(self) -> str:
        return "\n".join(self.notes)

    def prompt_text(self) -> str:
        return "\n".join(self.shown)

    async def add(self, attachments: Iterable[Any], where: str = "") -> None:
        """Read `attachments`; `where` says whose they are in the note, e.g. " in the message replied to"."""
        for attachment in attachments or ():
            if content_type(attachment).startswith("image/"):
                await self._add_image(attachment, where)
            elif is_text(attachment):
                await self._add_text(attachment, where)
            else:
                self._note(f"[file{where}: {safe_filename(attachment.filename)}, {self._describe(attachment)}, not opened]")

    def _note(self, note: str, shown: str | None = None) -> None:
        self.notes.append(note)
        self.shown.append(note if shown is None else shown)

    @staticmethod
    def _describe(attachment: Any) -> str:
        return f"{content_type(attachment) or 'unknown type'}, {human_size(attachment.size)}"

    async def _add_image(self, attachment: Any, where: str) -> None:
        name = safe_filename(attachment.filename)
        kind = content_type(attachment)
        if not self.vision:
            self._note(f"[image{where}: {name}]")
            return
        if kind not in IMAGE_TYPES:
            reason = "not a format you can view"
        elif attachment.size > MAX_IMAGE_BYTES:
            reason = "too big"
        elif len(self.images) >= MAX_IMAGES:
            reason = "too many images"
        else:
            data = await self._download(attachment)
            if data is not None and len(data) <= MAX_IMAGE_BYTES:
                self.images.append(f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}")
                self._note(f"[image{where}: {name}]")
                return
            reason = "could not be loaded"
        self._note(f"[image{where}: {name}, not shown: {reason}]")

    async def _add_text(self, attachment: Any, where: str) -> None:
        name = safe_filename(attachment.filename)
        note = f"[file{where}: {name}]"
        if not self.read_files or self._chars_left <= 0 or attachment.size > MAX_FILE_BYTES:
            reason = "not opened" if not self.read_files else "too long to read"
            self._note(f"[file{where}: {name}, {self._describe(attachment)}, {reason}]")
            return
        data = await self._download(attachment)
        if data is None or b"\x00" in data:  # NUL bytes: binary after all
            self._note(f"[file{where}: {name}, {self._describe(attachment)}, not opened]")
            return
        raw = data.decode("utf-8", errors="replace")
        full = sanitize_user_text(raw, len(raw))  # untrusted: no forged tags or cipher script
        text = full[: self._chars_left].rstrip()
        cut = "\n[the rest of the file was cut off]" if len(full) > self._chars_left else ""
        self._chars_left -= len(text)
        self._note(note, f"{note}\n```\n{text}\n```{cut}")

    @staticmethod
    async def _download(attachment: Any) -> bytes | None:
        try:
            return await attachment.read()
        except (discord.HTTPException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.info("Could not download attachment %r (%s)", getattr(attachment, "filename", "?"), exc)
            return None


def has_images(messages: list[dict[str, Any]]) -> bool:
    return any(isinstance(m.get("content"), list) for m in messages)


def drop_images(messages: list[dict[str, Any]]) -> None:
    """Replace every message that carries images with a text-only copy, in place. Used when the server
    refuses them (the model loaded in LM Studio can't see), so the reply still happens."""
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            note = "(The images attached to this message could not be shown to you: you can't see images right now.)"
            messages[index] = {**message, "content": f"{text}\n{note}".strip()}
