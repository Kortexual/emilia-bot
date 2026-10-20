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

"""Persistent conversation memory, kept in SQLite (standard library only).

LLMs are stateless: "memory" means re-sending recent messages with every request. We keep the last
few exchanges per channel on disk so they survive restarts, and cap how much is sent back so a small
context window isn't overflowed.

Only messages addressed to the bot (and its replies) are stored.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    author     TEXT,
    author_id  TEXT,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, id);
"""


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" or "assistant"
    author: str  # speaker's display name (user messages only)
    author_id: str  # speaker's Discord user ID (user messages only), so people can be told apart and pinged
    content: str


class MemoryStore:
    """Thread-safe: methods are blocking but fast, so the bot calls them via asyncio.to_thread."""

    def __init__(self, path: str = "memory.db", max_turns: int = 10) -> None:
        self._max_messages = max(1, max_turns) * 2  # a turn is a user message plus the reply
        self._lock = threading.Lock()
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._db:
            self._db.executescript(_SCHEMA)
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(messages)")}
            if "author_id" not in columns:  # memory.db files from before user IDs were stored
                self._db.execute("ALTER TABLE messages ADD COLUMN author_id TEXT")

    def recent(self, channel_id: str, max_chars: int | None = None) -> list[StoredMessage]:
        """Remembered messages for a channel, oldest first.

        Always starts with a user message and alternates user/assistant (some chat templates
        insist on that). If `max_chars` is given, the oldest exchanges are dropped until the rest fit.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT role, author, author_id, content FROM ("
                "  SELECT id, role, author, author_id, content FROM messages"
                "  WHERE channel_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (channel_id, self._max_messages),
            ).fetchall()

        messages = [
            StoredMessage(role, author or "", author_id or "", content) for role, author, author_id, content in rows
        ]
        total = sum(len(m.content) for m in messages)

        def drop_oldest() -> None:
            nonlocal total
            total -= len(messages.pop(0).content)

        while messages and messages[0].role != "user":
            drop_oldest()
        while messages and max_chars is not None and total > max_chars:
            drop_oldest()
            while messages and messages[0].role != "user":
                drop_oldest()
        return messages

    def add_turn(self, channel_id: str, author: str, author_id: str, user_text: str, reply_text: str) -> None:
        """Store one exchange atomically and trim the channel to the newest `max_turns`."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (channel_id, role, author, author_id, content) VALUES (?, 'user', ?, ?, ?)",
                (channel_id, author, author_id, user_text),
            )
            self._db.execute(
                "INSERT INTO messages (channel_id, role, author, content) VALUES (?, 'assistant', NULL, ?)",
                (channel_id, reply_text),
            )
            self._db.execute(
                "DELETE FROM messages WHERE channel_id = ? AND id NOT IN ("
                "  SELECT id FROM messages WHERE channel_id = ? ORDER BY id DESC LIMIT ?)",
                (channel_id, channel_id, self._max_messages),
            )

    def clear(self, channel_id: str) -> int:
        """Forget a channel. Returns how many messages were removed."""
        with self._lock, self._db:
            return self._db.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,)).rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
