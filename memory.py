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
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, id);
"""


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" or "assistant"
    author: str  # speaker's display name (user messages only)
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

    def recent(self, channel_id: str, max_chars: int | None = None) -> list[StoredMessage]:
        """Remembered messages for a channel, oldest first.

        Always starts with a user message and alternates user/assistant (some chat templates
        insist on that). If `max_chars` is given, the oldest exchanges are dropped until the rest fit.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT role, author, content FROM ("
                "  SELECT id, role, author, content FROM messages"
                "  WHERE channel_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (channel_id, self._max_messages),
            ).fetchall()

        messages = [StoredMessage(role, author or "", content) for role, author, content in rows]
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

    def add_turn(self, channel_id: str, author: str, user_text: str, reply_text: str) -> None:
        """Store one exchange atomically and trim the channel to the newest `max_turns`."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (channel_id, role, author, content) VALUES (?, 'user', ?, ?)",
                (channel_id, author, user_text),
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
