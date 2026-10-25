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

"""Persistent conversation memory, kept in SQLite (standard library only).

LLMs are stateless: "memory" means re-sending recent messages with every request. We keep the last
few exchanges per channel on disk so they survive restarts, and cap how much is sent back so a small
context window isn't overflowed.

Only messages addressed to the bot (and its replies) are stored, plus a small directory of the people
involved in them: user ID, the names they were seen under, and an optional note they wrote themselves.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

-- Who people are: one row per Discord user ID, the names they have been seen under, and an optional
-- note they wrote about themselves with "!about".
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    about      TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_aliases (
    user_id   TEXT NOT NULL,
    alias_key TEXT NOT NULL,   -- lower-case, so "Bob" and "bob" are one alias
    alias     TEXT NOT NULL,
    kind      TEXT NOT NULL,   -- display (nickname or display name), username, global or nick
    last_seen REAL NOT NULL,
    PRIMARY KEY (user_id, alias_key)
);

-- When the bot last spoke in each channel (Unix time), and when it last had a "scene" (a wake-up or a
-- random event, see presence.py). Timing only: no message text is kept here.
CREATE TABLE IF NOT EXISTS presence (
    channel_id TEXT PRIMARY KEY,
    last_sent  REAL,
    last_event REAL
);
"""

MAX_ALIASES_PER_USER = 12  # the newest names are kept; Discord names change, old ones go stale


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" or "assistant"
    author: str  # speaker's display name (user messages only)
    author_id: str  # speaker's Discord user ID (user messages only), so people can be told apart and pinged
    content: str


@dataclass(frozen=True)
class Presence:
    """When the bot last spoke (Unix time; None = never): in one channel, and anywhere. Also the last scene."""

    sent_here: float | None
    sent_anywhere: float | None
    event_anywhere: float | None


class MemoryStore:
    """Thread-safe: methods are blocking but fast, so the bot calls them via asyncio.to_thread.

    A channel keeps at most `max_turns` exchanges (and, if `max_chars` is given, that much text). By default
    the oldest exchange is forgotten as each new one arrives, so the start of the remembered conversation
    moves with every message. With `trim_chunk` the store forgets `trim_chunk` exchanges more than it has to
    each time it is full, so the start stays put for several messages in a row. That matters for speed: the
    model server can only reuse its processed prompt while the start of the prompt is unchanged.
    """

    def __init__(self, path: str = "memory.db", max_turns: int = 10, *, max_chars: int | None = None, trim_chunk: int = 0) -> None:
        self._max_messages = max(1, max_turns) * 2  # a turn is a user message plus the reply
        self._max_chars = max_chars
        self._trim_chunk = max(0, trim_chunk)
        self._lock = threading.Lock()
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")  # in WAL mode: no disk flush on every commit, and still safe if the bot crashes
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
        """Store one exchange atomically, then forget the channel's oldest exchanges if it holds too many."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (channel_id, role, author, author_id, content) VALUES (?, 'user', ?, ?, ?)",
                (channel_id, author, author_id, user_text),
            )
            self._db.execute(
                "INSERT INTO messages (channel_id, role, author, content) VALUES (?, 'assistant', NULL, ?)",
                (channel_id, reply_text),
            )
            self._trim(channel_id)

    def _trim(self, channel_id: str) -> None:
        """Forget the oldest exchanges of a channel once it holds more than `max_turns` (or `max_chars`).
        It is then cut back by `trim_chunk` exchanges more than needed. Call it inside the write transaction."""
        rows = self._db.execute(
            "SELECT id, LENGTH(content) FROM messages WHERE channel_id = ? ORDER BY id DESC", (channel_id,)
        ).fetchall()  # newest first
        if len(rows) <= self._max_messages and (self._max_chars is None or sum(size for _, size in rows) <= self._max_chars):
            return
        target = max(2, self._max_messages - 2 * self._trim_chunk)
        allowed = None if self._max_chars is None else self._max_chars * target // self._max_messages
        keep = chars = 0
        for _, size in rows:
            if keep >= target or (allowed is not None and keep >= 2 and chars + size > allowed):
                break
            keep += 1
            chars += size
        keep = max(2, keep - keep % 2)  # whole exchanges: the oldest message kept is a user message
        if keep < len(rows):
            self._db.execute("DELETE FROM messages WHERE channel_id = ? AND id < ?", (channel_id, rows[keep - 1][0]))

    def last_replies(self, channel_id: str, count: int) -> list[str]:
        """The bot's own last `count` replies in a channel, oldest first."""
        if count <= 0:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT content FROM messages WHERE channel_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT ?",
                (channel_id, count),
            ).fetchall()
        return [row[0] for row in reversed(rows)]

    # --- when she last spoke ---------------------------------------------------------
    def presence(self, channel_id: str) -> Presence:
        with self._lock:
            row = self._db.execute("SELECT last_sent FROM presence WHERE channel_id = ?", (channel_id,)).fetchone()
            anywhere = self._db.execute("SELECT MAX(last_sent), MAX(last_event) FROM presence").fetchone()
        return Presence(row[0] if row else None, anywhere[0], anywhere[1])

    def note_sent(self, channel_id: str, when: float, event: bool = False) -> None:
        """Record that the bot spoke in a channel at `when`, and that it was a scene if `event` is true."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO presence (channel_id, last_sent, last_event) VALUES (?, ?, ?)"
                " ON CONFLICT(channel_id) DO UPDATE SET last_sent = excluded.last_sent,"
                " last_event = COALESCE(excluded.last_event, last_event)",
                (channel_id, when, when if event else None),
            )

    # --- people ------------------------------------------------------------------
    def observe_users(self, people: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> None:
        """Record who was involved in a message: (user_id, [(kind, name), ...]) for each person."""
        now = time.time()
        with self._lock, self._db:
            for user_id, names in people:
                self._db.execute(
                    "INSERT INTO users (user_id, first_seen, last_seen) VALUES (?, ?, ?)"
                    " ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen",
                    (user_id, now, now),
                )
                for kind, name in names:
                    name = name.strip()
                    if not name:
                        continue
                    self._db.execute(
                        "INSERT INTO user_aliases (user_id, alias_key, alias, kind, last_seen) VALUES (?, ?, ?, ?, ?)"
                        " ON CONFLICT(user_id, alias_key) DO UPDATE SET"
                        " alias = excluded.alias, kind = excluded.kind, last_seen = excluded.last_seen",
                        (user_id, name.lower(), name, kind, now),
                    )
                self._db.execute(
                    "DELETE FROM user_aliases WHERE user_id = ? AND alias_key NOT IN ("
                    "  SELECT alias_key FROM user_aliases WHERE user_id = ? ORDER BY last_seen DESC, alias_key LIMIT ?)",
                    (user_id, user_id, MAX_ALIASES_PER_USER),
                )

    def aliases(self, user_ids: Sequence[str]) -> dict[str, list[str]]:
        """Every name known for each of the given users, newest first."""
        found: dict[str, list[str]] = {}
        with self._lock:
            for user_id in user_ids:
                rows = self._db.execute(
                    "SELECT alias FROM user_aliases WHERE user_id = ? ORDER BY last_seen DESC, alias_key", (user_id,)
                ).fetchall()
                if rows:
                    found[user_id] = [row[0] for row in rows]
        return found

    def backfill_ids(self, channel_id: str, user_id: str, names: Sequence[str]) -> int:
        """Attach a real user ID to this channel's older user messages that have none (saved before IDs
        were recorded) and whose author name is one of `names`. Returns how many were updated."""
        keys = sorted({name.strip().lower() for name in names if name and name.strip()})
        if not keys:
            return 0
        marks = ",".join("?" for _ in keys)
        with self._lock, self._db:
            return self._db.execute(
                "UPDATE messages SET author_id = ? WHERE channel_id = ? AND role = 'user'"
                f" AND (author_id IS NULL OR author_id = '') AND lower(author) IN ({marks})",
                (user_id, channel_id, *keys),
            ).rowcount

    def find_by_alias(self, prefix: str, limit: int = 20) -> list[str]:
        """User IDs with a saved name that starts with `prefix` (ignoring case), newest first. A starting
        point only: the caller still checks the person is real and where they are."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            rows = self._db.execute(
                "SELECT user_id FROM user_aliases WHERE alias_key LIKE ? ESCAPE '\\'"
                " GROUP BY user_id ORDER BY MAX(last_seen) DESC LIMIT ?",
                (escaped + "%", limit),
            ).fetchall()
        return [row[0] for row in rows]

    def set_about(self, user_id: str, text: str) -> None:
        """Save (or, with an empty string, remove) the note a person wrote about themselves."""
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO users (user_id, about, first_seen, last_seen) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET about = excluded.about, last_seen = excluded.last_seen",
                (user_id, text, now, now),
            )

    def abouts(self, user_ids: Sequence[str]) -> dict[str, str]:
        """The notes people wrote about themselves, for those who have one."""
        found: dict[str, str] = {}
        with self._lock:
            for user_id in user_ids:
                row = self._db.execute("SELECT about FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if row and row[0]:
                    found[user_id] = row[0]
        return found

    def forget_user(self, user_id: str) -> bool:
        """Delete a person's note and saved names, and detach them from the stored messages (the text of
        those messages stays until it ages out of the window or the channel is cleared). True if anything
        was removed."""
        with self._lock, self._db:
            removed = self._db.execute("DELETE FROM users WHERE user_id = ?", (user_id,)).rowcount
            removed += self._db.execute("DELETE FROM user_aliases WHERE user_id = ?", (user_id,)).rowcount
            removed += self._db.execute(
                "UPDATE messages SET author = 'someone', author_id = NULL WHERE author_id = ?", (user_id,)
            ).rowcount
        return removed > 0

    def clear(self, channel_id: str) -> int:
        """Forget a channel. Returns how many messages were removed."""
        with self._lock, self._db:
            return self._db.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,)).rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
