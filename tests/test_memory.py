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

import os
import sqlite3
import tempfile
import unittest

from memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:", max_turns=3)
        self.addCleanup(self.store.close)

    def test_returns_exchanges_oldest_first_with_authors(self):
        self.store.add_turn("c1", "Alice", "1", "hi", "hello Alice")
        self.store.add_turn("c1", "Bob", "2", "hey", "hello Bob")
        history = self.store.recent("c1")
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m.content for m in history], ["hi", "hello Alice", "hey", "hello Bob"])
        self.assertEqual([m.author for m in history if m.role == "user"], ["Alice", "Bob"])
        self.assertEqual([m.author_id for m in history if m.role == "user"], ["1", "2"])

    def test_keeps_only_the_newest_turns(self):
        for i in range(6):
            self.store.add_turn("c1", "A", "1", f"q{i}", f"a{i}")
        contents = [m.content for m in self.store.recent("c1")]
        self.assertEqual(contents, ["q3", "a3", "q4", "a4", "q5", "a5"])

    def test_channels_are_independent(self):
        self.store.add_turn("c1", "A", "1", "one", "uno")
        self.store.add_turn("c2", "B", "2", "two", "dos")
        self.assertEqual([m.content for m in self.store.recent("c1")], ["one", "uno"])
        self.assertEqual([m.content for m in self.store.recent("c2")], ["two", "dos"])
        self.assertEqual(self.store.recent("nobody"), [])

    def test_character_budget_drops_whole_oldest_exchanges(self):
        for i in range(3):
            self.store.add_turn("c1", "A", "1", f"u{i}" * 10, f"a{i}" * 10)  # 40 characters per exchange
        self.assertEqual(len(self.store.recent("c1", max_chars=100)), 4)  # newest two exchanges
        self.assertEqual(len(self.store.recent("c1", max_chars=1000)), 6)
        self.assertEqual(self.store.recent("c1", max_chars=30), [])  # not even one exchange fits

    def test_history_never_starts_with_an_assistant_message(self):
        # Simulate a hand-edited / damaged database containing an orphaned reply.
        self.store._db.execute("INSERT INTO messages (channel_id, role, content) VALUES ('c1', 'assistant', 'orphan')")
        self.store.add_turn("c1", "A", "1", "q", "a")
        history = self.store.recent("c1")
        self.assertEqual(history[0].role, "user")
        self.assertEqual([m.content for m in history], ["q", "a"])

    def test_clear_forgets_only_that_channel(self):
        self.store.add_turn("c1", "A", "1", "q", "a")
        self.store.add_turn("c2", "A", "1", "q", "a")
        self.assertEqual(self.store.clear("c1"), 2)
        self.assertEqual(self.store.recent("c1"), [])
        self.assertEqual(len(self.store.recent("c2")), 2)

    def test_memory_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "memory.db")  # missing folder is created
            first = MemoryStore(path, max_turns=5)
            first.add_turn("c1", "Alice", "1", "remember me", "I will")
            first.close()

            second = MemoryStore(path, max_turns=5)
            try:
                self.assertEqual([m.content for m in second.recent("c1")], ["remember me", "I will"])
            finally:
                second.close()

    def test_a_database_from_before_user_ids_keeps_working(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "memory.db")
            old = sqlite3.connect(path)
            old.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL, role TEXT NOT NULL,"
                " author TEXT, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            old.execute("INSERT INTO messages (channel_id, role, author, content) VALUES ('c1', 'user', 'Alice', 'old')")
            old.commit()
            old.close()

            store = MemoryStore(path, max_turns=5)
            try:
                store.add_turn("c1", "Bob", "2", "new", "reply")
                self.assertEqual([(m.author, m.author_id) for m in store.recent("c1") if m.role == "user"], [("Alice", ""), ("Bob", "2")])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
