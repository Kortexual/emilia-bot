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

import os
import sqlite3
import tempfile
import unittest

from memory import MemoryStore, Presence


class PeopleDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:", max_turns=3)
        self.addCleanup(self.store.close)

    def test_names_are_collected_per_user_and_deduplicated_ignoring_case(self):
        self.store.observe_users([("1", [("display", "Bob"), ("username", "bobby_99"), ("global", "")])])
        self.store.observe_users([("1", [("display", "bob"), ("nick", "Robert")])])
        names = self.store.aliases(["1"])["1"]
        self.assertEqual(sorted(n.lower() for n in names), ["bob", "bobby_99", "robert"])
        self.assertEqual(self.store.aliases(["2"]), {})

    def test_only_the_newest_names_are_kept(self):
        for i in range(20):
            self.store.observe_users([("1", [("display", f"name{i:02d}")])])
        names = self.store.aliases(["1"])["1"]
        self.assertEqual(len(names), 12)
        self.assertIn("name19", names)
        self.assertNotIn("name00", names)

    def test_a_note_can_be_saved_replaced_and_removed(self):
        self.assertEqual(self.store.abouts(["1"]), {})
        self.store.set_about("1", "gardener")
        self.store.set_about("2", "likes tea")
        self.assertEqual(self.store.abouts(["1", "2", "3"]), {"1": "gardener", "2": "likes tea"})
        self.store.set_about("1", "")
        self.assertEqual(self.store.abouts(["1"]), {})

    def test_forgetting_a_user_removes_their_profile_and_detaches_their_messages(self):
        self.store.observe_users([("1", [("display", "Alice")]), ("2", [("display", "Bob")])])
        self.store.set_about("1", "gardener")
        self.store.add_turn("c1", "Alice", "1", "hi", "hello")
        self.store.add_turn("c1", "Bob", "2", "hey", "hello Bob")
        self.assertTrue(self.store.forget_user("1"))
        self.assertEqual(self.store.aliases(["1"]), {})
        self.assertEqual(self.store.abouts(["1"]), {})
        users = [(m.author, m.author_id) for m in self.store.recent("c1") if m.role == "user"]
        self.assertEqual(users, [("someone", ""), ("Bob", "2")])
        self.assertEqual(self.store.aliases(["2"])["2"], ["Bob"])  # others are untouched
        self.assertFalse(self.store.forget_user("1"))  # nothing left to remove

    def test_find_by_alias_matches_name_starts_ignoring_case(self):
        self.store.observe_users([("1", [("display", "Jack P."), ("username", "jackp")]), ("2", [("display", "Jackson")]), ("3", [("display", "Bob")])])
        self.assertEqual(sorted(self.store.find_by_alias("JACK")), ["1", "2"])
        self.assertEqual(self.store.find_by_alias("jackp"), ["1"])
        self.assertEqual(self.store.find_by_alias("nobody"), [])
        self.assertEqual(self.store.find_by_alias("  "), [])

    def test_find_by_alias_treats_wildcards_as_plain_characters(self):
        self.store.observe_users([("1", [("display", "Jack")]), ("2", [("display", "a_b")])])
        self.assertEqual(self.store.find_by_alias("%"), [])
        self.assertEqual(self.store.find_by_alias("_"), [])
        self.assertEqual(self.store.find_by_alias("a_"), ["2"])

    def test_backfill_attaches_ids_only_to_older_messages_that_have_none(self):
        self.store.close()
        self.store = MemoryStore(":memory:", max_turns=10)  # roomy enough that nothing is trimmed away
        self.addCleanup(self.store.close)
        self.store.add_turn("c1", "Jack Larpel", "", "old", "r1")
        self.store._db.execute("INSERT INTO messages (channel_id, role, author, content) VALUES ('c1', 'user', 'jack larpel', 'older')")
        self.store.add_turn("c1", "Jack Larpel", "5", "has an id already", "r2")
        self.store.add_turn("c1", "Someone Else", "", "other author", "r3")
        self.store.add_turn("c2", "Jack Larpel", "", "other channel", "r4")
        self.assertEqual(self.store.backfill_ids("c1", "77", ["Jack Larpel", "jackpapel"]), 2)
        rows = {m.content: m.author_id for m in self.store.recent("c1") if m.role == "user"}
        self.assertEqual(rows, {"old": "77", "older": "77", "has an id already": "5", "other author": ""})
        self.assertEqual(self.store.recent("c2")[0].author_id, "")
        self.assertEqual(self.store.backfill_ids("c1", "77", []), 0)

    def test_last_replies(self):
        for i in range(4):
            self.store.add_turn("c1", "A", "1", f"q{i}", f"a{i}")
        self.store.add_turn("c2", "A", "1", "other", "elsewhere")
        self.assertEqual(self.store.last_replies("c1", 2), ["a2", "a3"])
        self.assertEqual(self.store.last_replies("c1", 0), [])
        self.assertEqual(self.store.last_replies("nobody", 3), [])


class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:", max_turns=3)
        self.addCleanup(self.store.close)

    def test_nothing_is_known_at_first(self):
        self.assertEqual(self.store.presence("c1"), Presence(None, None, None))

    def test_remembers_when_she_spoke_here_and_anywhere(self):
        self.store.note_sent("c1", 100.0)
        self.store.note_sent("c2", 200.0)
        self.assertEqual(self.store.presence("c1"), Presence(100.0, 200.0, None))
        self.assertEqual(self.store.presence("c2"), Presence(200.0, 200.0, None))
        self.assertEqual(self.store.presence("c3"), Presence(None, 200.0, None))  # never spoke there, but did elsewhere

    def test_an_event_is_remembered_until_the_next_one(self):
        self.store.note_sent("c1", 100.0, event=True)
        self.store.note_sent("c1", 150.0)  # an ordinary message does not erase the last event
        self.assertEqual(self.store.presence("c1"), Presence(150.0, 150.0, 100.0))
        self.store.note_sent("c2", 300.0, event=True)
        self.assertEqual(self.store.presence("c1").event_anywhere, 300.0)

    def test_clearing_a_channel_keeps_the_timing(self):
        self.store.add_turn("c1", "A", "1", "q", "a")
        self.store.note_sent("c1", 100.0)
        self.store.clear("c1")
        self.assertEqual(self.store.presence("c1").sent_here, 100.0)

    def test_it_survives_a_restart_and_an_older_database_gets_the_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "memory.db")
            old = sqlite3.connect(path)
            old.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL, role TEXT NOT NULL,"
                " author TEXT, author_id TEXT, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            old.commit()
            old.close()
            first = MemoryStore(path)
            first.note_sent("c1", 42.0, event=True)
            first.close()
            second = MemoryStore(path)
            try:
                self.assertEqual(second.presence("c1"), Presence(42.0, 42.0, 42.0))
            finally:
                second.close()

    def test_a_message_of_her_own_keeps_the_conversation_alternating(self):
        self.store.add_turn("c1", "A", "1", "hi", "hello")
        self.store.add_turn("c1", "", "", "(placeholder)", "I found a flower!")
        self.store.add_turn("c1", "A", "1", "where?", "In the garden.")
        history = self.store.recent("c1")
        self.assertEqual([m.role for m in history], ["user", "assistant"] * 3)
        self.assertEqual(history[2].author_id, "")


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

    def test_trimming_in_chunks_keeps_the_start_of_the_history_steady(self):
        store = MemoryStore(":memory:", max_turns=6, trim_chunk=2)
        self.addCleanup(store.close)
        starts = []
        for i in range(14):
            store.add_turn("c1", "A", "1", f"q{i}", f"a{i}")
            history = store.recent("c1")
            self.assertEqual(history[0].role, "user")
            self.assertLessEqual(len(history), 12)  # never more than max_turns exchanges
            starts.append(history[0].content)
        self.assertEqual(list(dict.fromkeys(starts)), ["q0", "q3", "q6", "q9"])  # it moves every few messages, not every message

    def test_a_character_limit_forgets_whole_old_exchanges(self):
        store = MemoryStore(":memory:", max_turns=10, max_chars=100)
        self.addCleanup(store.close)
        for i in range(3):
            store.add_turn("c1", "A", "1", f"u{i}" * 10, f"a{i}" * 10)  # 40 characters per exchange
        self.assertEqual([m.content[:2] for m in store.recent("c1")], ["u1", "a1", "u2", "a2"])
        store.add_turn("c2", "A", "1", "u" * 500, "a" * 500)  # one exchange is never forgotten just for being long
        self.assertEqual(len(store._db.execute("SELECT id FROM messages WHERE channel_id = 'c2'").fetchall()), 2)

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
