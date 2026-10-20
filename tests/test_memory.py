import os
import tempfile
import unittest

from memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:", max_turns=3)
        self.addCleanup(self.store.close)

    def test_returns_exchanges_oldest_first_with_authors(self):
        self.store.add_turn("c1", "Alice", "hi", "hello Alice")
        self.store.add_turn("c1", "Bob", "hey", "hello Bob")
        history = self.store.recent("c1")
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m.content for m in history], ["hi", "hello Alice", "hey", "hello Bob"])
        self.assertEqual([m.author for m in history if m.role == "user"], ["Alice", "Bob"])

    def test_keeps_only_the_newest_turns(self):
        for i in range(6):
            self.store.add_turn("c1", "A", f"q{i}", f"a{i}")
        contents = [m.content for m in self.store.recent("c1")]
        self.assertEqual(contents, ["q3", "a3", "q4", "a4", "q5", "a5"])

    def test_channels_are_independent(self):
        self.store.add_turn("c1", "A", "one", "uno")
        self.store.add_turn("c2", "B", "two", "dos")
        self.assertEqual([m.content for m in self.store.recent("c1")], ["one", "uno"])
        self.assertEqual([m.content for m in self.store.recent("c2")], ["two", "dos"])
        self.assertEqual(self.store.recent("nobody"), [])

    def test_character_budget_drops_whole_oldest_exchanges(self):
        for i in range(3):
            self.store.add_turn("c1", "A", f"u{i}" * 10, f"a{i}" * 10)  # 40 characters per exchange
        self.assertEqual(len(self.store.recent("c1", max_chars=100)), 4)  # newest two exchanges
        self.assertEqual(len(self.store.recent("c1", max_chars=1000)), 6)
        self.assertEqual(self.store.recent("c1", max_chars=30), [])  # not even one exchange fits

    def test_history_never_starts_with_an_assistant_message(self):
        # Simulate a hand-edited / damaged database containing an orphaned reply.
        self.store._db.execute("INSERT INTO messages (channel_id, role, content) VALUES ('c1', 'assistant', 'orphan')")
        self.store.add_turn("c1", "A", "q", "a")
        history = self.store.recent("c1")
        self.assertEqual(history[0].role, "user")
        self.assertEqual([m.content for m in history], ["q", "a"])

    def test_clear_forgets_only_that_channel(self):
        self.store.add_turn("c1", "A", "q", "a")
        self.store.add_turn("c2", "A", "q", "a")
        self.assertEqual(self.store.clear("c1"), 2)
        self.assertEqual(self.store.recent("c1"), [])
        self.assertEqual(len(self.store.recent("c2")), 2)

    def test_memory_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "memory.db")  # missing folder is created
            first = MemoryStore(path, max_turns=5)
            first.add_turn("c1", "Alice", "remember me", "I will")
            first.close()

            second = MemoryStore(path, max_turns=5)
            try:
                self.assertEqual([m.content for m in second.recent("c1")], ["remember me", "I will"])
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
