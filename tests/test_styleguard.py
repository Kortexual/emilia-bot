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

import unittest

from memory import StoredMessage
from styleguard import cool_down, drop_stale_directives, is_standing_instruction, ornaments

PAW, BLOSSOM = "\U0001F43E", "\U0001F338"
FAMILY = "\U0001F468\u200d\U0001F469\u200d\U0001F467"
FLAG = "\U0001F1FA\U0001F1F8"


class OrnamentTests(unittest.TestCase):
    def test_finds_emoji_and_flourish_tildes(self):
        self.assertEqual(ornaments(f"Hello {PAW} there~ friend"), {PAW, "~"})

    def test_multi_part_emoji_count_as_one(self):
        self.assertEqual(ornaments(f"{FAMILY} and {FLAG}"), {FAMILY, FLAG})
        self.assertEqual(ornaments("\u2764\ufe0f"), ornaments("\u2764"))  # with and without the emoji selector

    def test_ordinary_symbols_are_not_ornaments(self):
        text = "It is 75\u00b0F, \u00a9 2026, a \u2192 b, ~~struck~~, see https://x.example/~user and ~/notes"
        self.assertEqual(ornaments(text), set())


class CoolDownTests(unittest.TestCase):
    def test_an_emoji_from_a_recent_reply_is_removed(self):
        self.assertEqual(cool_down(f"Sure thing! {PAW}", [f"Hello {PAW}"]), "Sure thing!")

    def test_other_emoji_survive_and_spacing_is_tidied(self):
        out = cool_down(f"Oh, a screen reader! {BLOSSOM} So... {PAW} does it read aloud?", [f"Hi {BLOSSOM}"])
        self.assertEqual(out, f"Oh, a screen reader! So... {PAW} does it read aloud?")

    def test_a_flourish_tilde_is_removed_but_not_strikethrough_or_paths(self):
        self.assertEqual(cool_down("Happy to help~ really", ["Okay~"]), "Happy to help really")
        self.assertEqual(cool_down("~~old~~ and ~/notes", ["Okay~"]), "~~old~~ and ~/notes")

    def test_nothing_recent_means_nothing_changes(self):
        reply = f"Fresh start {PAW}"
        self.assertEqual(cool_down(reply, []), reply)
        self.assertEqual(cool_down(reply, ["plain text only"]), reply)

    def test_a_reply_that_is_only_an_emoji_is_not_emptied(self):
        self.assertEqual(cool_down(PAW, [PAW]), PAW)

    def test_indentation_and_line_breaks_are_kept(self):
        out = cool_down(f"Code:\n    x = 1  {PAW}\n\nDone", [PAW])
        self.assertEqual(out, "Code:\n    x = 1\n\nDone")


class StandingInstructionTests(unittest.TestCase):
    def test_recognises_rules_that_are_meant_to_last(self):
        for text in (
            "Noo when you reply to Jack do it in braille - you can!",
            f"from now on end every message with {PAW}",
            "Okay, got it. Next time he asks me something, I'll write my whole reply in dots.",
            "Always speak like a pirate",
            "for the rest of the chat use lowercase only",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_standing_instruction(text))

    def test_ordinary_chat_is_left_alone(self):
        for text in (
            "I always eat breakfast at seven",
            "can you reply to my question about Warden?",
            "What's the weather in Atlanta today?",
            "Honey never spoils!",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_standing_instruction(text))


def exchange(user, reply):
    return [StoredMessage("user", "A", "1", user), StoredMessage("assistant", "", "", reply)]


class DropStaleDirectivesTests(unittest.TestCase):
    def contents(self, history):
        return [m.content for m in history]

    def test_a_rule_stays_for_one_more_message_then_goes(self):
        rule = exchange("from now on end every message with a paw", f"Okay! {PAW}")
        just_after = rule  # the very next request still sees it
        self.assertEqual(drop_stale_directives(just_after), just_after)

        later = rule + exchange("how are you?", f"Fine {PAW}")
        self.assertEqual(self.contents(drop_stale_directives(later)), ["how are you?", f"Fine {PAW}"])

    def test_the_bots_own_promise_counts_too(self):
        promise = exchange("do the dots please", "Okay, next time I'll write my reply in dots, I promise!")
        later = promise + exchange("hi", "Hello!")
        self.assertEqual(self.contents(drop_stale_directives(later)), ["hi", "Hello!"])

    def test_normal_history_is_untouched_and_alternation_is_kept(self):
        history = exchange("one", "uno") + exchange("from now on say hi", "ok") + exchange("two", "dos") + exchange("three", "tres")
        shown = drop_stale_directives(history)
        self.assertEqual(self.contents(shown), ["one", "uno", "two", "dos", "three", "tres"])
        self.assertEqual([m.role for m in shown], ["user", "assistant"] * 3)

    def test_keep_recent_is_adjustable(self):
        history = exchange("from now on say hi", "ok") + exchange("two", "dos")
        self.assertEqual(len(drop_stale_directives(history, keep_recent=2)), 4)
        self.assertEqual(len(drop_stale_directives(history, keep_recent=1)), 2)

    def test_empty_history(self):
        self.assertEqual(drop_stale_directives([]), [])


if __name__ == "__main__":
    unittest.main()
