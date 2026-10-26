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

"""The logic behind her day: the clock, active hours, how likely she is to speak up, and scenes."""
import random
import unittest
from datetime import datetime, time, timedelta, timezone

import prompts
import presence
from presence import (
    Scene,
    check_seconds,
    clock_text,
    is_active,
    local_now,
    parse_clock,
    phase_of,
    scene_for,
    send_chance,
)

UTC = timezone.utc


class AlwaysLuckyRNG(random.Random):
    """Every roll succeeds; sampling still works."""

    def random(self):
        return 0.0


class NeverLuckyRNG(random.Random):
    def random(self):
        return 0.999


class ParseClockTests(unittest.TestCase):
    def test_accepts_24_hour_and_12_hour_forms(self):
        cases = {
            "08:00": time(8, 0),
            "8:00": time(8, 0),
            "8:00AM": time(8, 0),
            "8:00 am": time(8, 0),
            "10:00PM": time(22, 0),
            "10 pm": time(22, 0),
            "10p.m.": time(22, 0),
            "22:15": time(22, 15),
            "12:00AM": time(0, 0),
            "12:30pm": time(12, 30),
            "12:00PM": time(12, 0),
            "  7:05 AM  ": time(7, 5),
            "0:00": time(0, 0),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_clock(text), expected)

    def test_rejects_anything_else(self):
        for text in ("", "8", "25:00", "8:60", "13pm", "0am", "ab", "8:00 xm", "8:00:00", "-1:00", "noon"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_clock(text)


class ActiveHoursTests(unittest.TestCase):
    def test_a_daytime_window_includes_its_start_but_not_its_end(self):
        start, end = time(8, 0), time(22, 0)
        self.assertTrue(is_active(time(8, 0), start, end))
        self.assertTrue(is_active(time(15, 30), start, end))
        self.assertTrue(is_active(time(21, 59), start, end))
        self.assertFalse(is_active(time(22, 0), start, end))
        self.assertFalse(is_active(time(7, 59), start, end))
        self.assertFalse(is_active(time(0, 0), start, end))

    def test_a_window_may_run_past_midnight(self):
        start, end = time(20, 0), time(2, 0)
        self.assertTrue(is_active(time(20, 0), start, end))
        self.assertTrue(is_active(time(23, 59), start, end))
        self.assertTrue(is_active(time(0, 0), start, end))
        self.assertTrue(is_active(time(1, 59), start, end))
        self.assertFalse(is_active(time(2, 0), start, end))
        self.assertFalse(is_active(time(12, 0), start, end))


class PhaseTests(unittest.TestCase):
    def test_every_hour_has_a_phase_with_text_and_ideas(self):
        seen = {phase_of(hour) for hour in range(24)}
        self.assertEqual(seen, {phase for _, phase in presence.PHASES})
        for phase in seen:
            self.assertIn(phase, prompts.PHASE_LABELS)
            self.assertGreaterEqual(len(prompts.EVENT_IDEAS[phase]), presence.IDEAS_PER_EVENT)

    def test_boundaries(self):
        expected = {
            0: "late_night", 4: "late_night", 5: "early_morning", 9: "early_morning", 10: "late_morning",
            11: "late_morning", 12: "midday", 13: "midday", 14: "afternoon", 17: "afternoon", 18: "evening",
            21: "evening", 22: "night", 23: "night",
        }
        for hour, phase in expected.items():
            with self.subTest(hour=hour):
                self.assertEqual(phase_of(hour), phase)


class ClockTextTests(unittest.TestCase):
    def test_reads_like_a_person_would_say_it(self):
        self.assertEqual(clock_text(datetime(2026, 10, 23, 7, 5, tzinfo=UTC)), "7:05 AM on Friday")
        self.assertEqual(clock_text(datetime(2026, 10, 23, 0, 30, tzinfo=UTC)), "12:30 AM on Friday")
        self.assertEqual(clock_text(datetime(2026, 10, 23, 12, 0, tzinfo=UTC)), "12:00 PM on Friday")
        self.assertEqual(clock_text(datetime(2026, 10, 24, 22, 45, tzinfo=UTC)), "10:45 PM on Saturday")

    def test_local_now_uses_the_zone_and_the_given_clock(self):
        stamp = datetime(2026, 10, 23, 12, 0, tzinfo=UTC).timestamp()
        tokyo = presence.load_timezone("Asia/Tokyo")
        self.assertEqual(local_now(tokyo, clock=lambda: stamp).hour, 21)
        self.assertEqual(local_now(None, clock=lambda: stamp).timestamp(), stamp)
        self.assertIsNotNone(local_now(None, clock=lambda: stamp).tzinfo)
        self.assertIsNone(presence.load_timezone(""))


class SendChanceTests(unittest.TestCase):
    MIN, FULL = 3600, 8 * 3600

    @staticmethod
    def gaps(min_seconds, full_seconds, runs=300, seed=3):
        """How long she stays quiet, in seconds, over many simulated silences checked as the bot checks them."""
        rng = random.Random(seed)
        interval = check_seconds(full_seconds)
        gaps = []
        for _ in range(runs):
            elapsed = 0.0
            while True:
                elapsed += interval
                if rng.random() < send_chance(elapsed, min_seconds, full_seconds, interval):
                    break
            gaps.append(elapsed)
        return sorted(gaps)

    def test_she_never_speaks_up_before_the_minimum_wait(self):
        self.assertEqual(send_chance(0, self.MIN, self.FULL), 0.0)
        self.assertEqual(send_chance(self.MIN - 1, self.MIN, self.FULL), 0.0)

    def test_the_longer_since_her_last_message_the_likelier(self):
        chances = [send_chance(hours * 3600, self.MIN, self.FULL) for hours in (1, 2, 3, 4, 5, 6, 7, 8)]
        self.assertEqual(chances, sorted(chances))
        self.assertEqual(len(set(chances)), len(chances))  # strictly rising, not flat
        self.assertEqual(chances[0], 0.0)
        self.assertLess(chances[1], chances[-1] / 4)  # low at first, then it climbs

    def test_it_levels_off_at_the_full_wait(self):
        self.assertEqual(send_chance(self.FULL, self.MIN, self.FULL), send_chance(100 * 3600, self.MIN, self.FULL))
        self.assertTrue(0.2 < send_chance(self.FULL, self.MIN, self.FULL) < 0.3)  # about one check in four at the defaults
        self.assertTrue(0 < send_chance(10, 0, 0) <= 1)  # a degenerate range does not divide by zero

    def test_the_chance_per_check_follows_how_often_she_checks(self):
        elapsed = 5 * 3600
        often, rarely = send_chance(elapsed, self.MIN, self.FULL, 60), send_chance(elapsed, self.MIN, self.FULL, 300)
        self.assertLess(often, rarely)
        self.assertAlmostEqual(1 - (1 - often) ** 5, rarely)  # five short checks add up to one long one

    def test_checks_scale_with_the_wait(self):
        self.assertEqual(check_seconds(8 * 3600), presence.CHECK_SECONDS)  # the defaults: every 5 minutes
        self.assertEqual(check_seconds(48 * 3600), presence.CHECK_SECONDS)  # never rarer
        self.assertEqual(check_seconds(30 * 60), 30 * 60 / presence.CHECKS_PER_WAIT)
        self.assertEqual(check_seconds(60), presence.MIN_CHECK_SECONDS)  # never more often

    def test_over_a_long_silence_she_does_speak_eventually(self):
        gaps = [gap / 3600 for gap in self.gaps(self.MIN, self.FULL)]
        self.assertGreaterEqual(gaps[0], 1)  # never before the minimum
        self.assertLess(gaps[-1], 12)  # and never a very long silence either
        self.assertTrue(3 < gaps[len(gaps) // 2] < 5, gaps[len(gaps) // 2])

    def test_short_waits_have_the_same_shape(self):
        gaps = [gap / 60 for gap in self.gaps(5 * 60, 15 * 60)]  # every 5 to 15 minutes
        self.assertGreaterEqual(gaps[0], 5)
        self.assertLess(gaps[-1], 20)
        self.assertTrue(8 < gaps[len(gaps) // 2] < 11, gaps[len(gaps) // 2])  # like 4 hours out of 1 to 8


class SceneTests(unittest.TestCase):
    def scene(self, now, last_sent=None, last_event=None, rng=None, cooldown=120, **kwargs):
        return scene_for(
            now,
            None if last_sent is None else last_sent.timestamp(),
            None if last_event is None else last_event.timestamp(),
            ideas=prompts.EVENT_IDEAS,
            event_cooldown_minutes=cooldown,
            rng=rng or NeverLuckyRNG(),
            **kwargs,
        )

    def test_the_first_message_of_an_early_morning_is_a_wake_up(self):
        now = datetime(2026, 10, 23, 7, 42, tzinfo=UTC)
        for last in (None, now - timedelta(hours=9), now - timedelta(days=3)):
            with self.subTest(last=last):
                scene = self.scene(now, last)
                self.assertEqual((scene.kind, scene.phase, scene.clock), ("wake_up", "early_morning", "7:42 AM on Friday"))

    def test_a_second_message_the_same_morning_is_not(self):
        now = datetime(2026, 10, 23, 8, 10, tzinfo=UTC)
        self.assertEqual(self.scene(now, now - timedelta(minutes=20)).kind, "")

    def test_the_first_message_of_a_day_later_than_the_morning_is_not_a_wake_up(self):
        now = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)
        self.assertEqual(self.scene(now, None).kind, "")

    def test_the_day_is_the_local_day_not_a_24_hour_gap(self):
        late = datetime(2026, 10, 22, 23, 50, tzinfo=UTC)
        now = datetime(2026, 10, 23, 0, 20, tzinfo=UTC)  # 30 minutes later, but a new day (and the middle of the night)
        self.assertEqual(self.scene(now, late).kind, "")  # not early morning yet, so no wake-up
        morning = datetime(2026, 10, 23, 5, 5, tzinfo=UTC)
        self.assertEqual(self.scene(morning, late).kind, "wake_up")

    def test_a_random_event_needs_luck_and_a_finished_cooldown(self):
        now = datetime(2026, 10, 23, 13, 0, tzinfo=UTC)
        lucky = AlwaysLuckyRNG()
        event = self.scene(now, now - timedelta(minutes=5), None, lucky)
        self.assertEqual(event.kind, "event")
        self.assertEqual(len(event.ideas), presence.IDEAS_PER_EVENT)
        self.assertTrue(set(event.ideas) <= set(prompts.EVENT_IDEAS["midday"]))
        self.assertEqual(self.scene(now, now - timedelta(minutes=5), None, NeverLuckyRNG()).kind, "")
        recent = now - timedelta(minutes=119)
        self.assertEqual(self.scene(now, recent, recent, lucky).kind, "")
        old = now - timedelta(minutes=120)
        self.assertEqual(self.scene(now, old, old, lucky).kind, "event")

    def test_a_cooldown_of_zero_turns_random_events_off(self):
        now = datetime(2026, 10, 23, 13, 0, tzinfo=UTC)
        self.assertEqual(self.scene(now, None, None, AlwaysLuckyRNG(), cooldown=0).kind, "")

    def test_the_ideas_vary_from_one_event_to_the_next(self):
        now = datetime(2026, 10, 23, 16, 0, tzinfo=UTC)
        rng = random.Random(5)
        picks = {self.scene(now, None, None, rng, event_chance=1.0).ideas for _ in range(20)}
        self.assertGreater(len(picks), 5)

    def test_a_message_she_starts_herself_always_has_something_going_on(self):
        now = datetime(2026, 10, 23, 16, 0, tzinfo=UTC)
        recent = now - timedelta(minutes=1)
        scene = self.scene(now, recent, recent, NeverLuckyRNG(), cooldown=0, spontaneous=True)
        self.assertEqual(scene.kind, "event")
        self.assertTrue(scene.ideas)

    def test_a_spontaneous_message_in_the_early_morning_is_still_a_wake_up(self):
        now = datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
        self.assertEqual(self.scene(now, now - timedelta(hours=10), spontaneous=True).kind, "wake_up")


class NoteTextTests(unittest.TestCase):
    def test_every_scene_kind_and_phase_renders(self):
        for phase in prompts.PHASE_LABELS:
            for kind in ("", "wake_up", "event"):
                scene = Scene("1:00 PM on Friday", phase, kind, ("reading", "napping", "baking") if kind == "event" else ())
                for text in (prompts.time_note(scene), prompts.unprompted_note(scene)):
                    with self.subTest(phase=phase, kind=kind):
                        self.assertIn("1:00 PM on Friday", text)
                        self.assertIn(prompts.PHASE_LABELS[phase], text)

    def test_the_late_hours_make_her_sleepy_and_the_day_does_not(self):
        late = prompts.time_note(Scene("2:00 AM on Friday", "late_night"))
        self.assertIn("drowsy", late)
        self.assertIn("sleepy", prompts.time_note(Scene("11:00 PM on Friday", "night")))
        self.assertNotIn("sleepy", prompts.time_note(Scene("1:00 PM on Friday", "midday")))
        self.assertNotIn("drowsy", prompts.time_note(Scene("1:00 PM on Friday", "midday")))

    def test_a_wake_up_asks_for_a_yawn_and_a_good_morning_and_an_event_gives_examples(self):
        wake = prompts.time_note(Scene("7:00 AM on Friday", "early_morning", "wake_up"))
        self.assertIn("woken up", wake)
        self.assertIn("good morning", wake)
        event = prompts.time_note(Scene("1:00 PM on Friday", "midday", "event", ("eating lunch", "reading", "napping")))
        self.assertIn("eating lunch; reading; napping", event)
        self.assertIn("respond to what they said", event)

    def test_a_message_of_her_own_does_not_talk_about_replying(self):
        note = prompts.unprompted_note(Scene("1:00 PM on Friday", "midday", "event", ("eating lunch", "reading", "napping")))
        self.assertIn("nobody has spoken to you", note)
        self.assertIn("don't ping anyone", note)
        self.assertNotIn("message above", note)

    def test_the_new_rules_are_part_of_the_leak_check(self):
        self.assertTrue(prompts.LEAK_DETECTOR.leaks(prompts.TIME_RULES))


if __name__ == "__main__":
    unittest.main()
