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
from datetime import time
from unittest import mock

import config
from config import ConfigError, Settings


def load(**env):
    # Don't let a real .env file in the project folder leak into the tests.
    with mock.patch("config.load_dotenv"), mock.patch.dict("os.environ", env, clear=True):
        return Settings.from_env()


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = load(DISCORD_TOKEN="abc")
        self.assertEqual(s.discord_token, "abc")
        self.assertEqual(s.lmstudio_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(s.lmstudio_model, "")
        self.assertIsNone(s.temperature)  # left to LM Studio's own settings, as before
        self.assertIsNone(s.max_tokens)
        self.assertFalse(s.respond_to_dms)
        self.assertFalse(s.respond_to_bots)
        self.assertEqual(s.bot_reply_limit, 3)
        self.assertTrue(s.web_search)
        self.assertFalse(s.computer_tools)
        self.assertEqual(s.command_timeout, 30.0)
        self.assertFalse(s.computer_allow_host_access)
        self.assertTrue(s.computer_require_confirmation)
        self.assertFalse(s.browser_tools)
        self.assertTrue(s.browser_require_confirmation)
        self.assertFalse(s.multi_turn_tools)
        self.assertEqual(s.max_tool_turns, 8)
        self.assertEqual(s.memory_turns, 10)
        self.assertEqual(s.style_cooldown, 4)
        self.assertTrue(s.forget_standing_instructions)
        self.assertFalse(s.yandere_mode)

    def test_members_intent_flag(self):
        self.assertFalse(load(DISCORD_TOKEN="a").members_intent)
        self.assertTrue(load(DISCORD_TOKEN="a", MEMBERS_INTENT="yes").members_intent)
        with self.assertRaises(ConfigError):
            load(DISCORD_TOKEN="a", MEMBERS_INTENT="maybe")

    def test_yandere_mode_flag(self):
        self.assertTrue(load(DISCORD_TOKEN="a", YANDERE_MODE="true").yandere_mode)
        self.assertFalse(load(DISCORD_TOKEN="a", YANDERE_MODE="off").yandere_mode)
        with self.assertRaises(ConfigError):
            load(DISCORD_TOKEN="a", YANDERE_MODE="sometimes")

    def test_style_settings(self):
        s = load(DISCORD_TOKEN="a", STYLE_COOLDOWN="0", FORGET_STANDING_INSTRUCTIONS="no")
        self.assertEqual(s.style_cooldown, 0)
        self.assertFalse(s.forget_standing_instructions)

    def test_token_is_required(self):
        with self.assertRaises(ConfigError):
            load()

    def test_existing_env_file_variables_keep_working(self):
        s = load(
            DISCORD_TOKEN="abc",
            LMSTUDIO_URL="http://localhost:1235/v1",
            LMSTUDIO_MODEL="huihui-qwen3.8-27b-abliterated",
            LMSTUDIO_API_KEY="",
        )
        self.assertEqual(s.lmstudio_url, "http://localhost:1235/v1")
        self.assertEqual(s.lmstudio_model, "huihui-qwen3.8-27b-abliterated")
        self.assertEqual(s.lmstudio_api_key, "")

    def test_url_is_normalised(self):
        self.assertEqual(load(DISCORD_TOKEN="a", LMSTUDIO_URL="http://localhost:1234").lmstudio_url, "http://localhost:1234/v1")
        self.assertEqual(load(DISCORD_TOKEN="a", LMSTUDIO_URL="http://localhost:1234/v1/").lmstudio_url, "http://localhost:1234/v1")

    def test_placeholder_model_means_auto_detect(self):
        self.assertEqual(load(DISCORD_TOKEN="a", LMSTUDIO_MODEL="your-model-name").lmstudio_model, "")

    def test_flags_and_numbers(self):
        s = load(
            DISCORD_TOKEN="a",
            RESPOND_TO_DMS="Yes",
            RESPOND_TO_BOTS="true",
            BOT_REPLY_LIMIT="0",
            WEB_SEARCH="off",
            LMSTUDIO_TEMPERATURE="0.7",
            LMSTUDIO_MAX_TOKENS="512",
            MEMORY_TURNS="4",
            COMPUTER_TOOLS="yes",
            COMPUTER_WORKSPACE="/tmp/emilia-work",
            COMMAND_TIMEOUT="15",
            COMPUTER_ALLOW_HOST_ACCESS="true",
            COMPUTER_REQUIRE_CONFIRMATION="false",
            BROWSER_TOOLS="on",
            BROWSER_REQUIRE_CONFIRMATION="no",
            MULTI_TURN_TOOLS="true",
            MAX_TOOL_TURNS="4",
        )
        self.assertTrue(s.respond_to_dms)
        self.assertTrue(s.respond_to_bots)
        self.assertEqual(s.bot_reply_limit, 0)  # loop guard off
        self.assertFalse(s.web_search)
        self.assertEqual(s.temperature, 0.7)
        self.assertEqual(s.max_tokens, 512)
        self.assertEqual(s.memory_turns, 4)
        self.assertTrue(s.computer_tools)
        self.assertEqual(s.computer_workspace, "/tmp/emilia-work")
        self.assertEqual(s.command_timeout, 15.0)
        self.assertTrue(s.computer_allow_host_access)
        self.assertFalse(s.computer_require_confirmation)
        self.assertTrue(s.browser_tools)
        self.assertFalse(s.browser_require_confirmation)
        self.assertTrue(s.multi_turn_tools)
        self.assertEqual(s.max_tool_turns, 4)

    def test_bad_values_give_clear_errors(self):
        for bad in (
            {"WEB_SEARCH": "maybe"},
            {"MEMORY_TURNS": "lots"},
            {"MEMORY_TURNS": "0"},
            {"LOG_LEVEL": "LOUD"},
            {"LMSTUDIO_TEMPERATURE": "-1"},
            {"STYLE_COOLDOWN": "-1"},
            {"COMMAND_TIMEOUT": "0"},
            {"MAX_TOOL_TURNS": "1"},
            {"BOT_REPLY_LIMIT": "-1"},
            {"RESPOND_TO_BOTS": "sometimes"},
            {"BROWSER_TOOLS": "sometimes"},
            {"BROWSER_REQUIRE_CONFIRMATION": "2"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(DISCORD_TOKEN="a", **bad)


class DaySettingsTests(unittest.TestCase):
    def test_defaults_leave_everything_off(self):
        s = load(DISCORD_TOKEN="a")
        self.assertFalse(s.time_awareness)
        self.assertFalse(s.random_messages)
        self.assertEqual(s.random_message_channels, ())
        self.assertEqual(s.timezone, "")
        self.assertEqual((s.active_start, s.active_end), (time(8, 0), time(22, 0)))
        self.assertEqual(s.random_event_cooldown_minutes, 120)
        self.assertEqual(s.random_message_min_minutes, 60)
        self.assertEqual(s.random_message_full_hours, 8.0)

    def test_values_are_read(self):
        s = load(
            DISCORD_TOKEN="a",
            TIME_AWARENESS="yes",
            TIMEZONE="America/New_York",
            RANDOM_EVENT_COOLDOWN_MINUTES="45",
            RANDOM_MESSAGES="true",
            RANDOM_MESSAGE_CHANNELS="123, 456 789,123",
            ACTIVE_START="9:30AM",
            ACTIVE_END="11:15 PM",
            RANDOM_MESSAGE_MIN_MINUTES="30",
            RANDOM_MESSAGE_FULL_HOURS="6.5",
        )
        self.assertTrue(s.time_awareness)
        self.assertEqual(s.timezone, "America/New_York")
        self.assertEqual(s.random_event_cooldown_minutes, 45)
        self.assertTrue(s.random_messages)
        self.assertEqual(s.random_message_channels, (123, 456, 789))  # duplicates dropped, order kept
        self.assertEqual((s.active_start, s.active_end), (time(9, 30), time(23, 15)))
        self.assertEqual(s.random_message_min_minutes, 30)
        self.assertEqual(s.random_message_full_hours, 6.5)

    def test_active_hours_may_run_past_midnight(self):
        s = load(DISCORD_TOKEN="a", ACTIVE_START="20:00", ACTIVE_END="02:00")
        self.assertEqual((s.active_start, s.active_end), (time(20, 0), time(2, 0)))

    def test_a_cooldown_of_zero_is_allowed(self):
        self.assertEqual(load(DISCORD_TOKEN="a", RANDOM_EVENT_COOLDOWN_MINUTES="0").random_event_cooldown_minutes, 0)

    def test_bad_values_give_clear_errors(self):
        for bad in (
            {"TIME_AWARENESS": "maybe"},
            {"RANDOM_MESSAGES": "sometimes"},
            {"TIMEZONE": "Mars/Olympus"},
            {"TIMEZONE": "../etc"},
            {"ACTIVE_START": "8"},
            {"ACTIVE_END": "25:00"},
            {"ACTIVE_START": "09:00", "ACTIVE_END": "9:00AM"},  # the same time twice
            {"RANDOM_MESSAGE_CHANNELS": "general"},
            {"RANDOM_MESSAGE_CHANNELS": "0"},
            {"RANDOM_EVENT_COOLDOWN_MINUTES": "-5"},
            {"RANDOM_MESSAGE_MIN_MINUTES": "-1"},
            {"RANDOM_MESSAGE_FULL_HOURS": "0"},
            {"RANDOM_MESSAGE_MIN_MINUTES": "120", "RANDOM_MESSAGE_FULL_HOURS": "2"},  # must be longer than the minimum
            {"RANDOM_MESSAGES": "true"},  # no channels listed
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(DISCORD_TOKEN="a", **bad)


class RestartEnvironmentTests(unittest.TestCase):
    def test_a_restarted_bot_starts_from_the_environment_before_dotenv(self):
        def fake_dotenv():
            import os
            os.environ.setdefault("MULTI_TURN_TOOLS", "false")  # what load_dotenv does: never overrides

        with mock.patch.object(config, "_ENV_BEFORE_DOTENV", None), mock.patch("config.load_dotenv", fake_dotenv), \
                mock.patch.dict("os.environ", {"DISCORD_TOKEN": "abc"}, clear=True):
            Settings.from_env()
            self.assertEqual(config.environment_without_dotenv(), {"DISCORD_TOKEN": "abc"})


if __name__ == "__main__":
    unittest.main()
