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

"""The settings for reading web pages (FETCH_PAGES, PAGE_MAX_CHARS)."""
import unittest
from unittest import mock

from config import ConfigError, Settings


def load(**env):
    # Don't let a real .env file in the project folder leak into the tests.
    with mock.patch("config.load_dotenv"), mock.patch.dict("os.environ", env, clear=True):
        return Settings.from_env()


class PageSettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = load(DISCORD_TOKEN="a")
        self.assertTrue(s.fetch_pages)
        self.assertEqual(s.page_max_chars, 6000)

    def test_values_are_read(self):
        s = load(DISCORD_TOKEN="a", FETCH_PAGES="off", PAGE_MAX_CHARS="12000")
        self.assertFalse(s.fetch_pages)
        self.assertEqual(s.page_max_chars, 12000)

    def test_bad_values_give_clear_errors(self):
        for bad in ({"PAGE_MAX_CHARS": "100"}, {"PAGE_MAX_CHARS": "lots"}, {"FETCH_PAGES": "maybe"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(DISCORD_TOKEN="a", **bad)


if __name__ == "__main__":
    unittest.main()
