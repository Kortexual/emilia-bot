import unittest
from unittest import mock

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
        self.assertTrue(s.web_search)
        self.assertEqual(s.memory_turns, 10)

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
            WEB_SEARCH="off",
            LMSTUDIO_TEMPERATURE="0.7",
            LMSTUDIO_MAX_TOKENS="512",
            MEMORY_TURNS="4",
        )
        self.assertTrue(s.respond_to_dms)
        self.assertFalse(s.web_search)
        self.assertEqual(s.temperature, 0.7)
        self.assertEqual(s.max_tokens, 512)
        self.assertEqual(s.memory_turns, 4)

    def test_bad_values_give_clear_errors(self):
        for bad in (
            {"WEB_SEARCH": "maybe"},
            {"MEMORY_TURNS": "lots"},
            {"MEMORY_TURNS": "0"},
            {"LOG_LEVEL": "LOUD"},
            {"LMSTUDIO_TEMPERATURE": "-1"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(DISCORD_TOKEN="a", **bad)


if __name__ == "__main__":
    unittest.main()
