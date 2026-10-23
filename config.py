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

"""Settings, read once at startup from environment variables (and a .env file if there is one)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# The original template used this placeholder; treat it the same as "not set".
_PLACEHOLDER_MODEL = "your-model-name"
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """A setting is missing or has an unusable value."""


def _text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    raw = _text(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _number(name: str, default, cast, minimum=None):
    raw = _text(name)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    discord_token: str

    # LM Studio
    lmstudio_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_model: str = ""  # empty = use the first chat model LM Studio lists
    lmstudio_api_key: str = ""
    temperature: float | None = None  # None = leave it to LM Studio's own settings
    max_tokens: int | None = None  # same
    request_timeout: float = 300.0  # seconds; big or reasoning models can be slow
    max_concurrent_requests: int = 1  # simultaneous generations sent to LM Studio

    # Behaviour
    respond_to_dms: bool = False  # False = strictly @mentions only
    web_search: bool = True
    search_results: int = 4
    show_sources: bool = True
    members_intent: bool = False  # see the whole member list for pinging by name; needs the Server Members Intent
    yandere_mode: bool = False  # an obsessive, possessive (but harmless) version of Emilia; see prompts.YANDERE_PERSONA

    # Memory
    memory_db: str = str(BASE_DIR / "memory.db")
    memory_turns: int = 10  # exchanges (user + bot) remembered per channel
    memory_max_chars: int = 6000  # cap on how much memory is sent to the model per request

    # Keeping habits from sticking (see styleguard.py)
    style_cooldown: int = 4  # an emoji (or "~") used in one of the last N replies is removed from the next; 0 = off
    forget_standing_instructions: bool = True  # leave "from now on..." rules out of the prompt after one exchange

    max_input_chars: int = 2048
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        token = _text("DISCORD_TOKEN")
        if not token:
            raise ConfigError("DISCORD_TOKEN is not set (put it in .env or the environment)")

        url = _text("LMSTUDIO_URL", cls.lmstudio_url).rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"  # people often paste just http://localhost:1234

        model = _text("LMSTUDIO_MODEL")
        if model == _PLACEHOLDER_MODEL:
            model = ""

        level = _text("LOG_LEVEL", "INFO").upper()
        if level not in _LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {', '.join(sorted(_LOG_LEVELS))}, got {level!r}")

        return cls(
            discord_token=token,
            lmstudio_url=url,
            lmstudio_model=model,
            lmstudio_api_key=_text("LMSTUDIO_API_KEY"),
            temperature=_number("LMSTUDIO_TEMPERATURE", None, float, minimum=0),
            max_tokens=_number("LMSTUDIO_MAX_TOKENS", None, int, minimum=1),
            request_timeout=_number("LMSTUDIO_TIMEOUT", cls.request_timeout, float, minimum=1),
            max_concurrent_requests=_number("MAX_CONCURRENT_REQUESTS", 1, int, minimum=1),
            respond_to_dms=_flag("RESPOND_TO_DMS", cls.respond_to_dms),
            web_search=_flag("WEB_SEARCH", cls.web_search),
            search_results=_number("SEARCH_RESULTS", cls.search_results, int, minimum=1),
            show_sources=_flag("SHOW_SOURCES", cls.show_sources),
            members_intent=_flag("MEMBERS_INTENT", cls.members_intent),
            yandere_mode=_flag("YANDERE_MODE", cls.yandere_mode),
            memory_db=_text("MEMORY_DB") or cls.memory_db,
            memory_turns=_number("MEMORY_TURNS", cls.memory_turns, int, minimum=1),
            memory_max_chars=_number("MEMORY_MAX_CHARS", cls.memory_max_chars, int, minimum=200),
            style_cooldown=_number("STYLE_COOLDOWN", cls.style_cooldown, int, minimum=0),
            forget_standing_instructions=_flag("FORGET_STANDING_INSTRUCTIONS", cls.forget_standing_instructions),
            max_input_chars=_number("MAX_INPUT_CHARS", cls.max_input_chars, int, minimum=50),
            log_level=level,
        )
