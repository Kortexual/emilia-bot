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
from datetime import time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from presence import parse_clock

BASE_DIR = Path(__file__).resolve().parent

# The environment as it was before .env was loaded. A restarted bot starts from this one, so that
# it reads .env afresh: load_dotenv never overrides variables that are already set.
_ENV_BEFORE_DOTENV: dict[str, str] | None = None


def environment_without_dotenv() -> dict[str, str]:
    return dict(os.environ if _ENV_BEFORE_DOTENV is None else _ENV_BEFORE_DOTENV)

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


def _clock(name: str, default: clock_time) -> clock_time:
    raw = _text(name)
    if not raw:
        return default
    try:
        return parse_clock(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a time of day such as 08:00 or 8:00AM, got {raw!r}") from None


def _timezone(name: str) -> str:
    raw = _text(name)
    if raw:
        try:
            ZoneInfo(raw)
        except (KeyError, ValueError, OSError):  # unknown name, malformed name, or a folder instead of a zone
            raise ConfigError(
                f"{name} must be a time zone name such as America/New_York, got {raw!r} "
                "(on Windows, run: pip install tzdata)"
            ) from None
    return raw


def _channel_ids(name: str) -> tuple[int, ...]:
    ids = []
    for part in _text(name).replace(",", " ").split():
        if not part.isdigit() or int(part) == 0:
            raise ConfigError(f"{name} must be a list of Discord channel IDs separated by commas, got {part!r}")
        if int(part) not in ids:
            ids.append(int(part))
    return tuple(ids)


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
    respond_to_bots: bool = False  # opt-in: other bots' @mentions are answered too (never her own messages)
    bot_reply_limit: int = 3  # replies in a row to bots in one channel before a human must speak; 0 = no limit
    web_search: bool = True
    search_results: int = 4
    fetch_pages: bool = True  # let the model read whole pages (links from search results or the sender's message)
    page_max_chars: int = 6000  # characters of page text handed to the model per fetch
    show_sources: bool = True
    computer_tools: bool = False  # opt-in: model can operate the configured workspace
    computer_workspace: str = str(BASE_DIR)
    command_timeout: float = 30.0
    computer_allow_host_access: bool = False  # opt-in: paths are not limited to computer_workspace
    computer_require_confirmation: bool = True  # each mutating call must contain confirm=true
    browser_tools: bool = False  # opt-in: a headless browser the model can open, read and click around in
    browser_require_confirmation: bool = True  # each click, typing and dropdown choice must contain confirm=true
    discord_tools: bool = False  # opt-in: the model can read Discord (server info, channels, messages, members)
    discord_actions: bool = False  # opt-in, needs discord_tools: react, edit/delete her messages, threads, posts, pins, polls
    discord_max_actions: int = 5  # action tool calls per reply
    discord_cross_server: bool = False  # let the Discord tools reach channels in other servers the sender is in
    multi_turn_tools: bool = False  # after a tool call the model keeps taking turns until it calls stop_talking
    max_tool_turns: int = 8  # cap on generations for one message when multi_turn_tools is on
    members_intent: bool = False  # see the whole member list for pinging by name; needs the Server Members Intent
    yandere_mode: bool = False  # an obsessive, possessive (but harmless) version of Emilia; see prompts.YANDERE_PERSONA

    # Her day (see presence.py)
    time_awareness: bool = False  # tell her the time of day: wake-up on the first message, sleepy at night, random events
    timezone: str = ""  # IANA name such as "America/New_York"; empty = this computer's local time
    random_event_cooldown_minutes: int = 120  # least time between random events; 0 = no random events
    random_messages: bool = False  # opt-in: she may speak up unprompted in random_message_channels
    random_message_channels: tuple[int, ...] = ()
    active_start: clock_time = clock_time(8, 0)  # she only speaks up unprompted between these two times
    active_end: clock_time = clock_time(22, 0)
    random_message_min_minutes: int = 60  # never speaks up sooner than this after her last message
    random_message_full_hours: float = 8.0  # the wait after which speaking up is as likely as it gets

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
        global _ENV_BEFORE_DOTENV
        if _ENV_BEFORE_DOTENV is None:
            _ENV_BEFORE_DOTENV = dict(os.environ)
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

        active_start = _clock("ACTIVE_START", cls.active_start)
        active_end = _clock("ACTIVE_END", cls.active_end)
        if active_start == active_end:
            raise ConfigError("ACTIVE_START and ACTIVE_END must be different times")
        min_minutes = _number("RANDOM_MESSAGE_MIN_MINUTES", cls.random_message_min_minutes, int, minimum=0)
        full_hours = _number("RANDOM_MESSAGE_FULL_HOURS", cls.random_message_full_hours, float, minimum=0.1)
        if full_hours * 60 <= min_minutes:
            raise ConfigError(
                f"RANDOM_MESSAGE_FULL_HOURS ({full_hours:g} hours) must be longer than RANDOM_MESSAGE_MIN_MINUTES ({min_minutes} minutes)"
            )
        random_messages = _flag("RANDOM_MESSAGES", cls.random_messages)
        channels = _channel_ids("RANDOM_MESSAGE_CHANNELS")
        if random_messages and not channels:
            raise ConfigError("RANDOM_MESSAGES is on, but RANDOM_MESSAGE_CHANNELS lists no channel IDs")

        discord_tools = _flag("DISCORD_TOOLS", cls.discord_tools)
        discord_actions = _flag("DISCORD_ACTIONS", cls.discord_actions)
        if discord_actions and not discord_tools:
            raise ConfigError("DISCORD_ACTIONS needs DISCORD_TOOLS=true as well")

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
            respond_to_bots=_flag("RESPOND_TO_BOTS", cls.respond_to_bots),
            bot_reply_limit=_number("BOT_REPLY_LIMIT", cls.bot_reply_limit, int, minimum=0),
            web_search=_flag("WEB_SEARCH", cls.web_search),
            search_results=_number("SEARCH_RESULTS", cls.search_results, int, minimum=1),
            fetch_pages=_flag("FETCH_PAGES", cls.fetch_pages),
            page_max_chars=_number("PAGE_MAX_CHARS", cls.page_max_chars, int, minimum=500),
            show_sources=_flag("SHOW_SOURCES", cls.show_sources),
            computer_tools=_flag("COMPUTER_TOOLS", cls.computer_tools),
            computer_workspace=_text("COMPUTER_WORKSPACE") or cls.computer_workspace,
            command_timeout=_number("COMMAND_TIMEOUT", cls.command_timeout, float, minimum=1),
            computer_allow_host_access=_flag("COMPUTER_ALLOW_HOST_ACCESS", cls.computer_allow_host_access),
            computer_require_confirmation=_flag("COMPUTER_REQUIRE_CONFIRMATION", cls.computer_require_confirmation),
            browser_tools=_flag("BROWSER_TOOLS", cls.browser_tools),
            browser_require_confirmation=_flag("BROWSER_REQUIRE_CONFIRMATION", cls.browser_require_confirmation),
            discord_tools=discord_tools,
            discord_actions=discord_actions,
            discord_max_actions=_number("DISCORD_MAX_ACTIONS", cls.discord_max_actions, int, minimum=1),
            discord_cross_server=_flag("DISCORD_CROSS_SERVER", cls.discord_cross_server),
            multi_turn_tools=_flag("MULTI_TURN_TOOLS", cls.multi_turn_tools),
            max_tool_turns=_number("MAX_TOOL_TURNS", cls.max_tool_turns, int, minimum=2),
            members_intent=_flag("MEMBERS_INTENT", cls.members_intent),
            yandere_mode=_flag("YANDERE_MODE", cls.yandere_mode),
            time_awareness=_flag("TIME_AWARENESS", cls.time_awareness),
            timezone=_timezone("TIMEZONE"),
            random_event_cooldown_minutes=_number("RANDOM_EVENT_COOLDOWN_MINUTES", cls.random_event_cooldown_minutes, int, minimum=0),
            random_messages=random_messages,
            random_message_channels=channels,
            active_start=active_start,
            active_end=active_end,
            random_message_min_minutes=min_minutes,
            random_message_full_hours=full_hours,
            memory_db=_text("MEMORY_DB") or cls.memory_db,
            memory_turns=_number("MEMORY_TURNS", cls.memory_turns, int, minimum=1),
            memory_max_chars=_number("MEMORY_MAX_CHARS", cls.memory_max_chars, int, minimum=200),
            style_cooldown=_number("STYLE_COOLDOWN", cls.style_cooldown, int, minimum=0),
            forget_standing_instructions=_flag("FORGET_STANDING_INSTRUCTIONS", cls.forget_standing_instructions),
            max_input_chars=_number("MAX_INPUT_CHARS", cls.max_input_chars, int, minimum=50),
            log_level=level,
        )
