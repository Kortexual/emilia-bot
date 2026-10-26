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

"""Emilia's day: what time it is for her, what she is up to, and when she speaks up on her own.

Everything here is plain logic with the clock and the random numbers passed in, so it can be tested
without Discord or a model. The wording the model sees lives in prompts.py; this module only decides
*what happens*:

* phase_of: which part of the day it is (early morning, midday, late at night...).
* scene_for: whether this message is her first of the day in the morning (she has just woken up),
  whether the sender has stumbled on her in the middle of something (a random event, with a
  cooldown), or neither. The model invents the actual event; we only pick the occasion and a few
  ideas to get it started.
* is_active / send_chance / check_seconds: the hours she is "up", how likely she is to say something
  unprompted right now (which grows the longer it has been since she last spoke), and how often to ask.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time, tzinfo
from typing import Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

CHECK_SECONDS = 300  # how often the bot considers saying something unprompted, at most...
MIN_CHECK_SECONDS = 10  # ...and at least, for very short waits
CHECKS_PER_WAIT = 96  # checks over the full wait: every 5 minutes with the default 8 hours
# How eagerly she speaks up once the wait is as long as it gets, in messages per length of the ramp (the
# time from the minimum wait to the full one). The rate grows with the square of the way along the ramp,
# so half of all gaps end within about 44% of it: with the defaults (1 to 8 hours), about 4 hours.
FULL_RATE = 24.0
EVENT_CHANCE = 0.3  # chance that an eligible reply (cooldown over) comes with a random event
IDEAS_PER_EVENT = 3  # how many activities are suggested to the model as a starting point

# The first hour of each phase, in order. 22:00 to 23:59 is "night", midnight to 04:59 "late at night".
PHASES = (
    (0, "late_night"),
    (5, "early_morning"),
    (10, "late_morning"),
    (12, "midday"),
    (14, "afternoon"),
    (18, "evening"),
    (22, "night"),
)


@dataclass(frozen=True)
class Scene:
    """What is going on around Emilia for one message."""

    clock: str  # "7:42 AM on Friday"
    phase: str  # a key of PHASES
    kind: str = ""  # "" (nothing special), "wake_up" (first message of the morning) or "event"
    ideas: tuple[str, ...] = ()  # a few activities to start from when kind == "event"


# --- Clock and time zone -------------------------------------------------------------------
def parse_clock(text: str) -> clock_time:
    """"08:00", "8:00AM", "10 pm", "22:30" -> a time of day. Raises ValueError for anything else
    (a bare "8" is refused: it could be 8 in the morning or in the evening)."""
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?m\.?)?\s*", text, re.IGNORECASE)
    if not match or (match.group(2) is None and match.group(3) is None):
        raise ValueError(f"not a time of day: {text!r}")
    hour, minute, meridiem = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            raise ValueError(f"not a time of day: {text!r}")
        hour = hour % 12 + (12 if meridiem.lower() == "p" else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"not a time of day: {text!r}")
    return clock_time(hour, minute)


def load_timezone(name: str) -> tzinfo | None:
    """The named IANA time zone, or None for the computer's own local time."""
    return ZoneInfo(name) if name else None


def local_now(zone: tzinfo | None = None, clock: Callable[[], float] = time.time) -> datetime:
    """The current time as an aware datetime, in `zone` or in the computer's local time."""
    stamp = clock()
    return datetime.fromtimestamp(stamp, zone) if zone is not None else datetime.fromtimestamp(stamp).astimezone()


def clock_text(moment: datetime) -> str:
    """"7:42 AM on Friday". (Built by hand: strftime's "%-I" does not exist on Windows.)"""
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'} on {moment:%A}"


def phase_of(hour: int) -> str:
    phase = PHASES[0][1]
    for first_hour, name in PHASES:
        if hour >= first_hour:
            phase = name
    return phase


# --- Active hours and speaking up ------------------------------------------------------------
def is_active(now: clock_time, start: clock_time, end: clock_time) -> bool:
    """Whether `now` is inside the hours from `start` up to (not including) `end`. The window may run
    past midnight: 20:00 to 02:00 covers the evening and the small hours."""
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def check_seconds(full_seconds: float) -> float:
    """How often to consider speaking up: CHECKS_PER_WAIT times over the full wait, so a wait of minutes is
    looked at as finely as one of hours, but no more often than MIN_CHECK_SECONDS and no less than CHECK_SECONDS."""
    return min(CHECK_SECONDS, max(MIN_CHECK_SECONDS, full_seconds / CHECKS_PER_WAIT))


def send_chance(elapsed: float, min_seconds: float, full_seconds: float, interval: float = CHECK_SECONDS) -> float:
    """The chance that she says something unprompted at one check, given the seconds since she last spoke
    and the seconds between checks (`interval`). Zero before `min_seconds`; after that she speaks up at a
    rate that grows with the square of how far along she is towards `full_seconds` (low at first, then
    climbing), up to FULL_RATE messages per length of that ramp. Because the rate is measured against the
    ramp, the gaps have the same shape at any scale and however often she checks: 1 to 8 hours checked
    every 5 minutes and 5 to 15 minutes checked every 10 seconds are the same curve."""
    if elapsed < min_seconds:
        return 0.0
    span = full_seconds - min_seconds
    if span <= 0:  # a degenerate range: as likely as it gets as soon as the minimum is over
        span, ramp = max(interval, 1.0), 1.0
    else:
        ramp = min(1.0, (elapsed - min_seconds) / span)
    return 1.0 - math.exp(-FULL_RATE * ramp * ramp * interval / span)


# --- Scenes --------------------------------------------------------------------------------------
def scene_for(
    now: datetime,
    last_sent: float | None,
    last_event: float | None,
    *,
    ideas: Mapping[str, Sequence[str]],
    event_cooldown_minutes: int,
    rng: random.Random,
    spontaneous: bool = False,
    event_chance: float = EVENT_CHANCE,
) -> Scene:
    """Decide what is happening around her for this message.

    `last_sent` and `last_event` are Unix times of her last message anywhere and of the last scene
    (None if never). Her first message of a local day, when it falls in the early morning, is a
    wake-up. Otherwise a random event happens with probability `event_chance`, but only once
    `event_cooldown_minutes` have passed since the last scene (0 turns events off). A message she
    starts herself (`spontaneous`) always comes with something she is doing, since she has nothing
    to answer, and the cooldown does not apply.
    """
    phase = phase_of(now.hour)
    clock = clock_text(now)
    first_today = last_sent is None or datetime.fromtimestamp(last_sent, now.tzinfo).date() != now.date()
    if first_today and phase == "early_morning":
        return Scene(clock, phase, "wake_up")

    cooled = event_cooldown_minutes > 0 and (last_event is None or now.timestamp() - last_event >= event_cooldown_minutes * 60)
    pool = list(ideas.get(phase, ()))
    if pool and (spontaneous or (cooled and rng.random() < event_chance)):
        return Scene(clock, phase, "event", tuple(rng.sample(pool, min(IDEAS_PER_EVENT, len(pool)))))
    return Scene(clock, phase)
