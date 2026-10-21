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

"""Keeps habits from sticking to the bot.

Language models imitate their own earlier replies and obey instructions that are still in the
remembered conversation, so one "end every message with a paw print" (or a persuasive "reply in
braille from now on") can turn into a habit that lasts for the whole memory window. Two small,
content-agnostic mechanisms stop that, whatever the habit is:

* Cooldown: a decorative emoji (or trailing "~") that was used in one of the last few replies is
  removed from the next reply. A flourish can show up now and then, but never in every message.
* Expiry: a message that sets a standing rule ("from now on...", "every message...", "when you
  reply to...") is only kept in the prompt for the one exchange after it. The bot's own "I'll do
  that from now on" reply is treated the same way. Older ones are left out of what the model sees.
"""
from __future__ import annotations

import re
from typing import Sequence

from memory import StoredMessage

# --- Emoji cooldown -----------------------------------------------------------------
# Emoji and pictographic symbols only. Deliberately not "all symbols": the degree sign in "75°F",
# the copyright sign, arrows and maths symbols are ordinary text.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, flags, supplemental symbols
    (0x2600, 0x27BF),  # miscellaneous symbols and dingbats (sun, hearts, sparkles, check marks...)
    (0x231A, 0x231B),
    (0x23E9, 0x23F3),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
)
_ZWJ = 0x200D
_JOINERS = {0xFE0F, 0x20E3, *range(0x1F3FB, 0x1F400)}  # variation selector, keycap, skin tones
_REGIONAL = range(0x1F1E6, 0x1F200)

# A lone "~" glued to the end of a word ("I'll try~"). Not "~~strike~~", not "/~user" in a URL or path.
_TILDE = re.compile(r"(?<=[^\s/~])~(?=\s|$)")


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _EMOJI_RANGES)


def _clusters(text: str) -> list[tuple[int, int]]:
    """Start and end of every emoji in `text`, keeping multi-part ones (flags, families, skin tones) whole."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        if not _is_emoji(text[i]):
            i += 1
            continue
        start = i
        if ord(text[i]) in _REGIONAL and i + 1 < n and ord(text[i + 1]) in _REGIONAL:
            i += 2  # a flag is two regional indicators
        else:
            i += 1
        while i < n:
            code = ord(text[i])
            if code in _JOINERS:
                i += 1
            elif code == _ZWJ and i + 1 < n and _is_emoji(text[i + 1]):
                i += 2
            else:
                break
        spans.append((start, i))
    return spans


def _key(cluster: str) -> str:
    return "".join(char for char in cluster if ord(char) != 0xFE0F)  # "❤" and "❤️" are the same emoji


def ornaments(text: str) -> set[str]:
    """The decorative bits in a reply: each distinct emoji, plus "~" if it is used as a flourish."""
    found = {_key(text[start:end]) for start, end in _clusters(text)}
    if _TILDE.search(text):
        found.add("~")
    return found


def _tidy(text: str) -> str:
    """Close the gaps left by a removed emoji without disturbing indentation or line breaks."""
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+(?=\n|$)", "", text)
    text = re.sub(r"(?<=\S) +([.,!?;:])", r"\1", text)
    return text.strip()


def cool_down(reply: str, recent_replies: Sequence[str]) -> str:
    """Remove every emoji (and flourish "~") from `reply` that appeared in any of `recent_replies`."""
    banned: set[str] = set()
    for earlier in recent_replies:
        banned |= ornaments(earlier)
    if not banned:
        return reply

    pieces, last = [], 0
    for start, end in _clusters(reply):
        if _key(reply[start:end]) in banned:
            pieces.append(reply[last:start])
            last = end
    pieces.append(reply[last:])
    text = "".join(pieces)
    if "~" in banned:
        text = _TILDE.sub("", text)
    text = _tidy(text)
    return text or reply  # never turn a reply that is only an emoji into an empty one


# --- Standing instructions ------------------------------------------------------------
# A standing instruction needs both a "keep doing this" word and a word about how the bot writes.
# Either alone is ordinary conversation ("I always eat breakfast", "reply to my email").
_PERSIST = re.compile(
    r"\b(?:from now on|from here on|going forward|for the rest of|until i say|never again|next time"
    r"|every (?:single )?(?:message|reply|response|time|sentence)|each (?:message|reply|response|time)"
    r"|always|whenever|when you|keep (?:on )?\w+ing)\b",
    re.IGNORECASE,
)
_FORM = re.compile(
    r"\b(?:reply|replies|respond|response|answer|speak|talk|write|type|say|end|start|begin|finish|add"
    r"|include|use|sign|prefix|append|call me|pretend|act (?:like|as)|roleplay|simulat\w*|sound like"
    r"|style|tone|voice|language|format|emoji)\b",
    re.IGNORECASE,
)


def is_standing_instruction(text: str) -> bool:
    return bool(_PERSIST.search(text) and _FORM.search(text))


def _exchanges(history: Sequence[StoredMessage]) -> list[list[StoredMessage]]:
    """Group remembered messages into exchanges: a user message and the reply that followed it."""
    groups: list[list[StoredMessage]] = []
    for item in history:
        if item.role == "user" or not groups:
            groups.append([])
        groups[-1].append(item)
    return groups


def drop_stale_directives(history: Sequence[StoredMessage], keep_recent: int = 1) -> list[StoredMessage]:
    """The history to show the model: exchanges that set a standing rule (in the user's message or in
    the bot's promise to follow it) are left out once they are older than the `keep_recent` newest
    exchanges. With the default, a rule is visible for exactly one more message and then forgotten."""
    groups = _exchanges(history)
    newest_start = max(len(groups) - keep_recent, 0)
    shown: list[StoredMessage] = []
    for index, group in enumerate(groups):
        if index < newest_start and any(is_standing_instruction(item.content) for item in group):
            continue
        shown.extend(group)
    return shown
