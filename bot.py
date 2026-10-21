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

"""Discord bot that chats through a local LM Studio model.

It answers only when @mentioned (and, if you opt in with RESPOND_TO_DMS=true, in DMs), remembers
recent conversation per channel, and can search the web when the model is unsure of something.

Run with:  python bot.py     (see README.md for setup)
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import sys
from datetime import date
from typing import Any, Protocol

import discord

import prompts
from config import ConfigError, Settings
from llm import ChatResult, LLMError, LMStudioClient
from memory import MemoryStore, StoredMessage
from textutil import clean_reply, contains_cipher_script, sanitize_user_text, split_message
from tooling import Tool
from websearch import WebSearchTool, ddgs_installed

log = logging.getLogger("emilia.bot")

CHUNK_SIZE = 1900  # Discord rejects messages over 2000 characters; leave some headroom
RESET_COMMANDS = {"!forget", "!reset"}  # only recognised inside an @mention, e.g. "@Emilia !forget"


class ChatBackend(Protocol):
    """What the bot needs from the LLM side (LMStudioClient, or a fake in tests)."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def chat(self, messages: list[dict[str, Any]]) -> ChatResult: ...


def sources_footer(urls: list[str], max_links: int = 3) -> str:
    """A small grey line under the reply naming where web information came from.
    Links are wrapped in <> so Discord doesn't unfurl a big preview for each one."""
    picked: list[str] = []
    for url in urls:
        if len(url) > 200 or any(ch in url for ch in "<> \n\t") or f"<{url}>" in picked:
            continue
        picked.append(f"<{url}>")
        if len(picked) == max_links:
            break
    return "\n-# 🔎 Searched the web · " + " · ".join(picked) if picked else ""


def link_mentions(text: str, people: dict[str, int]) -> tuple[str, list[int]]:
    """Turn "@Name" in a reply into a real Discord ping, for the people in `people` (name -> user ID).

    Returns the new text and the IDs it pings. Anyone not in `people` stays plain text, so the model
    can only ping people who are part of the conversation.
    """
    if not people:
        return text, []
    by_name = {name.lower(): user_id for name, user_id in people.items()}
    names = sorted(people, key=len, reverse=True)  # "@Ann Marie" before "@Ann"
    pattern = re.compile(r"(?<![\w<])@(" + "|".join(map(re.escape, names)) + r")(?!\w)", re.IGNORECASE)
    pinged: list[int] = []

    def to_ping(match: re.Match[str]) -> str:
        user_id = by_name[match.group(1).lower()]
        if user_id not in pinged:
            pinged.append(user_id)
        return f"<@{user_id}>"

    return pattern.sub(to_ping, text), pinged


class EmiliaBot(discord.Client):
    def __init__(
        self,
        settings: Settings,
        *,
        llm: ChatBackend | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: also switch it on in the Developer Portal
        super().__init__(
            intents=intents,
            # Whatever the model writes can never ping @everyone, roles or arbitrary users. Replies
            # that ping people in the conversation allow exactly those users (see link_mentions).
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False, replied_user=True),
        )
        self.settings = settings
        self.memory = memory or MemoryStore(settings.memory_db, settings.memory_turns)
        self.llm: ChatBackend = llm or LMStudioClient(
            base_url=settings.lmstudio_url,
            model=settings.lmstudio_model,
            api_key=settings.lmstudio_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.request_timeout,
            max_concurrent=settings.max_concurrent_requests,
            tools=self._build_tools(),
        )
        self._locks: dict[str, asyncio.Lock] = {}  # one conversation at a time per channel
        self._busy: set[int] = set()  # users with a request in flight

    def _build_tools(self) -> list[Tool]:
        if not self.settings.web_search:
            return []
        if not ddgs_installed():
            log.warning("WEB_SEARCH is on but the 'ddgs' package is missing (pip install ddgs): web search disabled.")
            return []
        return [WebSearchTool(self.settings.search_results)]

    # --- lifecycle -------------------------------------------------------------
    async def setup_hook(self) -> None:
        await self.llm.start()

    async def on_ready(self) -> None:
        scope = "@mentions and DMs" if self.settings.respond_to_dms else "@mentions only"
        log.info("Logged in as %s (id %s); answering %s", self.user, self.user.id, scope)

    async def close(self) -> None:
        await self.llm.close()
        self.memory.close()
        await super().close()

    # --- events ------------------------------------------------------------------
    async def on_message(self, message: discord.Message) -> None:
        if not self._is_for_me(message):
            return
        try:
            await self._handle(message)
        except discord.Forbidden:
            log.warning("Missing permission to reply in channel %s", message.channel.id)
        except Exception:
            log.exception("Unhandled error while answering message %s", message.id)

    def _is_for_me(self, message: discord.Message) -> bool:
        """The whole "only respond to @mentions" rule lives here."""
        if self.user is None or message.author.bot:  # includes ourselves and other bots
            return False
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return False  # joins, pins, etc.
        # An explicit mention of this bot. (@everyone/@here and role pings don't count.)
        if any(user.id == self.user.id for user in message.mentions):
            return True
        return message.guild is None and self.settings.respond_to_dms

    # --- handling ----------------------------------------------------------------
    async def _handle(self, message: discord.Message) -> None:
        text = sanitize_user_text(self._render(message), self.settings.max_input_chars)
        key = str(message.channel.id)

        if text.lower() in RESET_COMMANDS:
            await self._reset(message, key)
            return
        if not text:  # a bare @mention
            await self._send(message, prompts.GREETING)
            return

        user_id = message.author.id
        if user_id in self._busy:
            await self._send(message, prompts.BUSY_REPLY)
            return
        self._busy.add(user_id)
        try:
            quote, quoted_author = await self._quoted_context(message)
            reply, pinged = await self._respond(message, key, quote + text, quoted_author)
        finally:
            self._busy.discard(user_id)
        await self._send(message, reply, pinged)

    async def _respond(
        self, message: discord.Message, key: str, user_text: str, quoted_author: discord.abc.User | None
    ) -> tuple[str, list[int]]:
        """The reply to post, and the IDs of the users it pings."""
        author = prompts.safe_name(message.author.display_name)
        async with message.channel.typing():  # keeps "typing..." showing, even while queued
            async with self._channel_lock(key):
                history = await asyncio.to_thread(self.memory.recent, key, self.settings.memory_max_chars)
                system_prompt = prompts.build_system_prompt(
                    date.today(),
                    web_search=self.settings.web_search,
                    show_sources=self.settings.show_sources,
                )
                messages = prompts.build_messages(
                    system_prompt, history, author, user_text, self._others(message, quoted_author)
                )
                try:
                    result = await self.llm.chat(messages)
                    reply = clean_reply(result.text, self._speaker_names())
                    if contains_cipher_script(reply):
                        # Someone talked the model into writing in braille (or similar). Try once more
                        # with a nudge; if it still won't write plain text, post a canned line instead.
                        log.warning("Reply was written in braille/cipher script; retrying once in plain text")
                        nudged = {**messages[-1], "content": messages[-1]["content"] + prompts.PLAIN_TEXT_HINT}
                        result = await self.llm.chat([*messages[:-1], nudged])
                        reply = clean_reply(result.text, self._speaker_names())
                        if contains_cipher_script(reply):
                            log.warning("Retry was still in braille/cipher script; posting a canned reply")
                            return prompts.FORMAT_REPLY, []  # not stored, so it can't teach the model the pattern
                except LLMError as exc:
                    log.error("LM Studio request failed: %s", exc)
                    return prompts.ERROR_REPLY, []

                if not reply:
                    log.warning("The model sent an empty reply")
                    return prompts.EMPTY_REPLY, []
                if prompts.LEAK_DETECTOR.leaks(reply):
                    log.warning("Blocked a reply that repeated the system prompt")
                    return prompts.BLOCKED_REPLY, []

                try:  # stored with plain "@Name"s, the way the model wrote them
                    await asyncio.to_thread(self.memory.add_turn, key, author, str(message.author.id), user_text, reply)
                except sqlite3.Error:
                    log.exception("Could not save this exchange to memory")  # still deliver the reply

        reply, pinged = link_mentions(reply, self._pingable(message, history, quoted_author))
        if result.sources and self.settings.show_sources:
            reply += sources_footer(result.sources)
        return reply, pinged

    async def _reset(self, message: discord.Message, key: str) -> None:
        if message.guild is not None and not message.channel.permissions_for(message.author).manage_messages:
            await self._send(message, prompts.RESET_DENIED)
            return
        async with self._channel_lock(key):
            removed = await asyncio.to_thread(self.memory.clear, key)
        log.info("Cleared %d remembered messages in channel %s", removed, key)
        await self._send(message, prompts.RESET_DONE)

    # --- helpers -----------------------------------------------------------------
    def _channel_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    def _pingable(
        self, message: discord.Message, history: list[StoredMessage], quoted_author: discord.abc.User | None
    ) -> dict[str, int]:
        """Everyone the reply may ping, by the name the model knows them by: people who spoke in the
        remembered conversation, the sender, and anyone the message mentions or replies to."""
        people = {item.author: int(item.author_id) for item in history if item.role == "user" and item.author_id}
        for user in [*message.mentions, quoted_author, message.author]:
            if user is not None and (self.user is None or user.id != self.user.id):
                people[prompts.safe_name(user.display_name)] = user.id
        return people

    def _others(self, message: discord.Message, quoted_author: discord.abc.User | None) -> list[str]:
        """Names of the other people the message mentions or replies to (not the bot, not the sender),
        so the prompt can say plainly that they are not the one who wrote it."""
        skip = {message.author.id, self.user.id if self.user else None}
        names: list[str] = []
        for user in [*message.mentions, quoted_author]:
            if user is None or user.id in skip:
                continue
            name = prompts.safe_name(user.display_name)
            if name not in names:
                names.append(name)
        return names

    def _speaker_names(self) -> tuple[str, ...]:
        return (prompts.PERSONA_NAME, self.user.display_name if self.user else "")

    def _render(self, message: discord.Message) -> str:
        """Message text with Discord's <@id> markup turned into readable names (our own mention removed)."""
        text = message.content
        bot_id = self.user.id if self.user else None
        for user in message.mentions:
            label = "" if user.id == bot_id else f"@{user.display_name}"
            text = text.replace(f"<@{user.id}>", label).replace(f"<@!{user.id}>", label)
        for role in message.role_mentions:
            text = text.replace(f"<@&{role.id}>", f"@{role.name}")
        for channel in message.channel_mentions:
            text = text.replace(f"<#{channel.id}>", f"#{channel.name}")
        return re.sub(r"<a?:(\w+):\d+>", r":\1:", text)  # custom emoji -> :name:

    async def _quoted_context(self, message: discord.Message) -> tuple[str, discord.abc.User | None]:
        """If this is a reply to someone else's message, include what it said (and who said it), so
        that '@Emilia is this true?' makes sense. Our own earlier messages are already in memory."""
        ref = message.reference
        if ref is None or ref.message_id is None:
            return "", None
        quoted = ref.resolved
        if quoted is None:
            try:
                quoted = await message.channel.fetch_message(ref.message_id)
            except discord.HTTPException:
                return "", None
        if not hasattr(quoted, "author"):  # the original was deleted
            return "", None
        if self.user is not None and quoted.author.id == self.user.id:
            return "", None
        body = sanitize_user_text(self._render(quoted), 300).replace('"', "'")
        if not body:
            return "", None
        return f'(replying to {prompts.safe_name(quoted.author.display_name)}: "{body}") ', quoted.author

    async def _send(self, message: discord.Message, text: str, pinged: list[int] | None = None) -> None:
        """Reply to the message, spilling over into follow-up messages if it is long.
        Only the users in `pinged` are actually notified by mentions in the text."""
        allowed = discord.AllowedMentions(users=[discord.Object(user_id) for user_id in pinged or []])
        for index, chunk in enumerate(split_message(text, CHUNK_SIZE)):
            if index == 0:
                try:
                    await message.reply(chunk, allowed_mentions=allowed)
                    continue
                except discord.Forbidden:
                    raise
                except discord.HTTPException as exc:  # e.g. the original message was deleted meanwhile
                    log.info("Could not reply to message %s (%s); posting normally instead", message.id, exc)
            await message.channel.send(chunk, allowed_mentions=allowed)


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "primp", "urllib3", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    bot = EmiliaBot(settings)
    try:
        bot.run(settings.discord_token, log_handler=None)  # we configured logging ourselves above
    except discord.LoginFailure:
        raise SystemExit("Discord rejected DISCORD_TOKEN. Reset the token in the Developer Portal and update .env.") from None
    except discord.PrivilegedIntentsRequired:
        raise SystemExit(
            "Turn on 'Message Content Intent': Developer Portal > your app > Bot > Privileged Gateway Intents."
        ) from None


if __name__ == "__main__":
    main()
