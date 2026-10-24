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
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Protocol

import discord

import browser
import prompts
import styleguard
from browser import BrowserSessions, browser_tools, playwright_installed
from config import ConfigError, Settings, environment_without_dotenv
from computer import computer_tools
from restart import RestartTool
from llm import ChatResult, LLMError, LMStudioClient, TextHandler, ToolHandler, ToolUse
from memory import MemoryStore, StoredMessage
from textutil import clean_reply, contains_cipher_script, sanitize_user_text, split_message
from tooling import Tool
from webpage import FetchPageTool, UrlRegistry, extract_urls
from websearch import WebSearchTool, ddgs_installed

log = logging.getLogger("emilia.bot")

RETIRE_TIMEOUT = 600  # seconds a restarted bot waits for its replies in progress before it shuts down anyway
CHUNK_SIZE = 1900  # Discord rejects messages over 2000 characters; leave some headroom
RESET_COMMANDS = {"!forget", "!reset"}  # only recognised inside an @mention, e.g. "@Emilia !forget"
FORGET_ME_COMMAND = "!forgetme"  # deletes the notes and names saved about the person who sends it
ABOUT_COMMAND = "!about"  # "@Emilia !about I garden and like tea" saves a note about yourself


class ChatBackend(Protocol):
    """What the bot needs from the LLM side (LMStudioClient, or a fake in tests)."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def chat(
        self,
        messages: list[dict[str, Any]],
        on_text: TextHandler | None = None,
        on_tool: ToolHandler | None = None,
        on_tool_done: ToolHandler | None = None,
    ) -> ChatResult: ...


def source_links(urls: list[str], max_links: int = 3) -> str:
    """A few of the links a tool consulted, as " · <url> · <url>". Links are wrapped in <> so Discord
    doesn't unfurl a big preview for each one."""
    picked: list[str] = []
    for url in urls:
        if len(url) > 200 or any(ch in url for ch in "<> \n\t") or f"<{url}>" in picked:
            continue
        picked.append(f"<{url}>")
        if len(picked) == max_links:
            break
    return "".join(f" · {link}" for link in picked)


def tool_line(use: ToolUse, *, show_sources: bool = False) -> str:
    """The small grey message posted when the model uses a tool, with its links once it has run."""
    return "-# " + use.description + (source_links(use.sources) if show_sources else "")


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


_AT_NAME = re.compile(r"(?<![\w<])@(\w[\w'.-]*(?:[ \t]+\w[\w'.-]*){0,2})")
MAX_MEMBER_LOOKUPS = 3  # per reply: each one may cost a request to Discord


def ping_candidates(reply: str) -> list[list[str]]:
    """For each "@..." in a reply, the names it could stand for, longest first ("@Jack Larpel hi" ->
    ["Jack Larpel hi", "Jack Larpel", "Jack"]), since the model doesn't mark where a name ends."""
    groups = []
    for match in _AT_NAME.finditer(reply):
        words = match.group(1).split()
        names = [" ".join(words[:count]).rstrip(".'-") for count in range(len(words), 0, -1)]
        groups.append([name for name in dict.fromkeys(names) if name])
    return groups


def _norm(text: str) -> str:
    """Lower-case, drop punctuation and collapse spaces: "Jack P." -> "jack p"."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _first_word(name: str) -> str:
    """The first word of a name, also splitting usernames at _ . -: "Jack P." and "jack_p" -> "jack"."""
    parts = re.split(r"[\s_.\-]+", name.strip().lower())
    return _norm(parts[0]) if parts and parts[0] else ""


@dataclass
class _Candidate:
    """A server member who might be who the sender meant, with every name they are known by."""

    member: discord.Member
    names: set[str] = field(default_factory=set)  # normalised full names
    firsts: set[str] = field(default_factory=set)  # normalised first words
    words: set[str] = field(default_factory=set)  # every normalised word of every name ("larpel" too)

    def add(self, name: str | None) -> None:
        if name and _norm(name):
            self.names.add(_norm(name))
            if _first_word(name):
                self.firsts.add(_first_word(name))
            self.words.update(part for part in (_norm(w) for w in re.split(r"[\s_.\-]+", name)) if part)


def _typed_words(words: list[str], typed_norm: str) -> list[str]:
    """The longest start of `words` that the sender really typed (as whole words, ignoring case and
    punctuation), or [] if not even the first word was typed."""
    padded = f" {typed_norm} "
    for count in range(len(words), 0, -1):
        anchor = _norm(" ".join(words[:count]))
        if anchor and f" {anchor} " in padded:
            return words[:count]
    return []


def _pick(pool: dict[int, _Candidate], reading: str, anchor_words: list[str]) -> _Candidate | None:
    """The one member a name refers to, trying stricter matches first:
    1. a name of theirs equals what the model wrote ("jack P." -> nickname "Jack P.")
    2. a name of theirs equals what the sender typed
    3. the sender typed one word and it is the first word of one of their names ("Jack" -> "Jack P.")
    4. the sender typed one word and it is any other word of one of their names ("Larpel" -> "Jack Larpel")
    If a tier matches several people the name is ambiguous and nobody is picked."""
    reading_n, anchor_n = _norm(reading), _norm(" ".join(anchor_words))
    tests = (
        lambda c: reading_n in c.names,
        lambda c: anchor_n in c.names,
        lambda c: " " not in anchor_n and anchor_n in c.firsts,
        lambda c: " " not in anchor_n and anchor_n in c.words,
    )
    for test in tests:
        hits = [candidate for candidate in pool.values() if test(candidate)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            return None
    return None


def _name_span(reading_words: list[str], candidate: _Candidate, anchor_n: str) -> str | None:
    """How much of "@jack P. says hi" is the person's name: the longest start that is one of their
    names (or is what the sender typed). Only that part is replaced by the ping."""
    for count in range(len(reading_words), 0, -1):
        text = " ".join(reading_words[:count])
        if _norm(text) in candidate.names or _norm(text) == anchor_n:
            return text.rstrip(".'-") or None
    return None


class EmiliaBot(discord.Client):
    def __init__(
        self,
        settings: Settings,
        *,
        llm: ChatBackend | None = None,
        memory: MemoryStore | None = None,
        restart_ready_fd: int | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: also switch it on in the Developer Portal
        intents.members = settings.members_intent  # privileged: Server Members Intent, only if MEMBERS_INTENT=true
        super().__init__(
            intents=intents,
            # Whatever the model writes can never ping @everyone, roles or arbitrary users. Replies
            # that ping people in the conversation allow exactly those users (see link_mentions).
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False, replied_user=True),
        )
        self.settings = settings
        self.urls = UrlRegistry()  # links the model may open with fetch_page/browser_open: search results and the sender's links
        self.browser = BrowserSessions()  # Chromium starts only when a browser tool is first used
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
            multi_turn=settings.multi_turn_tools,
            max_turns=settings.max_tool_turns,
        )
        self._locks: dict[str, asyncio.Lock] = {}  # one conversation at a time per channel
        self._busy: set[int] = set()  # users with a request in flight
        self._restart_ready_fd = restart_ready_fd
        self._retiring = False  # a replacement has taken over: answer no new messages, then shut down
        self._retire_task: asyncio.Task[None] | None = None

    def _build_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        if self.settings.computer_tools:
            tools.extend(
                computer_tools(
                    self.settings.computer_workspace,
                    command_timeout=self.settings.command_timeout,
                    allow_host_access=self.settings.computer_allow_host_access,
                    require_confirmation=self.settings.computer_require_confirmation,
                )
            )
            if os.name == "posix":  # the readiness handoff passes a pipe to the new process
                tools.append(
                    RestartTool(
                        [sys.executable, *sys.argv],
                        self._retire_after_handoff,
                        environment=environment_without_dotenv(),
                        require_confirmation=self.settings.computer_require_confirmation,
                    )
                )
        if self.settings.web_search:
            if ddgs_installed():
                tools.append(WebSearchTool(self.settings.search_results, registry=self.urls))
            else:
                log.warning("WEB_SEARCH is on but the 'ddgs' package is missing (pip install ddgs): web search disabled.")
        if self.settings.fetch_pages:
            tools.append(FetchPageTool(self.urls, max_chars=self.settings.page_max_chars))
        if self.settings.browser_tools:
            if playwright_installed():
                tools.extend(
                    browser_tools(
                        self.browser,
                        self.urls,
                        max_chars=self.settings.page_max_chars,
                        require_confirmation=self.settings.browser_require_confirmation,
                    )
                )
            else:
                log.warning(
                    "BROWSER_TOOLS is on but Playwright is missing (pip install playwright && playwright install chromium): "
                    "browser tools disabled."
                )
        return tools

    # --- lifecycle -------------------------------------------------------------
    async def setup_hook(self) -> None:
        await self.llm.start()

    async def on_ready(self) -> None:
        scope = "@mentions and DMs" if self.settings.respond_to_dms else "@mentions only"
        log.info("Logged in as %s (id %s); answering %s", self.user, self.user.id, scope)
        if self._restart_ready_fd is not None:
            try:
                os.write(self._restart_ready_fd, b"1")
            except OSError:
                log.warning("Could not notify the previous process that restart succeeded")
            finally:
                os.close(self._restart_ready_fd)
                self._restart_ready_fd = None

    async def _retire_after_handoff(self) -> None:
        """The replacement is connected: it answers new messages from now on. This bot finishes the
        replies it is working on (including the one that asked for the restart), then shuts down."""
        if self._retiring:
            return
        self._retiring = True
        log.info("A replacement bot has taken over; shutting down once the current replies are done")

        async def retire() -> None:
            deadline = asyncio.get_running_loop().time() + RETIRE_TIMEOUT
            await asyncio.sleep(1)  # let the tool result reach the model first
            while self._busy and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.5)
            await self.close()

        self._retire_task = asyncio.create_task(retire())  # keep a reference so it isn't garbage-collected

    async def close(self) -> None:
        await self.browser.close()
        await self.llm.close()
        self.memory.close()
        await super().close()

    # --- events ------------------------------------------------------------------
    async def on_message(self, message: discord.Message) -> None:
        if self._retiring or not self._is_for_me(message):  # once restarted, the replacement answers
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

        command, _, argument = text.partition(" ")
        if text.lower() in RESET_COMMANDS:
            await self._reset(message, key)
            return
        if text.lower() == FORGET_ME_COMMAND:
            await self._forget_me(message)
            return
        if command.lower() == ABOUT_COMMAND:
            await self._about(message, argument)
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
            quoted_body, quoted_author = await self._quoted_context(message)
            await self._respond(message, key, text, quoted_body, quoted_author)
        finally:
            self._busy.discard(user_id)

    async def _respond(
        self,
        message: discord.Message,
        key: str,
        text: str,
        quoted_body: str,
        quoted_author: discord.abc.User | None,
    ) -> None:
        """Generate and post messages; a tool-assisted reply can continue in further turns."""
        people = self._people(message, quoted_author)
        author_name = prompts.safe_name(message.author.display_name)  # what is stored; labels are worked out per request
        async with message.channel.typing():  # keeps "typing..." showing, even while queued
            async with self._channel_lock(key):
                await self._attach_ids(key, people)
                history = await asyncio.to_thread(self.memory.recent, key, self.settings.memory_max_chars)
                labels = self._labels(history, people)
                known = await self._remember_people(people, list(labels))
                author = labels[str(message.author.id)]
                others = [labels[str(user.id)] for user in people[1:]]
                quote = ""
                if quoted_body and quoted_author is not None:
                    quote = f'(replying to {labels[str(quoted_author.id)]}: "{quoted_body}") '
                user_text = quote + text

                shown = history
                if self.settings.forget_standing_instructions:
                    shown = styleguard.drop_stale_directives(shown)
                shown = [replace(m, author=labels.get(m.author_id, m.author)) if m.role == "user" else m for m in shown]
                profiles = await self._profiles(people, labels)

                system_prompt = prompts.build_system_prompt(
                    date.today(),
                    web_search=self.settings.web_search,
                    show_sources=self.settings.show_sources,
                    computer_tools=self.settings.computer_tools,
                    multi_turn=self.settings.multi_turn_tools,
                    yandere=self.settings.yandere_mode,
                    fetch_pages=self.settings.fetch_pages,
                    browser_tools=self.settings.browser_tools,
                )
                if self.settings.fetch_pages or self.settings.browser_tools:  # links the sender wrote (or replied to) may be opened
                    self.urls.allow(extract_urls(text) + extract_urls(quoted_body))
                browser.SESSION_KEY.set(key)  # each channel browses in its own browser context
                messages = prompts.build_messages(
                    system_prompt, shown, author, user_text, others, profiles, yandere=self.settings.yandere_mode
                )

                # A reply can be several Discord messages, posted as they happen: whatever the model writes
                # next to a tool call, a small line for each tool it uses (edited to add its links once it has
                # run), then its answer (and in multi-turn mode, more as it keeps working).
                recent: list[str] = []
                if self.settings.style_cooldown:
                    recent = await asyncio.to_thread(self.memory.last_replies, key, self.settings.style_cooldown)
                kept: list[str] = []  # what was posted, as the model wrote it (plain "@Name"s): this is remembered
                posted = False

                async def post(reply: str, *, from_model: bool = True) -> None:
                    """Post one message. Canned lines (from_model=False) are neither remembered nor allowed to ping anyone."""
                    nonlocal posted
                    pinged: list[int] = []
                    if reply and from_model:
                        if prompts.LEAK_DETECTOR.leaks(reply):
                            log.warning("Blocked a reply that repeated the system prompt")
                            reply, from_model = prompts.BLOCKED_REPLY, False
                        else:
                            if self.settings.style_cooldown:  # no emoji/flourish may stick from message to message
                                reply = styleguard.cool_down(reply, [*recent, *kept])
                            kept.append(reply)
                            pingable = self._pingable(labels, known)
                            pingable.update(await self._lookup_members(message, reply, text, pingable))
                            reply, pinged = link_mentions(reply, pingable)
                    if reply:
                        posted = True
                        await self._send(message, reply, pinged)

                async def on_text(draft: str) -> None:
                    reply = clean_reply(draft, self._speaker_names())
                    if contains_cipher_script(reply):
                        log.warning("Dropped a message written in braille/cipher script")
                    elif reply:
                        await post(reply)

                tool_messages: dict[int, discord.Message] = {}  # by id() of the ToolUse

                async def on_tool(use: ToolUse) -> None:
                    nonlocal posted
                    posted = True
                    sent = await message.channel.send(tool_line(use), allowed_mentions=discord.AllowedMentions.none())
                    tool_messages[id(use)] = sent

                async def on_tool_done(use: ToolUse) -> None:
                    sent = tool_messages.pop(id(use), None)
                    if sent is None or not (self.settings.show_sources and source_links(use.sources)):
                        return
                    try:
                        await sent.edit(content=tool_line(use, show_sources=True), allowed_mentions=discord.AllowedMentions.none())
                    except discord.HTTPException as exc:
                        log.info("Could not add links to a tool message (%s)", exc)

                try:
                    result = await self.llm.chat(messages, on_text, on_tool, on_tool_done)
                    reply = clean_reply(result.text, self._speaker_names())
                    if contains_cipher_script(reply) and not result.used_tools and not posted:
                        # Someone talked the model into writing in braille (or similar). Try once more
                        # with a nudge; nothing has run or been posted yet, so the retry is harmless.
                        log.warning("Reply was written in braille/cipher script; retrying once in plain text")
                        nudged = {**messages[-1], "content": messages[-1]["content"] + prompts.PLAIN_TEXT_HINT}
                        result = await self.llm.chat([*messages[:-1], nudged], on_text, on_tool, on_tool_done)
                        reply = clean_reply(result.text, self._speaker_names())
                except LLMError as exc:
                    log.error("LM Studio request failed: %s", exc)
                    await post(prompts.ERROR_REPLY, from_model=False)
                else:
                    if contains_cipher_script(reply):
                        log.warning("Reply was still in braille/cipher script; posting a canned reply")
                        await post(prompts.FORMAT_REPLY, from_model=False)  # not stored, so it can't teach the model the pattern
                    elif reply:
                        await post(reply)
                    elif not posted:
                        log.warning("The model sent an empty reply")
                        await post(prompts.EMPTY_REPLY, from_model=False)

                if kept:
                    try:  # stored with plain "@Name"s, the way the model wrote them
                        await asyncio.to_thread(
                            self.memory.add_turn, key, author_name, str(message.author.id), user_text, "\n\n".join(kept)
                        )
                    except sqlite3.Error:
                        log.exception("Could not save this exchange to memory")

    async def _reset(self, message: discord.Message, key: str) -> None:
        if message.guild is not None and not message.channel.permissions_for(message.author).manage_messages:
            await self._send(message, prompts.RESET_DENIED)
            return
        async with self._channel_lock(key):
            removed = await asyncio.to_thread(self.memory.clear, key)
        log.info("Cleared %d remembered messages in channel %s", removed, key)
        await self._send(message, prompts.RESET_DONE)

    async def _about(self, message: discord.Message, argument: str) -> None:
        """`!about <text>` saves a short note about yourself, `!about` shows it, `!about clear` removes it."""
        user_id = str(message.author.id)
        argument = argument.strip()
        try:
            if not argument:
                note = (await asyncio.to_thread(self.memory.abouts, [user_id])).get(user_id, "")
                await self._send(message, prompts.ABOUT_SHOW.format(text=note) if note else prompts.ABOUT_NONE)
            elif argument.lower() in {"clear", "reset", "delete", "remove"}:
                await asyncio.to_thread(self.memory.set_about, user_id, "")
                await self._send(message, prompts.ABOUT_CLEARED)
            else:
                note = prompts.clean_about(argument)
                await asyncio.to_thread(self.memory.set_about, user_id, note)
                await self._send(message, prompts.ABOUT_SAVED if note else prompts.ABOUT_NONE)
        except sqlite3.Error:
            log.exception("Could not read or save a note for user %s", user_id)
            await self._send(message, prompts.ERROR_REPLY)

    async def _forget_me(self, message: discord.Message) -> None:
        await asyncio.to_thread(self.memory.forget_user, str(message.author.id))
        log.info("Forgot the saved profile of user %s", message.author.id)
        await self._send(message, prompts.FORGOT_YOU)

    # --- helpers -----------------------------------------------------------------
    def _people(self, message: discord.Message, quoted_author: discord.abc.User | None) -> list[discord.abc.User]:
        """Everyone involved in this message except the bot: the sender first, then anyone it mentions
        or replies to, each once."""
        bot_id = self.user.id if self.user else None
        people = [message.author]
        for user in [*message.mentions, quoted_author]:
            if user is not None and user.id != bot_id and all(user.id != seen.id for seen in people):
                people.append(user)
        return people

    @staticmethod
    def _labels(history: list[StoredMessage], people: list[discord.abc.User]) -> dict[str, str]:
        """One unambiguous name per person (by user ID) for use in the prompt.

        A person keeps a single name even if they changed it (the newest one wins), and two different
        people who happen to share a display name get a short ID suffix ("Alex #4821") so the model
        can never take them for the same person."""
        names: dict[str, str] = {}
        for item in history:
            if item.role == "user" and item.author_id:
                names[item.author_id] = item.author or "someone"  # newer messages come later and win
        for user in people:
            names[str(user.id)] = prompts.safe_name(user.display_name)
        counts = Counter(name.lower() for name in names.values())
        return {
            user_id: name if counts[name.lower()] == 1 else f"{name} #{user_id[-4:]}"
            for user_id, name in names.items()
        }

    @staticmethod
    def _name_entries(users: list[discord.abc.User]) -> list[tuple[str, list[tuple[str, str]]]]:
        """(user ID, [(kind, name), ...]) for each user, with every name they go by, cleaned."""
        entries = []
        for user in users:
            names = [
                ("display", user.display_name),
                ("username", getattr(user, "name", None)),
                ("global", getattr(user, "global_name", None)),
                ("nick", getattr(user, "nick", None)),
            ]
            cleaned = [(kind, prompts.safe_name(name, default="")) for kind, name in names if name]
            entries.append((str(user.id), [(kind, name) for kind, name in cleaned if name]))
        return entries

    async def _remember_people(self, people: list[discord.abc.User], ids: list[str]) -> dict[str, list[str]]:
        """Save the names these people go by, and return every name known for the users in `ids`
        (user ID -> names): the people in this message and those from the remembered conversation."""
        try:
            await asyncio.to_thread(self.memory.observe_users, self._name_entries(people))
            return await asyncio.to_thread(self.memory.aliases, ids)
        except sqlite3.Error:
            log.exception("Could not read or save the people directory")  # the reply is still possible
            return {}

    async def _profiles(self, people: list[discord.abc.User], labels: dict[str, str]) -> dict[str, str]:
        """Notes the sender and the people they mention wrote about themselves, by prompt name."""
        ids = [str(user.id) for user in people]
        try:
            notes = await asyncio.to_thread(self.memory.abouts, ids)
        except sqlite3.Error:
            log.exception("Could not read the people notes")
            return {}
        return {labels[user_id]: note for user_id, note in notes.items()}

    def _channel_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    async def _lookup_members(
        self, message: discord.Message, reply: str, typed: str, already: dict[str, int]
    ) -> dict[str, int]:
        """People the sender asked for by name ("ping Jack") who are in this server but not in the
        conversation yet, as {name as written in the reply: user ID}.

        The sender must have typed the name (at least its first word) in this message, so text from the
        web or from old messages can never make the bot ping someone. The name is then matched against
        the server's members and the names saved in the directory, and only a single clear match counts:
        "Jack" finds the one member called "Jack P.", but not when "Jack S." exists too."""
        guild = getattr(message, "guild", None)
        if guild is None:
            return {}
        known = {name.lower() for name in already}
        typed_norm = _norm(typed)
        found: dict[str, int] = {}
        lookups = 0
        for readings in ping_candidates(reply):
            if any(reading.lower() in known or reading.lower() in {n.lower() for n in found} for reading in readings):
                continue  # handled by the conversation (or by an earlier lookup in this reply)
            first = readings[-1].split()[0]  # a reading is "Jack P" or "Jack P hi"; the shortest is the first word
            if not _typed_words([first], typed_norm):
                continue
            if lookups >= MAX_MEMBER_LOOKUPS:
                break
            lookups += 1
            pool = await self._member_pool(guild, first)
            for reading in readings:
                words = reading.split()
                anchor = _typed_words(words, typed_norm)
                match = _pick(pool, reading, anchor) if anchor else None
                if match is not None:
                    span = _name_span(words, match, _norm(" ".join(anchor)))
                    if span:
                        found[span] = match.member.id
                        await self._record_member(match.member)
                        await self._attach_ids(str(message.channel.id), [match.member])
                        break
        return found

    async def _member_pool(self, guild: discord.Guild, first_word: str) -> dict[int, _Candidate]:
        """Server members whose names could be what `first_word` starts: those Discord has already told
        us about, Discord's member search (which needs no privileged intent), and anyone saved in the
        directory under a matching name who is still in this server (that finds people by an old name)."""
        start = _norm(first_word)
        bot_id = self.user.id if self.user else None
        found = list(getattr(guild, "members", []) or [])
        search = getattr(guild, "search_members", None)
        if search is not None:
            try:
                found += await search(first_word, limit=25)
            except (discord.HTTPException, discord.Forbidden):
                log.info("Member search for %r was refused or failed", first_word)

        pool: dict[int, _Candidate] = {}

        def add(member) -> _Candidate | None:
            if getattr(member, "bot", False) or member.id == bot_id:
                return None
            candidate = pool.setdefault(member.id, _Candidate(member))
            for name in (member.display_name, getattr(member, "name", None), getattr(member, "global_name", None), getattr(member, "nick", None)):
                candidate.add(name)
            return candidate

        for member in found:
            add(member)
        pool = {mid: c for mid, c in pool.items() if start in c.words or any(n.startswith(start) for n in c.names)}

        try:
            saved_ids = await asyncio.to_thread(self.memory.find_by_alias, start)
        except sqlite3.Error:
            log.exception("Could not search the people directory")
            saved_ids = []
        for user_id in saved_ids[:5]:
            member = guild.get_member(int(user_id)) if hasattr(guild, "get_member") else None
            if member is None and hasattr(guild, "fetch_member"):
                try:
                    member = await guild.fetch_member(int(user_id))
                except discord.HTTPException:
                    member = None  # not in this server (or not allowed to ask): then they can't be pinged here
            candidate = add(member) if member is not None else None
            if candidate is not None:
                for alias in (await asyncio.to_thread(self.memory.aliases, [user_id])).get(user_id, []):
                    candidate.add(alias)
        return pool

    async def _attach_ids(self, key: str, users: list[discord.abc.User]) -> None:
        """Messages saved before user IDs were recorded name their author but have no ID, so that person
        could never be pinged from memory. When someone whose name matches takes part (or is found by
        name in the server), their real ID is attached to those messages. A display name shared by two
        people in the same message is skipped, since the old messages could belong to either."""
        counts = Counter(prompts.safe_name(user.display_name).lower() for user in users)
        try:
            for user in users:
                if counts[prompts.safe_name(user.display_name).lower()] > 1:
                    continue
                user_id = str(user.id)
                names = [name for _, name in self._name_entries([user])[0][1]]
                names += (await asyncio.to_thread(self.memory.aliases, [user_id])).get(user_id, [])
                await asyncio.to_thread(self.memory.backfill_ids, key, user_id, names)
        except sqlite3.Error:
            log.exception("Could not attach user IDs to older messages")  # only a convenience

    async def _record_member(self, member: discord.Member) -> None:
        """Save a member who was just pinged in the directory, with their real user ID and names."""
        names = [
            ("display", member.display_name),
            ("username", getattr(member, "name", None)),
            ("global", getattr(member, "global_name", None)),
            ("nick", getattr(member, "nick", None)),
        ]
        cleaned = [(kind, prompts.safe_name(name, default="")) for kind, name in names if name]
        try:
            await asyncio.to_thread(self.memory.observe_users, [(str(member.id), [(k, n) for k, n in cleaned if n])])
        except sqlite3.Error:
            log.exception("Could not save a pinged member to the people directory")

    @staticmethod
    def _pingable(labels: dict[str, str], known: dict[str, list[str]]) -> dict[str, int]:
        """Everyone the reply may ping, by every name the model might use for them: the name shown in
        the prompt, plus their username and earlier names (when those point to one person only). Only
        people in this conversation are included: the sender, anyone the message mentions or replies to,
        and people who spoke in the remembered conversation."""
        owners: dict[str, set[str]] = {}
        for user_id, names in known.items():
            for name in names:
                owners.setdefault(name.lower(), set()).add(user_id)
        people = {
            name: int(user_id)
            for user_id, names in known.items()
            for name in names
            if owners[name.lower()] == {user_id}
        }
        people.update({label: int(user_id) for user_id, label in labels.items()})  # the shown name always wins
        return people

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
        """If this is a reply to someone else's message, what it said and who said it, so that
        '@Emilia is this true?' makes sense. Our own earlier messages are already in memory."""
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
        return body, quoted.author

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

    ready_fd: int | None = None
    raw_ready_fd = os.environ.pop("EMILIA_RESTART_READY_FD", "")
    if raw_ready_fd:
        try:
            ready_fd = int(raw_ready_fd)
        except ValueError:
            log.warning("Ignoring invalid EMILIA_RESTART_READY_FD")
    bot = EmiliaBot(settings, restart_ready_fd=ready_fd)
    try:
        bot.run(settings.discord_token, log_handler=None)  # we configured logging ourselves above
    except discord.LoginFailure:
        raise SystemExit("Discord rejected DISCORD_TOKEN. Reset the token in the Developer Portal and update .env.") from None
    except discord.PrivilegedIntentsRequired:
        raise SystemExit(
            "Turn on 'Message Content Intent' (and 'Server Members Intent' if MEMBERS_INTENT=true): "
            "Developer Portal > your app > Bot > Privileged Gateway Intents."
        ) from None


if __name__ == "__main__":
    main()
