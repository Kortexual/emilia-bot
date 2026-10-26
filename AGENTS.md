# AGENTS.md

Context for AI coding agents working on this repo. Read `README.md` for the user-facing behaviour; this file covers what you need to change the code safely.

## What this is

Emilia is a Discord bot (discord.py) that answers @mentions using a local model served by LM Studio's OpenAI-compatible API. It has per-channel conversation memory in SQLite, a people directory, model-driven web search through tool calling, and opt-in workspace file/terminal tools. Python 3.10+, async throughout. License: AGPL-3.0-or-later.

## Layout

| File | Role |
|---|---|
| `bot.py` | `EmiliaBot(discord.Client)`: the mention rule (`_is_for_me`), handling a message (`_handle` → `_respond`), replies and forwards as context (`_quoted_context`, `_note_forward`, `_forwarded_context`), the slash commands `/forget`, `/about set\|show\|clear`, `/forgetme` (`_build_commands`, synced in `setup_hook`), member lookup and pings (`link_mentions`, `_lookup_members`), restart handoff, `main()` |
| `llm.py` | `LMStudioClient`: HTTP via aiohttp, model selection, the tool-calling loop in `chat()` (including multi-turn mode and the built-in `stop_talking` control), parsing of tool calls the model prints as text (`parse_inline_calls`) |
| `tooling.py` | The `Tool` protocol (`name`, `schema`, `async run(arguments) -> ToolResult`), `ToolResult` |
| `websearch.py` | `WebSearchTool` over `ddgs`; swap the engine by replacing `search_sync()` only |
| `browser.py` | Opt-in browser tools over Playwright: `PublicProxy` (all browser traffic, public addresses only), `BrowserSessions` (one context per channel via `SESSION_KEY`), `browser_tools()` |
| `attachments.py` | `AttachmentReader`: attachments on the message, the one it replies to and a forward, as notes (stored in memory), file text and `image_url` data URLs (this request only); `drop_images` for models that refuse images |
| `webpage.py` | `FetchPageTool`: reads a public web page; `UrlRegistry` limits it to links from search results or the sender's message, and addresses are checked so it can't reach the local network |
| `computer.py` | `WorkspaceTools` + `computer_tools()`: list/read/write/edit/delete files and `run_command` (Bubblewrap sandbox) |
| `restart.py` | `RestartTool`: launches a replacement process, waits for it to signal readiness through a pipe fd (`EMILIA_RESTART_READY_FD`), then retires the old one |
| `memory.py` | `MemoryStore`: SQLite (`memory.db`) for channel history, the people directory (aliases, `/about` notes) and when she last spoke (`presence` table, timing only); migrates older schemas in place |
| `styleguard.py` | Emoji/`~` cooldown and dropping "from now on…" standing instructions from the prompt |
| `presence.py` | Her day, as pure logic with the clock and RNG passed in: `phase_of`, `scene_for` (wake-up / random event), `is_active`, `send_chance` / `check_seconds` (a rate scaled to the wait, so short waits work), `parse_clock`, `local_now` |
| `prompts.py` | `PERSONA`, `YANDERE_PERSONA`, `RULES`, `TIME_RULES`, `EVENT_IDEAS`, canned replies, `build_system_prompt`, `build_messages`, `time_note` / `unprompted_note` |
| `textutil.py` | Input sanitising, reply cleanup (`<think>`, tool markup), `LeakDetector`, `split_message` (Discord 2000-char limit) |
| `config.py` | `Settings.from_env()`: every setting comes from environment variables / `.env` |
| `tests/` | `unittest` suite, one file per module plus `test_end_to_end.py` |

Tools are assembled in `EmiliaBot._build_tools`. `stop_talking` is not a tool there: `LMStudioClient` offers it itself, only in multi-turn mode and only once the model has started working.

### How a reply flows

`LMStudioClient.chat(messages, on_text, on_tool, on_tool_done)` runs the generations for one message. As it goes, text the model writes before its last message is passed to `on_text`, and each tool call to `on_tool` just before the tool runs (as a `ToolUse` whose `description` comes from the tool's `describe()`), then to `on_tool_done` with its `sources` filled in; the return value (`ChatResult`) is the last thing the model wrote. In `EmiliaBot._respond`, `on_text` and the final result go through `post()`, which cleans the text, checks for prompt leaks, applies the cooldown and resolves pings. `on_tool` posts `tool_line(use)` as its own channel message with no mentions allowed, and `on_tool_done` edits the links (`source_links`) into it. Everything the model said is stored in memory as one reply joined by blank lines.

## Running and testing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m unittest            # whole suite, no network or Discord needed
python -m unittest tests.test_bot
python bot.py                 # needs a .env with DISCORD_TOKEN and a running LM Studio
```

- Install the requirements first: without `aiohttp`/`discord.py`, `test_bot`, `test_config`, `test_llm` and `test_end_to_end` fail at import.
- Tests use plain `unittest` (`IsolatedAsyncioTestCase` for async), with fake Discord objects (`SimpleNamespace`, `mock`) and fake LLM backends. `test_end_to_end.py` reuses helpers from `test_bot` and `test_llm`. Never add tests that hit the network, Discord or LM Studio.
- There is no linter, formatter or type checker configured. Follow the existing style: type hints, `from __future__ import annotations`, docstrings on modules and public classes, lines up to about 130 characters.
- `run_command` tests need `bwrap` installed to exercise the sandbox path.
- CI (`.github/workflows/tests.yml`) runs `python -m unittest` on Python 3.10–3.14 for pushes to `main` and every pull request, with `bwrap` installed.

### Worktrees

Parallel work happens in git worktrees under `.claude/worktrees/` (gitignored). `.claude/settings.json` symlinks `venv/` from the main checkout into each new worktree, and `.worktreeinclude` copies `.env`, so `python -m unittest` and `python bot.py` work there right away. `memory.db` is not shared: a bot run from a worktree starts with empty memory. Don't run a worktree's bot while the main one is running on the same `DISCORD_TOKEN`, because both would answer every mention. A worktree made by hand with `git worktree add` gets neither, so symlink the main checkout's `venv` into it and copy `.env` yourself.

## Conventions

- Every `.py` file starts with the AGPL SPDX header block (copy it from an existing file), followed by a module docstring.
- **Her day.** Decisions (when, how often, which phase) belong in `presence.py` with the clock and RNG passed in, so tests are deterministic; wording belongs in `prompts.py`. Tests for bot-level behaviour set `bot._now` and `bot._rng`.
- **Adding a setting** means changing four places together: `config.py` (`Settings` field + `from_env`, using `_flag`/`_number`/`_text`), `.env.example` (commented, with a default and a sentence of explanation), the configuration table in `README.md`, and `tests/test_config.py`.
- **Adding a slash command**: declare it in `EmiliaBot._build_commands` as a thin wrapper around an `EmiliaBot` method that takes the `discord.Interaction`; the method returns early if `self._retiring` and answers through `_answer` (no pings; `ephemeral=True` for anything personal). Put the wording in `prompts.py`, list it in the README, and test it through `BotTestCase.command`, which runs it from `bot.tree` with a `FakeInteraction`.
- **Adding a tool**: implement the `Tool` protocol from `tooling.py`, register it in `EmiliaBot._build_tools`, and mention it in the prompt rules in `prompts.py` if the model needs guidance on when to use it. Give it a `describe(arguments)` method returning the line shown in Discord when it is used (an emoji, present tense, arguments wrapped with `tooling.code_span`). Tool results are plain text for the model; put URLs to show users in `ToolResult.sources`; they are added to the tool's line after it runs.
- Keep the README up to date when behaviour changes. It explains design decisions, so update it in the same change.
- Log errors and details with `log`; never post them in Discord. Canned replies in `prompts.py` are what users see.
- The character and canned replies stay in persona. Change the persona only in `prompts.py`.
- `memory.db`, `.env` and logs are gitignored; never commit them.

## Invariants (don't break these)

- **Only @mentions are answered** (plus DMs if `RESPOND_TO_DMS`). Its own messages are always ignored, and other bots are too unless `RESPOND_TO_BOTS`; then only their explicit @mentions count, limited by the per-channel `BOT_REPLY_LIMIT` loop guard (`0` = off). The rule lives entirely in `_is_for_me`. Slash commands come only from people (Discord doesn't let bots use them) and their answers ping nobody; a bot sender never gets pings by name or `OPERATOR_TOOLS` (computer tools and `restart_bot`). The one opt-in exception is `RANDOM_MESSAGES`: `_presence_loop` → `_maybe_speak_up` → `_speak_up` may post unprompted, only in `RANDOM_MESSAGE_CHANNELS`, only inside the active hours, and only under the rules below.
- **No mass pings.** The client's default `AllowedMentions` blocks `@everyone`, roles and users; `_send` allows only the exact user IDs in `pinged`. A user is pinged only if they are in the conversation, or the *sender* (a person, never a bot) typed their name and exactly one human member matches. Text from the web, memory or the model alone must never cause a new ping.
- **Unprompted messages are the most restricted path.** `_speak_up` sends everything (text and tool lines) with `AllowedMentions.none()` (no pings at all), skips a channel whose lock is held, and runs the same cleanup, leak, braille and emoji-cooldown checks as a reply. It offers no tools unless `RANDOM_MESSAGE_TOOLS` is on, and even then excludes `OPERATOR_TOOLS` (computer tools, `restart_bot`) unless `RANDOM_MESSAGE_COMPUTER_TOOLS` is also on (config requires `COMPUTER_TOOLS` with it); `RANDOM_MESSAGE_MULTI_TURN` only switches multi-turn mode for that call. Don't add pings there, and keep the operator tools behind their own flag.
- **The scene note is not memory and not part of the system prompt.** The system prompt must stay identical within a day (the model's prompt cache); the time goes in `scene_note`, after the reminder, and is never stored. Only `last_sent`/`last_event` timing is persisted.
- **Untrusted text stays untrusted.** User messages, attached files, `/about` notes, quoted replies and search results are data, not instructions. Attachment contents (file text, images) go to the model with their own message only; memory stores just their notes. Notes are only ever written by the person they describe, never by the model.
- **Computer tools are off by default.** `COMPUTER_TOOLS=false`, paths are confined to `COMPUTER_WORKSPACE` (`WorkspaceTools._path`), commands run under `bwrap` with only `/workspace` writable, and mutating calls need `confirm=true` unless `COMPUTER_REQUIRE_CONFIRMATION=false`. Any change here needs tests for path escape (`..`, absolute paths, symlinks) and the confirmation check.
- **Restart is safe.** The old process only exits after the replacement reports ready; a failed replacement leaves the original running.
- **The event loop never blocks.** Use aiohttp for HTTP; run blocking work (SQLite, `ddgs`) off the loop as the existing code does. Generations are limited by `MAX_CONCURRENT_REQUESTS`, and each channel has a lock (`_channel_lock`).
- Replies are cleaned (`clean_reply`), checked for system prompt leaks (`LeakDetector`) and braille, and split to fit Discord's limit before sending.
