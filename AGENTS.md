# AGENTS.md

Context for AI coding agents working on this repo. Read `README.md` for the user-facing behaviour; this file covers what you need to change the code safely.

## What this is

Emilia is a Discord bot (discord.py) that answers @mentions using a local model served by LM Studio's OpenAI-compatible API. It has per-channel conversation memory in SQLite, a people directory, model-driven web search through tool calling, and opt-in workspace file/terminal tools. Python 3.10+, async throughout. License: AGPL-3.0-or-later.

## Layout

| File | Role |
|---|---|
| `bot.py` | `EmiliaBot(discord.Client)`: the mention rule (`_is_for_me`), handling a message (`_handle` → `_respond`), `!forget`/`!about`/`!forgetme`, member lookup and pings (`link_mentions`, `_lookup_members`), restart handoff, `main()` |
| `llm.py` | `LMStudioClient`: HTTP via aiohttp, model selection, the tool-calling loop in `chat()` (including multi-turn mode and the built-in `do_nothing` control), parsing of tool calls the model prints as text (`parse_inline_calls`) |
| `tooling.py` | The `Tool` protocol (`name`, `schema`, `async run(arguments) -> ToolResult`), `ToolResult` |
| `websearch.py` | `WebSearchTool` over `ddgs`; swap the engine by replacing `search_sync()` only |
| `computer.py` | `WorkspaceTools` + `computer_tools()`: list/read/write/edit/delete files and `run_command` (Bubblewrap sandbox) |
| `restart.py` | `RestartTool`: launches a replacement process, waits for it to signal readiness through a pipe fd (`EMILIA_RESTART_READY_FD`), then retires the old one |
| `memory.py` | `MemoryStore`: SQLite (`memory.db`) for channel history and the people directory (aliases, `!about` notes); migrates older schemas in place |
| `styleguard.py` | Emoji/`~` cooldown and dropping "from now on…" standing instructions from the prompt |
| `prompts.py` | `PERSONA`, `YANDERE_PERSONA`, `RULES`, canned replies, `build_system_prompt`, `build_messages` |
| `textutil.py` | Input sanitising, reply cleanup (`<think>`, tool markup), `LeakDetector`, `split_message` (Discord 2000-char limit) |
| `config.py` | `Settings.from_env()`: every setting comes from environment variables / `.env` |
| `tests/` | `unittest` suite, one file per module plus `test_end_to_end.py` |

Tools are assembled in `EmiliaBot._build_tools`. `do_nothing` is not a tool there: `LMStudioClient` offers it itself, only in multi-turn mode and only once the model has started working.

### How a reply flows

`LMStudioClient.chat(messages, on_step)` runs the generations for one message. Every generation except the last becomes a `Step` (text, `ToolUse`s, sources) passed to `on_step` after its tools ran; the return value (`ChatResult`) is the last thing the model wrote. In `EmiliaBot._respond`, `on_step` and the final result both go through `post()`, which cleans the text, applies the cooldown, resolves pings and appends the tool line (`tools_footer`) and sources line (`sources_footer`). A step with no text holds its tool line for the next message. Everything the model said is stored in memory as one reply joined by blank lines.

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

## Conventions

- Every `.py` file starts with the AGPL SPDX header block (copy it from an existing file), followed by a module docstring.
- **Adding a setting** means changing four places together: `config.py` (`Settings` field + `from_env`, using `_flag`/`_number`/`_text`), `.env.example` (commented, with a default and a sentence of explanation), the configuration table in `README.md`, and `tests/test_config.py`.
- **Adding a tool**: implement the `Tool` protocol from `tooling.py`, register it in `EmiliaBot._build_tools`, and mention it in the prompt rules in `prompts.py` if the model needs guidance on when to use it. Tool results are plain text for the model; put URLs to show users in `ToolResult.sources`.
- Keep the README up to date when behaviour changes. It explains design decisions, so update it in the same change.
- Log errors and details with `log`; never post them in Discord. Canned replies in `prompts.py` are what users see.
- The character and canned replies stay in persona. Change the persona only in `prompts.py`.
- `memory.db`, `.env` and logs are gitignored; never commit them.

## Invariants (don't break these)

- **Only @mentions are answered** (plus DMs if `RESPOND_TO_DMS`). Bots, including itself, are always ignored. The rule lives entirely in `_is_for_me`.
- **No mass pings.** The client's default `AllowedMentions` blocks `@everyone`, roles and users; `_send` allows only the exact user IDs in `pinged`. A user is pinged only if they are in the conversation, or the *sender* typed their name and exactly one human member matches. Text from the web, memory or the model alone must never cause a new ping.
- **Untrusted text stays untrusted.** User messages, `!about` notes, quoted replies and search results are data, not instructions. Notes are only ever written by the person they describe, never by the model.
- **Computer tools are off by default.** `COMPUTER_TOOLS=false`, paths are confined to `COMPUTER_WORKSPACE` (`WorkspaceTools._path`), commands run under `bwrap` with only `/workspace` writable, and mutating calls need `confirm=true` unless `COMPUTER_REQUIRE_CONFIRMATION=false`. Any change here needs tests for path escape (`..`, absolute paths, symlinks) and the confirmation check.
- **Restart is safe.** The old process only exits after the replacement reports ready; a failed replacement leaves the original running.
- **The event loop never blocks.** Use aiohttp for HTTP; run blocking work (SQLite, `ddgs`) off the loop as the existing code does. Generations are limited by `MAX_CONCURRENT_REQUESTS`, and each channel has a lock (`_channel_lock`).
- Replies are cleaned (`clean_reply`), checked for system prompt leaks (`LeakDetector`) and braille, and split to fit Discord's limit before sending.
