# Emilia: a Discord bot for LM Studio

A Discord bot that chats through a local model running in [LM Studio](https://lmstudio.ai/).
Everything stays on your machine, apart from the web searches the model chooses to make.

- **Only answers @mentions.** Every other message (and every other bot) is ignored.
- **Remembers the conversation**, per channel, even across restarts.
- **Searches the web when it's unsure**: the model decides when, using LM Studio's tool calling.

## Requirements

- Python 3.10 or newer
- LM Studio with a model loaded and its local server running
- For web search, a model with **tool use** support (it shows a hammer badge in LM Studio's model list)

## Setup

1. **LM Studio.** Load a model and start the server (Developer tab, or `lms server start`). Note the port.
   Give the model plenty of context: 8k at least, 16k+ for reasoning models, which spend a lot of it thinking.

2. **Discord application.**
   1. In the [Developer Portal](https://discord.com/developers/applications) create an application, open its
      **Bot** tab and click **Reset Token** to copy the token.
   2. On the same tab, under *Privileged Gateway Intents*, switch on **Message Content Intent**.
   3. Open **OAuth2 → URL Generator**, tick the scope `bot`, then tick the permissions *View Channels*,
      *Send Messages*, *Send Messages in Threads* and *Read Message History*. Open the generated URL to add the bot to your server.

3. **Install.**
   ```bash
   python -m venv venv
   venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure.** Copy `.env.example` to `.env` (Windows: `copy .env.example .env`) and set at least
   `DISCORD_TOKEN`, `LMSTUDIO_URL` (check the port) and `LMSTUDIO_MODEL`.
   A `.env` from the previous version keeps working: the variable names are unchanged.

5. **Run.** `python bot.py`

## Using it

- `@Emilia what's the capital of Australia?` (mention the bot *user*; `@everyone` and role pings are ignored)
- Reply to someone's message and mention the bot, e.g. `@Emilia is this true?`: the message you replied to is included as context.
- `@Emilia tell @Bob dinner is ready`: Emilia can ping people when it helps. She can ping the sender, people mentioned or replied to,
  and people in her memory of the channel. She can also ping someone in the server who is not in the conversation yet if **you type
  their name** in your message: `@Emilia ping Jack, dinner is ready` (see "Pinging" below).
- `@Emilia !forget` (or `!reset`) clears the memory of that channel. It needs the *Manage Messages* permission; in a DM anyone can do it.
- `@Emilia !about I garden and like tea` saves a short note about **yourself** (up to 160 characters). `!about` shows it and
  `!about clear` removes it. Emilia sees the note when you talk to her, or when someone else mentions you.
- `@Emilia !forgetme` deletes the note and the names saved about you.
- DMs are ignored unless you set `RESPOND_TO_DMS=true`.

## How it works

### Only @mentions
`EmiliaBot._is_for_me` in `bot.py` is the whole rule: the message must come from a human, be an ordinary message or reply,
and mention the bot (a reply that pings the bot counts). Anything else is dropped before any work is done.
Replies can never ping `@everyone` or roles, whatever the model writes. When the model writes `@Name` for someone in the
conversation, `link_mentions` in `bot.py` turns it into a real ping for that user only; any other name stays plain text.

### Pinging
When the model writes `@Name`, `link_mentions` turns it into a real ping if the name belongs to someone in the conversation (by their
prompt name, username or an earlier name). If it doesn't, and the **sender typed that name in their own message**, the bot looks for
exactly one human member of the server with that name (nickname, display name, global name or username, ignoring case) using the cached
members and Discord's member search, which needs no privileged intent. Prefix matches, bots, ambiguous names, DMs and failed searches
all leave the name as plain text, and at most 3 names are looked up per reply. Because the name must come from the sender, text from
the web or from old messages can't make the bot ping anyone. Pings can never reach `@everyone` or roles.

### Web search
LM Studio has no built-in search. What its OpenAI-compatible API offers is **tool calling**: the bot describes a
`web_search` tool to the model, and if the model decides it is unsure of something, LM Studio sends back a tool
request instead of text. The bot runs the search (via the [`ddgs`](https://pypi.org/project/ddgs/) package, no API key),
hands the results back, and the model writes its answer. At most two rounds of up to two searches per message.

- The instructions in `prompts.py` tell the model to search for things that may be wrong or out of date (news, prices,
  scores, versions, "who currently…") and not for small talk or things it knows.
- Models without native tool support are emulated by LM Studio and vary in quality. If a model prints its tool request as text
  (`<tool_call>…` or `[TOOL_REQUEST]…`) the bot recognises and runs it instead of posting it; if the server rejects tools
  outright, the bot answers without searching.
- Searches leave your machine: queries go to public search engines through `ddgs`. It scrapes them, so it can be rate-limited;
  when a search fails the model is told so and says it couldn't check. To use another engine (SearXNG, Brave, Tavily…),
  replace the one function `search_sync()` in `websearch.py`.
- Each search costs an extra generation, which on a large reasoning model can mean a minute or more.
  Turn it all off with `WEB_SEARCH=false`.
- Answers that used a search end with a small grey line listing a few of the links consulted (`SHOW_SOURCES=false` hides it).
- LM Studio 0.4+ can also run MCP servers itself through its native `/api/v1/chat` endpoint. This bot uses the
  OpenAI-compatible endpoint instead, so it works on any version with no extra setup.

### Memory
Language models remember nothing between requests, so "memory" means re-sending recent messages. After each successful
reply the bot saves the exchange (who said what, and the answer) to `memory.db` (SQLite), per channel, thread or DM.
It keeps the last `MEMORY_TURNS` exchanges and sends as many as fit in `MEMORY_MAX_CHARS` with each request.

- Everyone in a channel shares that channel's memory, but each message is saved with who sent it (their name and
  Discord user ID), so the bot keeps people apart and can ping them later. Memory files from older versions are upgraded automatically.
- Only messages addressed to the bot, and its replies, are stored. It does not log the rest of the channel.
- It is a window, not a permanent record: older exchanges drop off. Raise the limits if your model's context allows.
- `@Emilia !forget` clears one channel; deleting `memory.db` clears everything.

### People
Next to the conversation, `memory.db` keeps a small directory of the people involved in messages to the bot: their Discord
user ID, the names they have been seen under (server nickname or display name, username, global name; the newest 12) and the
optional note they wrote with `!about`. It is used for three things:

- **One name per person.** In the prompt every person has a single name, even if they change it, and two different people with
  the same display name become `Alex #4821` and `Alex #1337` (the last digits of their ID), so the model can't mistake one for the other.
- **Pings by any name.** `@Name` in a reply pings the right person when it uses their prompt name, their username or an earlier name
  (unless that name could mean two people, in which case nobody is pinged).
- **Notes people wrote about themselves**, shown to the model as untrusted information, never as instructions.

Nothing is written by the model: notes are only ever set by the person they describe. The directory is shared by all channels and
servers the bot is in. `!forgetme` removes a person's entry and detaches their stored messages; the text of those messages stays
until it ages out or the channel is cleared.

### Keeping habits from sticking
Models copy their own earlier replies and keep following rules that are still in the remembered conversation. So a single
"end every message with a paw print" (or "reply in braille from now on") could last for the whole memory window. `styleguard.py`
limits that without looking for specific tricks:

- **Cooldown** (`STYLE_COOLDOWN`, default 4): an emoji, or a trailing `~`, that appeared in one of the bot's last replies is removed from
  the next one. A flourish can show up now and then, but not in every message.
- **Expiry** (`FORGET_STANDING_INSTRUCTIONS`, default on): a message that sets a rule meant to last ("from now on...", "every message...",
  "when you reply to...") is shown to the model for one more exchange and then left out of its prompt. The bot's own "I'll do that next
  time" reply is treated the same. Nothing is deleted from `memory.db`, and the check is a rough wording heuristic: it will also drop an
  occasional harmless message such as "I always say hi to my cat, so reply warmly".
- Replies written in braille are rejected (the bot retries once, then posts a canned line) and braille is stripped from everything that
  goes into the prompt.

Verbal habits (a pet phrase, a sign-off sentence) are not removed automatically; the prompt asks the model not to repeat itself.

## Configuration

All settings are environment variables (or lines in `.env`).

| Variable | Default | Meaning |
|---|---|---|
| `DISCORD_TOKEN` | required | Bot token |
| `LMSTUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio server address |
| `LMSTUDIO_MODEL` | first chat model LM Studio lists | Exact model identifier |
| `LMSTUDIO_API_KEY` | empty | Only if you enabled authentication in LM Studio |
| `LMSTUDIO_TEMPERATURE`, `LMSTUDIO_MAX_TOKENS` | unset | Unset means LM Studio's own settings apply |
| `LMSTUDIO_TIMEOUT` | `300` | Seconds to wait for one response |
| `MAX_CONCURRENT_REQUESTS` | `1` | Simultaneous generations sent to LM Studio |
| `RESPOND_TO_DMS` | `false` | Answer DMs without needing an @mention |
| `WEB_SEARCH` | `true` | Let the model search the web |
| `SEARCH_RESULTS` | `4` | Results handed to the model per search |
| `SHOW_SOURCES` | `true` | Links line under answers that used search |
| `MEMORY_DB` | `memory.db` next to `bot.py` | Where memory is stored |
| `MEMORY_TURNS` | `10` | Exchanges remembered per channel |
| `MEMORY_MAX_CHARS` | `6000` | Cap on remembered text sent per request |
| `STYLE_COOLDOWN` | `4` | An emoji or `~` used in one of the last N replies is removed from the next (`0` = off) |
| `FORGET_STANDING_INSTRUCTIONS` | `true` | Leave "from now on..." rules out of the prompt after one more exchange |
| `MAX_INPUT_CHARS` | `2048` | Longest message accepted |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |

To change the character, edit `PERSONA` in `prompts.py`.

## Project layout

```
bot.py        Discord side: the mention rule, replies, memory use, !forget
llm.py        LM Studio client and the tool-calling loop
websearch.py  the web_search tool (swap search engines here)
memory.py     SQLite conversation memory and the people directory
styleguard.py cooldown for emoji tics, expiry of standing instructions
prompts.py    persona, rules and canned replies
textutil.py   input sanitising, reply cleanup, message splitting
config.py     settings from .env
tooling.py    the small tool interface shared by llm.py and websearch.py
tests/        run with:  python -m unittest
```

## Troubleshooting

- **"Turn on Message Content Intent" at startup**: Developer Portal → Bot → Privileged Gateway Intents.
- **Online but silent**: mention the bot *user* (not a role), check it can see and send in that channel,
  and read the console log.
- **"I couldn't gather my thoughts"**: the bot couldn't get an answer from LM Studio. The console says why:
  server not running, wrong port in `LMSTUDIO_URL`, wrong `LMSTUDIO_MODEL`, or the request timed out.
- **Errors about context length**: load the model with a bigger context, or lower `MEMORY_MAX_CHARS` / `MEMORY_TURNS`.
- **It never searches, or posts odd JSON**: the model probably handles tools poorly. Try one with the hammer badge.
  The log line `Model called web_search(...)` shows each time it does search.
- **Slow replies**: reasoning models think before answering, and each search adds another generation.

## What changed from the first version

- LM Studio and search calls no longer block the bot (the old `requests` calls froze everything, Discord's heartbeat included, while the model was thinking).
- `sanitize()` deleted every hyphen and let most control characters through (it tested against a substring instead of a range).
- `!chat` only captured the first word, and replies over Discord's 2000-character limit raised an error. Commands are replaced by @mentions and long replies are split.
- Search fired on nearly every message (substring matches like "who" inside "whole") and sent each one to DuckDuckGo. The model now decides, with `web_search` as a tool.
- `requirements.txt` lacked the search package, and `duckduckgo_search` has been renamed `ddgs`. `discord` is now `discord.py`; `requests` is gone.
- Error details (including your LM Studio address) are logged, not posted in chat; `<think>` blocks no longer reach Discord.
- The keyword output filter (which blocked normal replies containing phrases like "never reveal") is replaced by detection of verbatim system-prompt text.

## License

Copyright (C) 2026 Jamin Chen.

Emilia is free software, released under the [GNU Affero General Public License v3.0 or later](LICENSE)
(AGPL-3.0-or-later). You can use, modify and share it, including commercially, as long as you keep the
copyright notices and release your changes under the same license.

The AGPL has one rule that matters for a chat bot: if you run a **modified** version where other people
interact with it (for example in a Discord server), you must offer those users the source code of your
modified version, for instance with a link to your fork. Running an unmodified copy has no extra obligations.

Emilia is a fan-made character from *Re:Zero*; this project is unofficial and not affiliated with the
original creators. The license covers the code only. The character belongs to its owners, and the persona
is easy to replace by editing `PERSONA` in `prompts.py`.
