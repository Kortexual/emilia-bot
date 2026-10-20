# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Emilia: a Discord bot for LM Studio
# Copyright (C) 2026 YOUR NAME
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

"""Everything wired together except Discord itself and the real search engine."""
import json
import unittest
from unittest import mock

import prompts
from bot import EmiliaBot
from config import Settings
from llm import LMStudioClient
from memory import MemoryStore
from websearch import SearchHit, WebSearchTool

from tests import test_bot, test_llm


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        patcher = mock.patch.object(EmiliaBot, "user", test_bot.BOT, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_search_then_follow_up_conversation(self):
        searches = []

        def fake_search(query, max_results):
            searches.append(query)
            return [
                SearchHit("Atlanta forecast", "https://wx.example/atl", "Sunny and 75F in Atlanta today."),
                SearchHit("Hourly", "https://wx.example/hourly", "Clear skies all afternoon."),
            ]

        server = test_llm.FakeLMStudio(
            [
                # 1st message: a reasoning model "thinks", then asks for a search...
                test_llm.tool_reply(("web_search", {"query": "Atlanta weather today"}), content="<think>I should check.</think>"),
                # ...and answers from the results.
                test_llm.text_reply("<think>The results say sunny.</think>It's sunny and about 75 degrees, I think!"),
                # 2nd message: answered from memory, no search.
                test_llm.text_reply("Tomorrow might be a little cloudy, or so."),
            ]
        )
        self.addCleanup(server.stop)
        client = LMStudioClient(
            base_url=server.url, model="my-chat-model", tools=[WebSearchTool(4, search_fn=fake_search)]
        )
        await client.start()
        self.addAsyncCleanup(client.close)

        settings = Settings(discord_token="x")
        memory = MemoryStore(":memory:", settings.memory_turns)
        self.addCleanup(memory.close)
        bot = EmiliaBot(settings, llm=client, memory=memory)
        channel = test_bot.FakeChannel(channel_id=42)
        alice = test_bot.person("Alice", 1)

        # --- first message: needs the web ---------------------------------------
        first = test_bot.FakeMessage(f"<@{test_bot.BOT.id}> what's the weather in Atlanta today?", channel=channel, author=alice)
        await bot.on_message(first)

        self.assertEqual(searches, ["Atlanta weather today"])
        reply = first.replies[0]
        self.assertTrue(reply.startswith("It's sunny and about 75 degrees, I think!"), reply)
        self.assertNotIn("<think>", reply)
        self.assertIn("-# 🔎 Searched the web · <https://wx.example/atl> · <https://wx.example/hourly>", reply)

        request1, request2 = server.requests[0], server.requests[1]
        self.assertEqual(request1["messages"][0]["role"], "system")
        self.assertIn("Re:Zero", request1["messages"][0]["content"])
        self.assertEqual(
            request1["messages"][1]["content"],
            '<USER_INPUT name="Alice">\nwhat\'s the weather in Atlanta today?\n</USER_INPUT>\n\n' + prompts.REMINDER,
        )
        self.assertEqual(request1["tools"][0]["function"]["name"], "web_search")
        roles = [m["role"] for m in request2["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertIn("Sunny and 75F in Atlanta today.", request2["messages"][3]["content"])

        # What was remembered is the clean exchange: no reasoning, no footer, no tool traffic.
        stored = memory.recent("42")
        self.assertEqual([m.role for m in stored], ["user", "assistant"])
        self.assertEqual(stored[1].content, "It's sunny and about 75 degrees, I think!")

        # --- second message: uses memory ------------------------------------------
        second = test_bot.FakeMessage(f"<@{test_bot.BOT.id}> and tomorrow?", channel=channel, author=alice)
        await bot.on_message(second)

        self.assertEqual(second.replies, ["Tomorrow might be a little cloudy, or so."])
        request3 = server.requests[2]
        self.assertEqual([m["role"] for m in request3["messages"]], ["system", "user", "assistant", "user"])
        self.assertIn("weather in Atlanta", request3["messages"][1]["content"])
        self.assertEqual(request3["messages"][2]["content"], "It's sunny and about 75 degrees, I think!")
        self.assertIn("and tomorrow?", request3["messages"][3]["content"])
        self.assertEqual(len(searches), 1)  # no second search was needed

        # Nothing the bot sent could ping anyone, and every request was well-formed JSON.
        for request in server.requests:
            json.dumps(request)
        self.assertEqual(len(memory.recent("42")), 4)


if __name__ == "__main__":
    unittest.main()
