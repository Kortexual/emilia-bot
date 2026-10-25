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

"""Tests the LM Studio client against a tiny fake server that speaks the documented response format."""
import asyncio
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llm
from llm import LLMError, LLMHTTPError, LMStudioClient
from tooling import ToolResult


# --- fakes -----------------------------------------------------------------------
class FakeLMStudio:
    """Replies to POST /v1/chat/completions from a script and records every request it receives.

    A script entry is a response body (200), a (status, body) tuple, or ("sleep", seconds, body).
    """

    def __init__(self, script=(), models=("my-chat-model", "text-embedding-nomic")):
        self.script = list(script)
        self.models = list(models)
        self.requests = []  # parsed JSON bodies sent to /chat/completions
        self.headers = []  # headers of every request
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up (timeout test)

            def do_GET(self):
                outer.headers.append(dict(self.headers))
                if self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if not outer.script:
                    return self._send(500, {"error": "script exhausted"})
                entry = outer.script.pop(0)
                if isinstance(entry, tuple) and entry[0] == "sleep":
                    time.sleep(entry[1])
                    return self._send(200, entry[2])
                if isinstance(entry, tuple):
                    return self._send(*entry)
                self._send(200, entry)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def text_reply(text):
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


def tool_reply(*calls, content=None):
    """calls: (name, arguments) pairs; arguments may be a dict or a raw string."""
    tool_calls = [
        {
            "id": str(1000 + i),
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    message = {"role": "assistant", "tool_calls": tool_calls}
    if content is not None:
        message["content"] = content
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]}


class FakeSearch:
    name = "web_search"
    schema = {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }

    def __init__(self):
        self.calls = []

    async def run(self, arguments):
        self.calls.append(arguments)
        return ToolResult(f"RESULT for {arguments.get('query')}", ["https://example.com/a"])


USER = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hello?"}]


class ClientTestCase(unittest.IsolatedAsyncioTestCase):
    def serve(self, script=(), **kwargs):
        server = FakeLMStudio(script, **kwargs)
        self.addCleanup(server.stop)
        return server

    async def client(self, server, *, tools=(), **kwargs):
        kwargs.setdefault("model", "my-chat-model")
        client = LMStudioClient(base_url=server.url, tools=tools, **kwargs)
        await client.start()
        self.addAsyncCleanup(client.close)
        return client


class ChatTests(ClientTestCase):
    async def test_plain_answer(self):
        server = self.serve([text_reply("Hello!")])
        search = FakeSearch()
        client = await self.client(server, tools=[search])

        result = await client.chat(USER)

        self.assertEqual(result.text, "Hello!")
        self.assertFalse(result.used_tools)
        sent = server.requests[0]
        self.assertEqual(sent["model"], "my-chat-model")
        self.assertEqual(sent["messages"], USER)
        self.assertEqual(sent["tools"][0], {"type": "function", "function": search.schema})
        self.assertNotIn("temperature", sent)  # LM Studio's own UI settings apply unless configured
        self.assertNotIn("max_tokens", sent)

    async def test_tools_can_be_withheld_for_one_request(self):
        server = self.serve([text_reply("A thought of my own."), text_reply("Now with tools.")])
        search = FakeSearch()
        client = await self.client(server, tools=[search])

        result = await client.chat(USER, tools=False)

        self.assertEqual(result.text, "A thought of my own.")
        self.assertNotIn("tools", server.requests[0])
        self.assertNotIn("tool_choice", server.requests[0])
        await client.chat(USER)  # the next request offers them again
        self.assertEqual(server.requests[1]["tools"][0], {"type": "function", "function": search.schema})

    async def test_a_printed_tool_call_is_not_run_when_tools_are_withheld(self):
        server = self.serve([text_reply('Sure<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>')])
        search = FakeSearch()
        client = await self.client(server, tools=[search])

        result = await client.chat(USER, tools=False)

        self.assertEqual(search.calls, [])
        self.assertEqual(result.text, "Sure")
        self.assertFalse(result.used_tools)

    async def test_tool_round_trip(self):
        server = self.serve([tool_reply(("web_search", {"query": "weather atlanta"})), text_reply("It is sunny.")])
        search = FakeSearch()
        client = await self.client(server, tools=[search])

        result = await client.chat(USER)

        self.assertEqual(result.text, "It is sunny.")
        self.assertTrue(result.used_tools)
        self.assertEqual(result.sources, ["https://example.com/a"])
        self.assertEqual(search.calls, [{"query": "weather atlanta"}])

        followup = server.requests[1]["messages"]
        self.assertEqual(followup[:2], USER)
        assistant, tool = followup[2], followup[3]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "1000")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"query": "weather atlanta"})
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], "1000")  # ties the result to the request
        self.assertIn("RESULT for weather atlanta", tool["content"])

    async def test_tool_call_printed_as_text_is_rescued(self):
        for printed in (
            '<tool_call>\n{"name": "web_search", "arguments": {"query": "qwen release"}}\n</tool_call>',
            '[TOOL_REQUEST]{"name": "web_search", "arguments": {"query": "qwen release"}}[END_TOOL_REQUEST]',
            '{"name": "web_search", "parameters": {"query": "qwen release"}}',
        ):
            with self.subTest(printed=printed[:20]):
                server = self.serve([text_reply(printed), text_reply("Found it.")])
                search = FakeSearch()
                client = await self.client(server, tools=[search])
                result = await client.chat(USER)
                self.assertEqual(result.text, "Found it.")
                self.assertEqual(search.calls, [{"query": "qwen release"}])
                self.assertEqual(server.requests[1]["messages"][2]["tool_calls"][0]["function"]["name"], "web_search")

    async def test_ordinary_json_in_an_answer_is_not_mistaken_for_a_tool_call(self):
        server = self.serve([text_reply('{"name": "Emilia", "mood": "shy"}')])
        search = FakeSearch()
        result = await (await self.client(server, tools=[search])).chat(USER)
        self.assertEqual(result.text, '{"name": "Emilia", "mood": "shy"}')
        self.assertEqual(search.calls, [])

    async def test_unreadable_arguments_are_reported_back_to_the_model(self):
        server = self.serve([tool_reply(("web_search", "this is not json")), text_reply("Sorry, I couldn't check.")])
        search = FakeSearch()
        result = await (await self.client(server, tools=[search])).chat(USER)

        self.assertEqual(result.text, "Sorry, I couldn't check.")
        self.assertEqual(search.calls, [])  # never ran with garbage
        self.assertIn("Error", server.requests[1]["messages"][3]["content"])

    async def test_unknown_tool_is_reported_back_to_the_model(self):
        server = self.serve([tool_reply(("launch_missiles", {})), text_reply("No.")])
        result = await (await self.client(server, tools=[FakeSearch()])).chat(USER)
        self.assertEqual(result.text, "No.")
        self.assertIn("no tool called", server.requests[1]["messages"][3]["content"])

    async def test_a_crashing_tool_does_not_crash_the_chat(self):
        class Crashy(FakeSearch):
            async def run(self, arguments):
                raise ValueError("boom")

        server = self.serve([tool_reply(("web_search", {"query": "x"})), text_reply("Still here.")])
        with self.assertLogs("emilia.llm", level="ERROR"):
            result = await (await self.client(server, tools=[Crashy()])).chat(USER)
        self.assertEqual(result.text, "Still here.")

    async def test_tool_rounds_are_limited_and_the_last_round_must_answer(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "one"})),
                tool_reply(("web_search", {"query": "two"})),
                text_reply("Final answer."),
            ]
        )
        search = FakeSearch()
        result = await (await self.client(server, tools=[search])).chat(USER)

        self.assertEqual(result.text, "Final answer.")
        self.assertEqual(len(search.calls), llm.MAX_TOOL_ROUNDS)
        self.assertEqual(len(server.requests), llm.MAX_TOOL_ROUNDS + 1)
        self.assertIn("tools", server.requests[0])
        self.assertIn("tools", server.requests[1])
        self.assertNotIn("tools", server.requests[-1])  # no tools offered on the final round

        last = server.requests[-1]["messages"]
        tool_messages = [m["content"] for m in last if m["role"] == "tool"]
        self.assertNotIn("final answer", tool_messages[0].lower())
        self.assertIn("final answer", tool_messages[-1].lower())  # nudge attached to the last result
        ids = [m["tool_call_id"] for m in last if m["role"] == "tool"]
        self.assertEqual(len(set(ids)), len(ids))

    async def test_parallel_tool_calls_are_capped(self):
        calls = [("web_search", {"query": f"q{i}"}) for i in range(5)]
        server = self.serve([tool_reply(*calls), text_reply("done")])
        search = FakeSearch()
        await (await self.client(server, tools=[search])).chat(USER)

        self.assertEqual(len(search.calls), llm.MAX_CALLS_PER_ROUND)
        followup = server.requests[1]["messages"]
        assistant = next(m for m in followup if m["role"] == "assistant")
        tool_msgs = [m for m in followup if m["role"] == "tool"]
        self.assertEqual(len(assistant["tool_calls"]), len(tool_msgs))  # every call gets exactly one result

    async def test_server_rejecting_tools_falls_back_to_a_plain_request(self):
        server = self.serve([(400, {"error": "this model's template has no tool support"}), text_reply("Plain answer.")])
        with self.assertLogs("emilia.llm", level="WARNING"):
            result = await (await self.client(server, tools=[FakeSearch()])).chat(USER)
        self.assertEqual(result.text, "Plain answer.")
        self.assertIn("tools", server.requests[0])
        self.assertNotIn("tools", server.requests[1])

    async def test_other_http_errors_are_raised(self):
        server = self.serve([(500, {"error": "model crashed"})])
        with self.assertRaises(LLMHTTPError) as caught:
            await (await self.client(server)).chat(USER)
        self.assertEqual(caught.exception.status, 500)

    async def test_http_400_without_tools_is_not_retried(self):
        server = self.serve([(400, {"error": "context length exceeded"})])
        with self.assertRaises(LLMHTTPError):
            await (await self.client(server)).chat(USER)
        self.assertEqual(len(server.requests), 1)

    async def test_reasoning_is_stripped_from_answers(self):
        server = self.serve([text_reply("<think>The user greets me. Be shy.</think>\n\nH-hello!")])
        result = await (await self.client(server)).chat(USER)
        self.assertEqual(result.text, "H-hello!")

    async def test_reasoning_is_not_echoed_back_in_tool_rounds(self):
        server = self.serve(
            [tool_reply(("web_search", {"query": "x"}), content="<think>I should search.</think>"), text_reply("ok")]
        )
        await (await self.client(server, tools=[FakeSearch()])).chat(USER)
        assistant = server.requests[1]["messages"][2]
        self.assertEqual(assistant["content"], "")

    async def test_empty_content_gives_empty_text(self):
        server = self.serve([{"choices": [{"message": {"role": "assistant", "content": None}}]}])
        result = await (await self.client(server)).chat(USER)
        self.assertEqual(result.text, "")

    async def test_unexpected_response_shape(self):
        server = self.serve([{"unexpected": True}])
        with self.assertRaises(LLMError):
            await (await self.client(server)).chat(USER)

    async def test_no_tools_registered_means_none_offered(self):
        server = self.serve([text_reply("hi")])
        await (await self.client(server)).chat(USER)
        self.assertNotIn("tools", server.requests[0])

    async def test_optional_parameters_are_sent_only_when_configured(self):
        server = self.serve([text_reply("hi")])
        await (await self.client(server, temperature=0.3, max_tokens=256)).chat(USER)
        self.assertEqual(server.requests[0]["temperature"], 0.3)
        self.assertEqual(server.requests[0]["max_tokens"], 256)

    async def test_excluded_tools_are_neither_offered_nor_run(self):
        class Command(FakeSearch):
            name = "run_command"
            schema = {**FakeSearch.schema, "name": "run_command"}

        command, search = Command(), FakeSearch()
        server = self.serve([tool_reply(("run_command", {"query": "rm -rf"})), text_reply("No.")])
        client = await self.client(server, tools=[search, command])
        reported = []

        async def on_tool(use):
            reported.append(use.name)

        result = await client.chat(USER, None, on_tool, exclude={"run_command"})
        self.assertEqual(result.text, "No.")
        self.assertFalse(result.used_tools)
        self.assertEqual([t["function"]["name"] for t in server.requests[0]["tools"]], ["web_search"])
        self.assertEqual(command.calls, [])  # asked for anyway: refused
        self.assertEqual(reported, [])  # and not shown in the chat as if it had run
        self.assertIn("no tool called", server.requests[1]["messages"][3]["content"])

        server = self.serve([text_reply("hi"), text_reply("hi")])
        client = await self.client(server, tools=[command])
        await client.chat(USER, exclude={"run_command"})
        self.assertNotIn("tools", server.requests[0])  # nothing left to offer
        await client.chat(USER)
        self.assertIn("tools", server.requests[1])  # only for that conversation



class ToolReportTests(ClientTestCase):
    async def test_each_call_is_reported_with_its_description_before_it_runs_and_its_sources_after(self):
        search = FakeSearch()
        search.describe = lambda arguments: f"searching for {arguments['query']}"
        seen = []

        async def on_tool(use):
            seen.append(("start", use.description, list(use.sources), len(search.calls)))

        async def on_tool_done(use):
            seen.append(("done", use.description, list(use.sources), len(search.calls)))

        server = self.serve([tool_reply(("web_search", {"query": "tea"})), text_reply("Tea is nice.")])
        await (await self.client(server, tools=[search])).chat(USER, None, on_tool, on_tool_done)
        self.assertEqual(
            seen,
            [("start", "searching for tea", [], 0), ("done", "searching for tea", ["https://example.com/a"], 1)],
        )

    async def described(self, tool):
        seen = []

        async def on_tool(use):
            seen.append(use.description)

        server = self.serve([tool_reply(("web_search", {"query": "tea"})), text_reply("ok")])
        await (await self.client(server, tools=[tool])).chat(USER, None, on_tool)
        return seen

    async def test_a_tool_without_describe_gets_a_default_line(self):
        self.assertEqual(await self.described(FakeSearch()), ["🛠️ `web_search: tea`"])

    async def test_a_broken_describe_falls_back_to_the_default_line(self):
        class Broken(FakeSearch):
            def describe(self, arguments):
                raise KeyError("query")

        with self.assertLogs("emilia.llm", level="ERROR"):
            self.assertEqual(await self.described(Broken()), ["🛠️ `web_search: tea`"])


class MultiTurnTests(ClientTestCase):
    """MULTI_TURN_TOOLS: after a tool call the model keeps taking turns until it calls stop_talking."""

    async def chat(self, server, multi_turn=True, **kwargs):
        """Returns the result and what was reported along the way: ("text", ...) and ("tool", name, arguments)."""
        events = []

        async def on_text(text):
            events.append(("text", text))

        async def on_tool(use):
            events.append(("tool", use.name, use.arguments))

        client = await self.client(server, tools=[FakeSearch()], multi_turn=multi_turn, **kwargs)
        return await client.chat(USER, on_text, on_tool), events

    @staticmethod
    def offered(request):
        return [t["function"]["name"] for t in request.get("tools", [])]

    async def test_without_tools_it_is_one_ordinary_reply(self):
        server = self.serve([text_reply("Hi!")])
        result, events = await self.chat(server)
        self.assertEqual(result.text, "Hi!")
        self.assertEqual(events, [])
        self.assertEqual(self.offered(server.requests[0]), ["web_search"])  # stop_talking can't end a reply before it starts

    async def test_it_keeps_working_and_talking_until_stop_talking(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "one"}), content="Let me check."),
                text_reply("Interesting, one more thing."),
                tool_reply(("web_search", {"query": "two"})),
                tool_reply((llm.STOP_TALKING, {}), content="All done!"),
            ]
        )
        result, events = await self.chat(server)

        self.assertEqual(result.text, "All done!")
        self.assertEqual(
            events,
            [
                ("text", "Let me check."),
                ("tool", "web_search", {"query": "one"}),
                ("text", "Interesting, one more thing."),
                ("tool", "web_search", {"query": "two"}),
            ],
        )
        self.assertEqual(result.sources, ["https://example.com/a"] * 2)
        self.assertTrue(result.used_tools)
        self.assertEqual(len(server.requests), 4)
        for request in server.requests[1:]:
            self.assertEqual(self.offered(request), ["web_search", llm.STOP_TALKING])
        # After a tool result the prompt rides along with it; after plain text it is its own message.
        second, third = server.requests[1]["messages"], server.requests[2]["messages"]
        self.assertIn(llm.CONTINUE_HINT, second[-1]["content"])
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(third[-2], {"role": "assistant", "content": "Interesting, one more thing."})
        self.assertEqual(third[-1], {"role": "user", "content": llm.CONTINUE_HINT})

    async def test_stop_talking_next_to_another_tool_runs_that_tool_then_stops(self):
        server = self.serve([tool_reply(("web_search", {"query": "a"})), tool_reply(("web_search", {"query": "b"}), (llm.STOP_TALKING, {}))])
        result, events = await self.chat(server)
        self.assertEqual(result.text, "")
        self.assertEqual(events, [("tool", "web_search", {"query": "a"}), ("tool", "web_search", {"query": "b"})])
        self.assertEqual(len(server.requests), 2)

    async def test_stop_talking_is_refused_before_any_work(self):
        server = self.serve([tool_reply((llm.STOP_TALKING, {})), text_reply("Oh, hello!")])
        client = await self.client(server, tools=[FakeSearch()], multi_turn=True)
        result = await client.chat(USER)
        self.assertIn("no tool called", server.requests[1]["messages"][3]["content"])
        self.assertEqual(result.text, "Oh, hello!")

    async def test_the_number_of_turns_is_capped_and_the_last_must_answer(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"})), tool_reply(("web_search", {"query": "y"})), text_reply("Final.")])
        result, events = await self.chat(server, max_turns=3)
        self.assertEqual(result.text, "Final.")
        self.assertEqual(len(server.requests), 3)
        self.assertNotIn("tools", server.requests[-1])
        self.assertIn("last turn", server.requests[-1]["messages"][-1]["content"])

    async def test_talking_on_the_last_turn_but_one_ends_the_reply(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"})), text_reply("Here's what I found.")])
        result, events = await self.chat(server, max_turns=3)
        self.assertEqual(result.text, "Here's what I found.")
        self.assertEqual(len(server.requests), 2)  # no extra prompt that would only get the message repeated

    async def test_repeating_the_last_message_ends_the_reply_without_posting_it_again(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "x"})),
                text_reply("The capital of Australia is Canberra!"),
                text_reply("The capital of Australia is Canberra! 😊"),
            ]
        )
        with self.assertLogs("emilia.llm", level="INFO"):
            result, events = await self.chat(server)
        self.assertEqual(result.text, "")
        self.assertEqual(events, [("tool", "web_search", {"query": "x"}), ("text", "The capital of Australia is Canberra!")])
        self.assertEqual(len(server.requests), 3)

    async def test_saying_nothing_while_working_ends_the_reply(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"}), content="Done, it's Canberra."), text_reply("")])
        result, events = await self.chat(server)
        self.assertEqual(result.text, "")
        self.assertEqual(events, [("text", "Done, it's Canberra."), ("tool", "web_search", {"query": "x"})])
        self.assertEqual(len(server.requests), 2)

    async def test_a_repeat_next_to_a_tool_call_is_not_posted_but_the_tool_still_runs(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "x"}), content="Checking the weather now."),
                tool_reply(("web_search", {"query": "y"}), content="Checking the weather now!"),
                tool_reply((llm.STOP_TALKING, {})),
            ]
        )
        result, events = await self.chat(server)
        self.assertEqual(
            events,
            [("text", "Checking the weather now."), ("tool", "web_search", {"query": "x"}), ("tool", "web_search", {"query": "y"})],
        )

    async def test_while_working_every_request_requires_a_tool_call(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "x"})),
                tool_reply((llm.STOP_TALKING, {}), content="It's Canberra."),
            ]
        )
        result, events = await self.chat(server)
        self.assertEqual(result.text, "It's Canberra.")  # its last words come with stop_talking
        self.assertNotIn("tool_choice", server.requests[0])  # a plain chat reply stays possible at first
        self.assertEqual(server.requests[1]["tool_choice"], "required")

    async def test_images_a_model_cannot_see_are_dropped_and_the_request_retried(self):
        server = self.serve([(400, {"error": "Model does not support images"}), text_reply("I can't see it, sorry!")])
        client = await self.client(server)
        parts = [{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        messages = [USER[0], {"role": "user", "content": parts}]

        result = await client.chat(messages)

        self.assertEqual(result.text, "I can't see it, sorry!")
        self.assertEqual(server.requests[0]["messages"][1]["content"], parts)
        retried = server.requests[1]["messages"][1]["content"]
        self.assertIsInstance(retried, str)
        self.assertTrue(retried.startswith("what is this?"))
        self.assertIn("can't see images", retried)
        self.assertEqual(messages[1]["content"], parts)  # the caller's list is left alone

    async def test_a_rejected_request_without_images_still_fails(self):
        server = self.serve([(400, {"error": "bad request"})])
        client = await self.client(server)
        with self.assertRaises(LLMHTTPError):
            await client.chat(USER, tools=False)
        self.assertEqual(len(server.requests), 1)

    async def test_a_server_that_rejects_tool_choice_is_asked_again_without_it(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "x"})),
                (400, {"error": "tool_choice 'required' is not supported"}),
                tool_reply(("web_search", {"query": "y"})),
                tool_reply((llm.STOP_TALKING, {})),
            ]
        )
        with self.assertLogs("emilia.llm", level="WARNING"):
            result, events = await self.chat(server)
        self.assertEqual(len(server.requests), 4)
        self.assertEqual(server.requests[1]["tool_choice"], "required")
        self.assertNotIn("tool_choice", server.requests[2])
        self.assertIn("tools", server.requests[2])  # tools are still offered; only tool_choice is dropped
        self.assertNotIn("tool_choice", server.requests[3])  # and it isn't tried again

    async def test_the_last_turn_requires_no_tool(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"})), tool_reply(("web_search", {"query": "y"})), text_reply("Final.")])
        await self.chat(server, max_turns=3)
        self.assertNotIn("tool_choice", server.requests[-1])

    async def test_the_same_tool_call_twice_in_a_row_is_not_run_again_and_a_third_time_ends_the_turn(self):
        same = ("web_search", {"query": "x"})
        server = self.serve([tool_reply(same), tool_reply(same), tool_reply(same)])
        search = FakeSearch()
        client = await self.client(server, tools=[search], multi_turn=True)
        with self.assertLogs("emilia.llm", level="INFO") as logs:
            result = await client.chat(USER)
        self.assertEqual(result.text, "")
        self.assertEqual(len(search.calls), 1)  # run once only
        self.assertEqual(len(server.requests), 3)
        self.assertIn("call stop_talking now", server.requests[2]["messages"][-1]["content"])  # told why, and how to stop
        self.assertTrue(any("kept repeating itself" in line for line in logs.output))

    async def test_repeating_a_message_next_to_reworded_tool_calls_ends_the_turn(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "canberra population"}), content="Let me look that up for you!"),
                tool_reply(("web_search", {"query": "population of canberra"}), content="Let me look that up for you!"),
                tool_reply(("web_search", {"query": "canberra population 2026"}), content="Let me look that up for you!"),
                tool_reply(("web_search", {"query": "never reached"})),
            ]
        )
        with self.assertLogs("emilia.llm", level="INFO") as logs:
            result, events = await self.chat(server)
        self.assertEqual(result.text, "")
        self.assertEqual(len(server.requests), 3)
        self.assertEqual([e for e in events if e[0] == "text"], [("text", "Let me look that up for you!")])  # posted once
        self.assertTrue(any("kept repeating itself" in line for line in logs.output))

    async def test_the_same_call_after_something_else_is_run_again(self):
        server = self.serve(
            [
                tool_reply(("web_search", {"query": "x"})),
                tool_reply(("web_search", {"query": "y"})),
                tool_reply(("web_search", {"query": "x"})),
                tool_reply((llm.STOP_TALKING, {})),
            ]
        )
        search = FakeSearch()
        await (await self.client(server, tools=[search], multi_turn=True)).chat(USER)
        self.assertEqual([c["query"] for c in search.calls], ["x", "y", "x"])

    async def test_ending_the_turn_with_stop_talking_is_logged(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"})), tool_reply((llm.STOP_TALKING, {}))])
        with self.assertLogs("emilia.llm", level="INFO") as logs:
            await self.chat(server)
        self.assertTrue(any("Model called stop_talking: turn over" in line for line in logs.output))

    async def test_new_text_is_not_mistaken_for_a_repeat(self):
        self.assertFalse(llm._repeats("Now I'll fix the second file.", ["Now I'll fix the first file and test it."]))
        self.assertTrue(llm._repeats("hello", ["Hello!"]))

    async def test_without_multi_turn_stop_talking_is_never_offered(self):
        server = self.serve([tool_reply(("web_search", {"query": "x"}), content="Checking!"), text_reply("Sunny.")])
        result, events = await self.chat(server, multi_turn=False)
        self.assertEqual(result.text, "Sunny.")
        self.assertEqual(events, [("text", "Checking!"), ("tool", "web_search", {"query": "x"})])
        self.assertTrue(all(llm.STOP_TALKING not in self.offered(r) for r in server.requests))


class ConnectionTests(ClientTestCase):
    async def test_model_is_auto_detected_and_embedding_models_are_skipped(self):
        server = self.serve([text_reply("hi")], models=("text-embedding-nomic", "qwen-chat", "other-chat"))
        client = await self.client(server, model="")
        self.assertEqual(client.model, "qwen-chat")
        await client.chat(USER)
        self.assertEqual(server.requests[0]["model"], "qwen-chat")

    async def test_configured_model_is_used_even_if_unlisted(self):
        server = self.serve([text_reply("hi")])
        with self.assertLogs("emilia.llm", level="WARNING"):  # warns about the mismatch, but proceeds
            client = await self.client(server, model="something-else")
        await client.chat(USER)
        self.assertEqual(server.requests[0]["model"], "something-else")

    async def test_api_key_is_sent_only_when_set(self):
        server = self.serve([text_reply("a"), text_reply("b")])
        with_key = await self.client(server, api_key="s3cret")
        await with_key.chat(USER)
        self.assertEqual(server.headers[-1].get("Authorization"), "Bearer s3cret")

        without_key = await self.client(server)
        await without_key.chat(USER)
        self.assertNotIn("Authorization", server.headers[-1])

    async def test_unreachable_lm_studio_starts_quietly_then_fails_clearly(self):
        server = self.serve()
        url = server.url
        await asyncio.to_thread(server.stop)  # nothing is listening any more
        client = LMStudioClient(base_url=url, model="m")
        with self.assertLogs("emilia.llm", level="WARNING"):
            await client.start()  # must not raise: LM Studio may simply not be running yet
        self.addAsyncCleanup(client.close)
        with self.assertRaises(LLMError) as caught:
            await client.chat(USER)
        self.assertIn("cannot reach LM Studio", str(caught.exception))

    async def test_late_start_of_lm_studio_is_picked_up(self):
        server = self.serve([text_reply("hi")])
        client = LMStudioClient(base_url=server.url, model="")  # model unknown, as if LM Studio started late
        client._session = None
        await client.start()
        client.model = ""  # pretend detection failed at startup
        await client.chat(USER)
        self.assertEqual(client.model, "my-chat-model")
        await client.close()

    async def test_slow_model_times_out_cleanly(self):
        server = self.serve([("sleep", 1.5, text_reply("too late"))])
        client = await self.client(server, timeout=0.3)
        with self.assertRaises(LLMError) as caught:
            await client.chat(USER)
        self.assertIn("timed out", str(caught.exception))

    async def test_using_the_client_before_start_is_an_error(self):
        client = LMStudioClient(base_url="http://127.0.0.1:9/v1", model="m")
        with self.assertRaises(LLMError):
            await client.chat(USER)


if __name__ == "__main__":
    unittest.main()
