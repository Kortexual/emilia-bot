# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio
import os
import sys
import tempfile
import time
import unittest
import unittest.mock

from restart import RestartTool


class RestartToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_ready_replacement_before_handoff(self):
        handed_off = False

        async def handoff():
            nonlocal handed_off
            handed_off = True

        script = "import os; os.write(int(os.environ['EMILIA_RESTART_READY_FD']), b'1')"
        tool = RestartTool([sys.executable, "-c", script], handoff, ready_timeout=2)
        result = await tool.run({"confirm": True})
        self.assertIn("connected and ready", result.text)
        self.assertTrue(handed_off)

    async def test_failed_replacement_leaves_old_process_in_place(self):
        handed_off = False

        async def handoff():
            nonlocal handed_off
            handed_off = True

        tool = RestartTool([sys.executable, "-c", "raise SystemExit(4)"], handoff, ready_timeout=2)
        result = await tool.run({"confirm": True})
        self.assertIn("current bot is still running", result.text)
        self.assertFalse(handed_off)

    async def test_confirmation_is_enforced_when_enabled(self):
        async def handoff():
            pass

        tool = RestartTool([sys.executable, "-c", "pass"], handoff)
        result = await tool.run({})
        self.assertIn("Confirmation required", result.text)


class RestartSafetyTests(unittest.IsolatedAsyncioTestCase):
    READY = "import os; os.write(int(os.environ['EMILIA_RESTART_READY_FD']), b'1')"

    async def handoff(self):
        self.handoffs += 1

    def setUp(self):
        self.handoffs = 0

    async def test_the_replacement_gets_the_given_environment_not_ours(self):
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "env")
            script = f"import os; open({out!r}, 'w').write(os.environ.get('FROM_DOTENV', '-') + os.environ['KEEP']); " + self.READY
            tool = RestartTool([sys.executable, "-c", script], self.handoff, environment={"KEEP": "yes"}, ready_timeout=5)
            with unittest.mock.patch.dict(os.environ, {"FROM_DOTENV": "stale"}):
                result = await tool.run({"confirm": True})
            self.assertIn("connected and ready", result.text)
            with open(out) as f:
                self.assertEqual(f.read(), "-yes")

    async def test_only_one_replacement_is_ever_started(self):
        tool = RestartTool([sys.executable, "-c", self.READY], self.handoff, ready_timeout=5)
        first, second = await asyncio.gather(tool.run({"confirm": True}), tool.run({"confirm": True}))
        self.assertIn("connected and ready", first.text)
        self.assertIn("already been restarted", second.text)
        self.assertEqual(self.handoffs, 1)

    async def test_a_replacement_that_never_connects_is_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_file = os.path.join(folder, "pid")
            script = f"import os, time; open({pid_file!r}, 'w').write(str(os.getpid())); time.sleep(60)"
            tool = RestartTool([sys.executable, "-c", script], self.handoff, ready_timeout=1)
            result = await tool.run({"confirm": True})
            self.assertIn("did not become ready", result.text)
            with open(pid_file) as f:
                pid = int(f.read())
        with self.assertRaises(ProcessLookupError):  # terminated and reaped, not left as a zombie
            os.kill(pid, 0)
        self.assertEqual(self.handoffs, 0)

    async def test_a_replacement_that_closes_the_pipe_without_connecting_is_stopped(self):
        script = "import os, time; os.close(int(os.environ['EMILIA_RESTART_READY_FD'])); time.sleep(60)"
        tool = RestartTool([sys.executable, "-c", script], self.handoff, ready_timeout=5)
        started = time.monotonic()
        result = await tool.run({"confirm": True})
        self.assertIn("ended before it became ready", result.text)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.handoffs, 0)

    async def test_a_command_that_cannot_start_is_reported(self):
        tool = RestartTool(["/nonexistent/python"], self.handoff)
        result = await tool.run({"confirm": True})
        self.assertIn("could not start", result.text)
