# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
import unittest

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
