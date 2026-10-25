# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from computer import WorkspaceTools, computer_tools


class WorkspaceToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = WorkspaceTools(self.root, command_timeout=1, require_confirmation=False)

    def tearDown(self):
        self.temp.cleanup()

    async def test_write_read_edit_and_delete_file(self):
        result = await self.tools.write_file({"path": "notes/hello.txt", "content": "hello world"})
        self.assertIn("Wrote", result.text)
        self.assertEqual((await self.tools.read_file({"path": "notes/hello.txt"})).text, "hello world")
        self.assertIn("Edited", (await self.tools.edit_file({"path": "notes/hello.txt", "old": "world", "new": "Emilia"})).text)
        self.assertEqual((await self.tools.read_file({"path": "notes/hello.txt"})).text, "hello Emilia")
        self.assertIn("Deleted", (await self.tools.delete_path({"path": "notes/hello.txt"})).text)

    async def test_edit_requires_one_exact_match(self):
        await self.tools.write_file({"path": "a.txt", "content": "x x"})
        result = await self.tools.edit_file({"path": "a.txt", "old": "x", "new": "y"})
        self.assertIn("found it 2 times", result.text)
        self.assertEqual((await self.tools.read_file({"path": "a.txt"})).text, "x x")

    async def test_paths_cannot_escape_workspace(self):
        result = await self.tools.read_file({"path": "../secret.txt"})
        self.assertIn("must stay inside", result.text)
        result = await self.tools.write_file({"path": str(Path(self.temp.name).parent / "nope.txt"), "content": "no"})
        self.assertIn("must stay inside", result.text)

    async def test_directory_deletion_needs_explicit_recursive_flag(self):
        await self.tools.write_file({"path": "folder/a.txt", "content": "a"})
        result = await self.tools.delete_path({"path": "folder"})
        self.assertIn("recursive=true", result.text)
        self.assertTrue((self.root / "folder").exists())
        self.assertIn("Deleted", (await self.tools.delete_path({"path": "folder", "recursive": True})).text)

    async def test_command_runs_in_workspace_and_captures_result(self):
        result = await self.tools.run_command({"command": "pwd; printf hello"})
        self.assertIn("Exit code: 0", result.text)
        self.assertIn("/workspace", result.text)
        self.assertIn("hello", result.text)

    async def test_a_stopped_command_is_killed(self):
        tools = WorkspaceTools(self.root, command_timeout=30, allow_host_access=True, require_confirmation=False)
        running = asyncio.create_task(tools.run_command({"command": "echo $$ > pid; exec sleep 30"}))
        pid_file = self.root / "pid"
        while not pid_file.exists() or not pid_file.read_text().strip():
            await asyncio.sleep(0.05)
        pid = int(pid_file.read_text())
        running.cancel()  # !giveup
        with self.assertRaises(asyncio.CancelledError):
            await running
        for _ in range(40):  # killed, then reaped by asyncio's child watcher
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("the command is still running")

    async def test_default_requires_explicit_confirmation_for_mutations(self):
        tools = WorkspaceTools(self.root)
        result = await tools.write_file({"path": "a.txt", "content": "a"})
        self.assertIn("Confirmation required", result.text)
        self.assertFalse((self.root / "a.txt").exists())

    def test_tool_bundle_has_all_operations(self):
        self.assertEqual(
            [tool.name for tool in computer_tools(self.root)],
            ["list_files", "read_file", "write_file", "edit_file", "delete_path", "run_command"],
        )
