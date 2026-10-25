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

"""The restart_bot tool: start a replacement process and hand over once it is connected.

The replacement gets the write end of a pipe (its number in EMILIA_RESTART_READY_FD) and writes b"1"
to it from on_ready. Only then does the current bot retire; if the replacement crashes or never
connects, it is stopped and the current bot carries on. POSIX only (it relies on pass_fds).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from tooling import ToolResult

READY_FD_VARIABLE = "EMILIA_RESTART_READY_FD"


class RestartTool:
    name = "restart_bot"
    schema = {
        "name": name,
        "description": "Restart the bot process so code changes take effect. The current bot stays running until its replacement connects successfully.",
        "parameters": {"type": "object", "properties": {"confirm": {"type": "boolean", "description": "Set true after confirming the restart."}}},
    }

    def __init__(
        self,
        command: Sequence[str],
        on_handoff: Callable[[], Awaitable[None]],
        *,
        environment: Mapping[str, str] | None = None,
        require_confirmation: bool = True,
        ready_timeout: float = 45.0,
    ) -> None:
        self._command = list(command)
        self._on_handoff = on_handoff
        # The environment the replacement starts with. Pass the one from before .env was loaded, or
        # the replacement inherits the old .env values and load_dotenv (which never overrides) keeps them.
        self._environment = dict(environment) if environment is not None else None
        self._require_confirmation = require_confirmation
        self._ready_timeout = ready_timeout
        self._lock = asyncio.Lock()  # one replacement at a time
        self._handed_off = False
        self._replacement: subprocess.Popen | None = None  # the new bot; it outlives this process

    def describe(self, arguments: dict[str, Any]) -> str:
        return "🔄 Restarting the bot"

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self._require_confirmation and arguments.get("confirm") is not True:
            return ToolResult("Confirmation required: repeat this call with confirm=true to restart the bot.")
        if not self._command:
            return ToolResult("Error: no restart command is configured.")
        async with self._lock:
            if self._handed_off:
                return ToolResult("The bot has already been restarted; the replacement is running. Don't restart it again.")
            return await self._restart()

    async def _restart(self) -> ToolResult:
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(read_fd, False)
            base = os.environ if self._environment is None else self._environment
            env = {**base, READY_FD_VARIABLE: str(write_fd)}
            process = subprocess.Popen(self._command, env=env, close_fds=True, pass_fds=(write_fd,))
        except (OSError, ValueError) as exc:  # ValueError: pass_fds is not supported on this platform
            os.close(read_fd)
            return ToolResult(f"Error: could not start replacement bot: {exc}")
        finally:
            # The child has its own copy; keeping ours open would make a failed child look like a hang.
            os.close(write_fd)

        try:
            ready = await self._wait_for_ready(read_fd, process)
        except asyncio.CancelledError:
            # The reply was stopped (!giveup): this bot carries on, so the replacement must not come up beside it.
            await self._stop(process)
            raise
        finally:
            os.close(read_fd)

        if ready is None:
            await self._stop(process)
            return ToolResult(
                f"Error: replacement bot did not become ready within {self._ready_timeout:g} seconds; current bot is still running."
            )
        if ready != b"1":
            status = await self._stop(process)
            detail = f" (exit code {status})" if status is not None else ""
            return ToolResult(f"Error: replacement bot ended before it became ready{detail}; current bot is still running.")

        self._handed_off = True
        self._replacement = process
        await self._on_handoff()
        return ToolResult(
            "Replacement bot is connected and ready. This bot finishes the current reply and then shuts down; "
            "don't restart again."
        )

    async def _wait_for_ready(self, read_fd: int, process: subprocess.Popen) -> bytes | None:
        """What the replacement wrote: b"1" when ready, b"" if it closed the pipe (usually by exiting), None on timeout."""
        deadline = time.monotonic() + self._ready_timeout
        while time.monotonic() < deadline:
            try:
                return os.read(read_fd, 1)
            except BlockingIOError:
                await asyncio.sleep(0.05)
        return None

    @staticmethod
    async def _stop(process: subprocess.Popen) -> int | None:
        """Make sure a failed replacement is gone (and reaped); returns its exit code."""
        if process.poll() is None:
            process.terminate()
        try:
            return await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.kill()
            return await asyncio.to_thread(process.wait)
