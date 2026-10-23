# SPDX-License-Identifier: AGPL-3.0-or-later
"""A self-restart tool with readiness handoff for the Discord process."""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from tooling import ToolResult


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
        require_confirmation: bool = True,
        ready_timeout: float = 45.0,
    ) -> None:
        self._command = list(command)
        self._on_handoff = on_handoff
        self._require_confirmation = require_confirmation
        self._ready_timeout = ready_timeout

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self._require_confirmation and arguments.get("confirm") is not True:
            return ToolResult("Confirmation required: repeat this call with confirm=true to restart the bot.")
        if not self._command:
            return ToolResult("Error: no restart command is configured.")

        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        os.set_blocking(read_fd, False)
        env = {**os.environ, "EMILIA_RESTART_READY_FD": str(write_fd)}
        try:
            process = subprocess.Popen(self._command, env=env, close_fds=True, pass_fds=(write_fd,))
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            return ToolResult(f"Error: could not start replacement bot: {exc}")
        finally:
            # The child inherited its own copy; keeping ours open would make a failed child look like a hang.
            try:
                os.close(write_fd)
            except OSError:
                pass

        deadline = time.monotonic() + self._ready_timeout
        ready = b""
        try:
            while time.monotonic() < deadline:
                try:
                    ready = os.read(read_fd, 1)
                except BlockingIOError:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                    continue
                break
            else:
                if process.poll() is None:
                    process.terminate()
                return ToolResult(f"Error: replacement bot did not become ready within {self._ready_timeout:g} seconds; current bot is still running.")
        finally:
            os.close(read_fd)

        if ready != b"1":
            status = process.poll()
            detail = f" (exit code {status})" if status is not None else ""
            return ToolResult(f"Error: replacement bot ended before it became ready{detail}; current bot is still running.")
        # A replacement normally keeps running. Reap the rare one that exits immediately after
        # signalling readiness, without ever waiting for a real bot process here.
        await asyncio.sleep(0.05)
        if process.poll() is not None:
            process.wait()
        await self._on_handoff()
        return ToolResult("Replacement bot is connected and ready. The current bot is shutting down shortly.")
