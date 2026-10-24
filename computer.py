# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tools for working on an explicitly configured local workspace.

They intentionally do not expose the whole host filesystem.  A Discord message is untrusted input,
even when a model is deciding which function to call, so all paths (including command working
directories) stay below ``workspace``.  Enable the bundle explicitly with COMPUTER_TOOLS=true.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from tooling import ToolResult, code_span

MAX_FILE_BYTES = 256_000
MAX_OUTPUT_CHARS = 12_000


class WorkspaceTools:
    """File and terminal tools rooted at one directory."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        command_timeout: float = 30.0,
        allow_host_access: bool = False,
        require_confirmation: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.command_timeout = command_timeout
        self.allow_host_access = allow_host_access
        self.require_confirmation = require_confirmation

    def _path(self, value: Any, *, allow_root: bool = False) -> Path:
        raw = str(value or ".")
        candidate = (self.workspace / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if not self.allow_host_access:
            try:
                candidate.relative_to(self.workspace)
            except ValueError:
                raise ValueError("path must stay inside the configured workspace") from None
        if candidate == self.workspace and not allow_root:
            raise ValueError("the workspace root itself is not a valid target")
        return candidate

    @staticmethod
    def _display(path: Path, root: Path) -> str:
        try:
            return "." if path == root else path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    def _confirmed(self, arguments: dict[str, Any]) -> ToolResult | None:
        if self.require_confirmation and arguments.get("confirm") is not True:
            return ToolResult("Confirmation required: repeat this call with confirm=true to make changes or run a command.")
        return None

    async def list_files(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = self._path(arguments.get("path", "."), allow_root=True)
            if not path.is_dir():
                return ToolResult(f"Error: {self._display(path, self.workspace)} is not a directory.")
            limit = min(max(int(arguments.get("max_entries", 100)), 1), 500)
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:limit]
            lines = [self._display(item, self.workspace) + ("/" if item.is_dir() else "") for item in entries]
            suffix = "\n(Results were limited.)" if len(list(path.iterdir())) > limit else ""
            return ToolResult("\n".join(lines) + suffix or "(empty directory)")
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")

    async def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = self._path(arguments.get("path"))
            if not path.is_file():
                return ToolResult(f"Error: {self._display(path, self.workspace)} is not a file.")
            if path.stat().st_size > MAX_FILE_BYTES:
                return ToolResult(f"Error: file is larger than {MAX_FILE_BYTES} bytes.")
            return ToolResult(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")

    async def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        if denied := self._confirmed(arguments):
            return denied
        try:
            path = self._path(arguments.get("path"))
            content = arguments.get("content")
            if not isinstance(content, str):
                return ToolResult("Error: 'content' must be a string.")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                return ToolResult(f"Error: content is larger than {MAX_FILE_BYTES} bytes.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(f"Wrote {len(content.encode('utf-8'))} bytes to {self._display(path, self.workspace)}.")
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")

    async def edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        if denied := self._confirmed(arguments):
            return denied
        try:
            path = self._path(arguments.get("path"))
            old, new = arguments.get("old"), arguments.get("new")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                return ToolResult("Error: 'old' must be a non-empty string and 'new' must be a string.")
            content = path.read_text(encoding="utf-8", errors="replace")
            count = content.count(old)
            if count != 1:
                return ToolResult(f"Error: expected 'old' exactly once, found it {count} times; file was not changed.")
            updated = content.replace(old, new, 1)
            if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
                return ToolResult(f"Error: edited file would exceed {MAX_FILE_BYTES} bytes.")
            path.write_text(updated, encoding="utf-8")
            return ToolResult(f"Edited {self._display(path, self.workspace)}.")
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")

    async def delete_path(self, arguments: dict[str, Any]) -> ToolResult:
        if denied := self._confirmed(arguments):
            return denied
        try:
            path = self._path(arguments.get("path"))
            recursive = arguments.get("recursive", False) is True
            if path.is_dir():
                if not recursive:
                    return ToolResult("Error: target is a directory; set recursive=true to delete it.")
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
            else:
                return ToolResult(f"Error: {self._display(path, self.workspace)} does not exist.")
            return ToolResult(f"Deleted {self._display(path, self.workspace)}.")
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")

    async def run_command(self, arguments: dict[str, Any]) -> ToolResult:
        if denied := self._confirmed(arguments):
            return denied
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult("Error: 'command' must be a non-empty string.")
        try:
            cwd = self._path(arguments.get("cwd", "."), allow_root=True)
            if not cwd.is_dir():
                return ToolResult("Error: cwd must be a directory inside the configured workspace.")
            env = {**os.environ, "TERM": "dumb"}
            if self.allow_host_access:
                # This is intentionally the unrestricted mode selected by COMPUTER_ALLOW_HOST_ACCESS.
                process = await asyncio.create_subprocess_shell(
                    command, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
                )
            else:
                sandbox = shutil.which("bwrap")
                if sandbox is None:
                    return ToolResult("Error: workspace command sandbox (bwrap) is not installed; commands are disabled.")
                internal_cwd = "/workspace" if cwd == self.workspace else "/workspace/" + cwd.relative_to(self.workspace).as_posix()
                # Bubblewrap creates a fresh mount namespace. The command can read standard system
                # binaries/libraries and can modify only the workspace mounted at /workspace.
                mounts: list[str] = []
                for system_dir in ("/usr", "/bin", "/lib", "/lib64"):
                    if Path(system_dir).exists():
                        mounts.extend(("--ro-bind", system_dir, system_dir))
                sandbox_env = {"PATH": "/usr/bin:/bin", "TERM": "dumb", "HOME": "/workspace", "LANG": os.environ.get("LANG", "C.UTF-8")}
                process = await asyncio.create_subprocess_exec(
                    sandbox,
                    "--die-with-parent",
                    *mounts,
                    "--proc", "/proc",
                    "--dev", "/dev",
                    "--tmpfs", "/tmp",
                    "--bind", str(self.workspace), "/workspace",
                    "--chdir", internal_cwd,
                    "/bin/sh", "-lc", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=sandbox_env,
                )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), self.command_timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(f"Error: command timed out after {self.command_timeout:g} seconds.")
            text = output.decode("utf-8", errors="replace")
            if len(text) > MAX_OUTPUT_CHARS:
                text = text[:MAX_OUTPUT_CHARS] + "\n(Output truncated.)"
            return ToolResult(f"Exit code: {process.returncode}\n{text}".rstrip())
        except (OSError, ValueError) as exc:
            return ToolResult(f"Error: {exc}")


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] = []) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}


# What the chat sees when the model uses each tool: the text, the argument it names, and its default.
_LABELS = {
    "list_files": ("📂 Looking through", "path", "."),
    "read_file": ("📄 Reading", "path", ""),
    "write_file": ("📝 Writing", "path", ""),
    "edit_file": ("✏️ Editing", "path", ""),
    "delete_path": ("🗑️ Deleting", "path", ""),
    "run_command": ("💻 Running", "command", ""),
}


class _WorkspaceTool:
    def __init__(self, tools: WorkspaceTools, name: str, schema: dict[str, Any]) -> None:
        self._tools, self.name, self.schema = tools, name, schema

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await getattr(self._tools, self.name)(arguments)

    def describe(self, arguments: dict[str, Any]) -> str:
        text, key, default = _LABELS[self.name]
        return f"{text} {code_span(arguments.get(key) or default)}".rstrip()


def computer_tools(
    workspace: str | Path,
    *,
    command_timeout: float = 30.0,
    allow_host_access: bool = False,
    require_confirmation: bool = True,
) -> list[_WorkspaceTool]:
    """Return the complete, separately named tools offered to the language model."""
    tools = WorkspaceTools(
        workspace,
        command_timeout=command_timeout,
        allow_host_access=allow_host_access,
        require_confirmation=require_confirmation,
    )
    path = {"path": {"type": "string", "description": "A path relative to the workspace."}}
    return [
        _WorkspaceTool(tools, "list_files", _schema("list_files", "List files and directories in the workspace.", {**path, "max_entries": {"type": "integer", "minimum": 1, "maximum": 500}})),
        _WorkspaceTool(tools, "read_file", _schema("read_file", "Read a UTF-8 text file from the workspace.", path, ["path"])),
        _WorkspaceTool(tools, "write_file", _schema("write_file", "Create or replace a UTF-8 text file.", {**path, "content": {"type": "string"}, "confirm": {"type": "boolean", "description": "Set true after confirming a mutating action."}}, ["path", "content"])),
        _WorkspaceTool(tools, "edit_file", _schema("edit_file", "Replace one unique exact string in a text file.", {**path, "old": {"type": "string"}, "new": {"type": "string"}, "confirm": {"type": "boolean"}}, ["path", "old", "new"])),
        _WorkspaceTool(tools, "delete_path", _schema("delete_path", "Delete a file, or a directory only when recursive is true.", {**path, "recursive": {"type": "boolean"}, "confirm": {"type": "boolean"}}, ["path"])),
        _WorkspaceTool(tools, "run_command", _schema("run_command", "Run a shell command. It may create, edit or delete files.", {"command": {"type": "string"}, "cwd": {"type": "string", "description": "Optional working directory."}, "confirm": {"type": "boolean"}}, ["command"])),
    ]
