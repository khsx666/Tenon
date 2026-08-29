"""bash tool: stateless subprocess + persistent cwd + timeout/truncation/exit code.

Design notes:
- A fresh `bash -c` per call (no PTY/tmux session) — complexity is not worth
  it; the working directory is persisted across calls instead, which covers
  the main need.
- Non-zero exit codes are normal feedback (the input of the test-driven
  repair loop), not exceptions.
- Known interactive commands are rejected with an actionable hint.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

from .base import Tool, ToolContext, ToolError, ToolResult, resolve_path

MAX_OUTPUT_CHARS = 30_000   # ~30KB, head+tail middle truncation
MAX_TIMEOUT_S = 600.0
_PWD_SENTINEL = "__TENON_PWD__"

_ALWAYS_INTERACTIVE = {
    "vim", "nvim", "nano", "emacs", "less", "more", "man", "top", "htop",
    "watch", "ssh", "telnet", "ftp", "sftp",
}
_REPL_WHEN_BARE = {"python", "python3", "node", "irb", "ghci"}


def _command_segments(command: str) -> list[list[str]]:
    """Tokenize each pipeline/sequence segment of a shell command line."""
    segments = []
    for seg in re.split(r"(?:\|\||&&|\||;|\n)", command):
        try:
            tokens = shlex.split(seg)
        except ValueError:
            continue
        if tokens:
            segments.append(tokens)
    return segments


def _find_interactive(command: str) -> str | None:
    """Return the offending program name if the command would block on a TTY.

    Only a truly bare interpreter (`python`, `node`, no arguments at all)
    is treated as interactive — `python3 --version` and friends exit
    immediately and must pass through.
    """
    for tokens in _command_segments(command):
        prog = Path(tokens[0]).name
        if prog in _ALWAYS_INTERACTIVE:
            return prog
        if prog in _REPL_WHEN_BARE and len(tokens) == 1:
            return prog
    return None


def truncate_middle(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """Keep the head and tail; cut the middle (newest output usually matters most)."""
    if len(text) <= limit:
        return text, False
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n[... {omitted} chars truncated ...]\n{text[-half:]}", True


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command in the session working directory (the directory "
        "persists across calls, so cd works). Returns stdout, stderr and the "
        "exit code — a non-zero exit code is normal feedback: read the output "
        f"and fix the cause. Default timeout 120s, max {MAX_TIMEOUT_S:.0f}s. Output is "
        "truncated in the middle past ~30KB. Interactive commands (vim, ssh, "
        "a bare python REPL, ...) are rejected — use a non-interactive form "
        "such as flags, pipes, or a script file. Do not use it to read/edit "
        "files when read_file/edit/grep/glob can do the job."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command line to run"},
            "timeout": {"type": "number", "description": f"Timeout in seconds (default 120, max {MAX_TIMEOUT_S:.0f})"},
        },
        "required": ["command"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = args["command"]
        interactive = _find_interactive(command)
        if interactive:
            raise ToolError(
                f"{interactive!r} is interactive and would block the session; "
                "use a non-interactive form (flags, piped input, or write a "
                "script file first and run that)"
            )
        try:
            timeout = min(float(args.get("timeout") or ctx.default_timeout), MAX_TIMEOUT_S)
        except (TypeError, ValueError):
            raise ToolError(f"timeout must be a number, got {args.get('timeout')!r}") from None

        wrapped = (
            f"{command}\n__rc=$?\n"
            f"printf '\\n{_PWD_SENTINEL}%s\\n' \"$PWD\"\n"
            "exit $__rc"
        )
        try:
            proc = subprocess.run(
                ["bash", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=ctx.cwd,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            partial, _ = truncate_middle(partial)
            return ToolResult(
                False,
                f"error=timeout: command exceeded {timeout:.0f}s and was killed. "
                f"Partial output:\n{partial or '(none)'}",
                truncated=True,
            )

        stdout = proc.stdout or ""
        lines = stdout.splitlines()
        if lines and lines[-1].startswith(_PWD_SENTINEL):
            new_cwd = Path(lines[-1][len(_PWD_SENTINEL):])
            lines = lines[:-1]
            if lines and lines[-1] == "":
                lines = lines[:-1]
            stdout = "\n".join(lines)
            if new_cwd.is_dir():
                ctx.cwd = new_cwd

        parts: list[str] = []
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")
        out, err = stdout.rstrip("\n"), (proc.stderr or "").rstrip("\n")
        if out:
            parts.append(out)
        if err:
            parts.append(("stderr:\n" + err) if out else err)
        body_full = "\n".join(parts) if parts else "(no output)"
        body, truncated = truncate_middle(body_full)
        if truncated and ctx.overflow_dir is not None:
            spill = _spill_overflow(ctx.overflow_dir, body_full)
            if spill is not None:
                body += f"\n[full output saved to {spill} — grep it or read it in pages]"
        return ToolResult(proc.returncode == 0, body, truncated=truncated)


def _spill_overflow(overflow_dir: Path, content: str) -> Path | None:
    """L1: persist an untruncated tool output so the model can page/grep it."""
    try:
        overflow_dir.mkdir(parents=True, exist_ok=True)
        spill = overflow_dir / f"bash-{uuid4().hex[:8]}.log"
        spill.write_text(content)
        return spill
    except OSError:
        return None
