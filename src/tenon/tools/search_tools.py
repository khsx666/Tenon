"""Search tools: grep (content search) and glob (file-name matching).

grep prefers the system ripgrep and falls back to a hand-written regex
walker, so the tool works on machines without rg installed.
"""
from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

from .base import Tool, ToolContext, ToolError, ToolResult, resolve_path

MAX_RESULTS = 100
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression (ripgrep when "
        "available, built-in fallback otherwise). Returns path:line:content "
        f"lines, capped at {MAX_RESULTS} matches — if capped, refine the "
        "pattern or narrow the path. Prefer this over running grep via bash. "
        "To match files by name, use glob instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "File or directory to search (default: working directory)"},
            "glob": {"type": "string", "description": "Only search files matching this pattern, e.g. '*.py'"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        try:
            regex = re.compile(pattern, re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as exc:
            raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc
        base = resolve_path(ctx, args.get("path", "."))
        if not base.exists():
            raise ToolError(f"path not found: {base}")
        include = args.get("glob")

        rg = shutil.which("rg")
        if rg is not None:
            return self._with_rg(rg, pattern, base, include, bool(args.get("ignore_case")))
        return self._with_fallback(regex, base, include)

    def _with_rg(self, rg: str, pattern: str, base: Path, include: str | None,
                 ignore_case: bool) -> ToolResult:
        cmd = [rg, "--line-number", "--no-heading", "--color=never", "-e", pattern]
        if ignore_case:
            cmd.append("--ignore-case")
        if include:
            cmd += ["--glob", include]
        cmd.append(str(base))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 1:
            return ToolResult(True, "no matches found")
        if proc.returncode not in (0, 1):
            raise ToolError(f"ripgrep failed: {proc.stderr.strip()[:300]}")
        return _cap_results(proc.stdout.splitlines())

    def _with_fallback(self, regex: re.Pattern, base: Path, include: str | None) -> ToolResult:
        lines: list[str] = []
        files = [base] if base.is_file() else _walk_files(base)
        for file in files:
            if include and not fnmatch.fnmatch(file.name, include):
                continue
            try:
                text = file.read_text(errors="replace")
            except (OSError, UnicodeError):
                continue
            if "\x00" in text[:1024]:
                continue  # skip binaries
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    lines.append(f"{file}:{lineno}:{line}")
        if not lines:
            return ToolResult(True, "no matches found")
        return _cap_results(lines)


def _walk_files(base: Path):
    for path in sorted(base.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _cap_results(lines: list[str]) -> ToolResult:
    truncated = len(lines) > MAX_RESULTS
    shown = lines[:MAX_RESULTS]
    body = "\n".join(shown)
    if truncated:
        body += (
            f"\n[truncated: showing {MAX_RESULTS} of {len(lines)} matches; "
            "refine the pattern or narrow the path]"
        )
    return ToolResult(True, body, truncated=truncated)


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by name pattern. A bare pattern like '*.py' matches at "
        "any depth below the search path; use '**' explicitly for recursive "
        "directory patterns. Results are sorted by most recently modified, "
        f"capped at {MAX_RESULTS}. To search file contents, use grep instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "File-name pattern, e.g. '*.py' or 'src/**/*.txt'"},
            "path": {"type": "string", "description": "Directory to search (default: working directory)"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        base = resolve_path(ctx, args.get("path", "."))
        if not base.exists():
            raise ToolError(f"path not found: {base}")
        if not base.is_dir():
            raise ToolError(f"{base} is not a directory")
        if "/" not in pattern:
            pattern = f"**/{pattern}"
        matches = [
            p for p in base.glob(pattern)
            if p.is_file() and ".git" not in p.relative_to(base).parts
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return ToolResult(True, "no files matched")
        truncated = len(matches) > MAX_RESULTS
        body = "\n".join(str(p) for p in matches[:MAX_RESULTS])
        if truncated:
            body += (
                f"\n[truncated: showing {MAX_RESULTS} of {len(matches)} files; "
                "use a more specific pattern]"
            )
        return ToolResult(True, body, truncated=truncated)
