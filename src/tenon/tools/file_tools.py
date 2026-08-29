"""File tools: read_file / write_file / edit (exact string replacement).

Editing primitive converged from Claude Code / Aider and others: exact
string replacement with a uniqueness requirement — no line-number edits,
no unified diffs (LLMs count lines unreliably).
"""
from __future__ import annotations

from .base import Tool, ToolContext, ToolError, ToolResult, resolve_path

MAX_READ_LINES = 2000
MAX_READ_BYTES = 50_000
MAX_LINE_CHARS = 2000


def _did_you_mean(text: str, old: str) -> str:
    """Locate near-miss lines to help the model fix a failed old_string."""
    stripped = old.strip()
    probe = stripped.splitlines()[0].strip() if stripped else ""
    if len(probe) < 4:
        return " Re-read the file section and copy the text exactly."
    candidates = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip() == probe or probe in line:
            candidates.append(str(i))
        if len(candidates) >= 3:
            break
    if candidates:
        return (
            f" Similar text found at line(s) {', '.join(candidates)} — re-read "
            "those lines and copy the text exactly (whitespace matters)."
        )
    return " Re-read the file section you intend to change and copy the text exactly."


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file with 1-based line numbers. Output is capped at "
        f"{MAX_READ_LINES} lines / ~{MAX_READ_BYTES // 1000}KB; use offset and limit to page "
        "through large files. Always read a file before overwriting or editing "
        "it. Do not use on directories (use glob) or binary files. Returns a "
        "structured error with a hint when the path is wrong."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the working directory"},
            "offset": {"type": "integer", "description": "First line to show, 1-based (default 1)"},
            "limit": {"type": "integer", "description": f"Max lines to show (default {MAX_READ_LINES})"},
        },
        "required": ["path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        if not path.exists():
            raise ToolError(f"file not found: {path}")
        if path.is_dir():
            raise ToolError(f"{path} is a directory — use glob to list files, not read_file")
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ToolError(f"{path} looks like a binary file; read_file only handles text")
        text = raw.decode("utf-8", errors="replace")
        ctx.read_files.add(path)

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult(True, f"(empty file: {path})")
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or MAX_READ_LINES)
        if offset > total:
            raise ToolError(f"offset {offset} is beyond end of file ({total} lines)")

        out_lines: list[str] = []
        bytes_used = 0
        for i in range(offset - 1, min(offset - 1 + limit, total)):
            line = lines[i]
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + "… [line truncated]"
            entry = f"{i + 1}\t{line}"
            if bytes_used + len(entry) > MAX_READ_BYTES:
                break
            out_lines.append(entry)
            bytes_used += len(entry) + 1

        shown_end = offset + len(out_lines) - 1
        body = "\n".join(out_lines)
        truncated = shown_end < total
        if truncated:
            body += (
                f"\n[truncated: showing lines {offset}–{shown_end} of {total}; "
                f"continue with offset={shown_end + 1}]"
            )
        return ToolResult(True, body, truncated=truncated)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write a whole file: create a new file or fully rewrite an existing "
        "one (parent directories are created). To overwrite an existing file "
        "you must have read it first in this session. For targeted changes "
        "prefer edit; use write_file for new files or full rewrites."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the working directory"},
            "content": {"type": "string", "description": "Full file content, written verbatim"},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        if path.is_dir():
            raise ToolError(f"{path} is a directory")
        if path.exists() and path not in ctx.read_files:
            raise ToolError(
                f"refusing to overwrite {path}: read it first with read_file "
                "(read-before-write contract)"
            )
        content = args["content"]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.on_file_write is not None:
                ctx.on_file_write(path)
            path.write_text(content)
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        ctx.read_files.add(path)
        n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return ToolResult(True, f"wrote {len(content.encode('utf-8'))} bytes ({n_lines} lines) to {path}")


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact string in a file with another string (search-and-"
        "replace editing). old_string must match the file content exactly, "
        "including indentation and whitespace, and must occur exactly once "
        "unless replace_all=true. You must have read the file first. On "
        "failure the error says whether the string was not found (with line "
        "hints) or ambiguous — then add more context lines to make it unique. "
        "Prefer this over write_file for targeted changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the working directory"},
            "old_string": {"type": "string", "description": "Exact text to replace; must be unique in the file"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        if not path.exists():
            raise ToolError(f"file not found: {path}; create new files with write_file")
        if path not in ctx.read_files:
            raise ToolError(
                f"read {path} with read_file before editing it "
                "(read-before-write contract)"
            )
        old, new = args["old_string"], args["new_string"]
        if not old:
            raise ToolError("old_string is empty; to write a whole file use write_file")
        if old == new:
            raise ToolError("old_string and new_string are identical — nothing to change")
        try:
            text = path.read_text()
        except UnicodeDecodeError as exc:
            raise ToolError(f"{path} is not valid UTF-8 text") from exc

        count = text.count(old)
        replace_all = bool(args.get("replace_all", False))
        if count == 0:
            raise ToolError(f"old_string not found in {path}.{_did_you_mean(text, old)}")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string matches {count} locations in {path}; add more "
                "surrounding context to make it unique, or set replace_all=true"
            )
        if ctx.on_file_write is not None:
            ctx.on_file_write(path)
        path.write_text(text.replace(old, new))
        replaced = count if replace_all else 1
        return ToolResult(True, f"replaced {replaced} occurrence(s) in {path}")
