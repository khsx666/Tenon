"""todo_write: externalized task list.

Recitation-style attention anchor: the model rewrites its plan into the
context as a checklist, keeps exactly one item in_progress, and marks
items done as it goes. The rendered list is also injected into the system
prompt's dynamic section.
"""
from __future__ import annotations

from .base import Tool, ToolContext, ToolError, ToolResult

_STATUSES = ("pending", "in_progress", "done")
_MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}


def render_todos(todos: list[dict]) -> str:
    return "\n".join(f"{_MARKS[t['status']]} {t['content']}" for t in todos)


class TodoWriteTool(Tool):
    name = "todo_write"
    description = (
        "Overwrite the shared task list for this session. Use it to plan "
        "multi-step tasks and track progress: exactly one item may be "
        "in_progress at a time; mark items done as soon as they are finished. "
        "The list stays visible in context, so keep it current."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full task list, replacing the previous one",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Task description"},
                        "status": {"type": "string", "enum": list(_STATUSES)},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        todos = args["todos"]
        if not todos:
            raise ToolError("todos list is empty")
        cleaned: list[dict] = []
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                raise ToolError(f"todos[{i}] must be an object with content and status")
            content = str(item.get("content", "")).strip()
            status = item.get("status")
            if not content:
                raise ToolError(f"todos[{i}].content is empty")
            if status not in _STATUSES:
                raise ToolError(
                    f"todos[{i}].status must be one of {list(_STATUSES)}, got {status!r}"
                )
            cleaned.append({"content": content, "status": status})
        n_in_progress = sum(1 for t in cleaned if t["status"] == "in_progress")
        if n_in_progress > 1:
            raise ToolError(
                f"only one todo may be in_progress, got {n_in_progress}; "
                "mark the others pending"
            )
        ctx.todos = cleaned
        return ToolResult(True, "task list updated:\n" + render_todos(cleaned))
