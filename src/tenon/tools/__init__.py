"""Built-in tool set, registry, and the dispatch entry point."""
from .base import Tool, ToolContext, ToolError, ToolResult, dispatch
from .bash_tool import BashTool
from .file_tools import EditTool, ReadFileTool, WriteFileTool
from .search_tools import GlobTool, GrepTool
from .todo_tool import TodoWriteTool, render_todos

ALL_TOOLS: list[Tool] = [
    ReadFileTool(),
    WriteFileTool(),
    EditTool(),
    BashTool(),
    GrepTool(),
    GlobTool(),
    TodoWriteTool(),
]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def all_schemas() -> list[dict]:
    """Tool schemas in chat-completions format, in stable registry order."""
    return [t.schema() for t in ALL_TOOLS]


__all__ = [
    "ALL_TOOLS",
    "TOOL_REGISTRY",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "all_schemas",
    "dispatch",
    "render_todos",
]
