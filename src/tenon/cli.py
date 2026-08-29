"""Command-line interface.

M3 delivers the non-interactive `-p` one-shot mode; the full interactive
REPL (prompt_toolkit, interrupts, slash commands) arrives in M6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .agent import Agent
from .config import ConfigError, load_config
from .llm import LLMClient
from .tools import ToolContext

MAX_ARG_PREVIEW = 120
MAX_RESULT_PREVIEW = 240


def _preview(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def make_printer(console: Console):
    def on_event(type_: str, **data) -> None:
        if type_ == "text":
            console.print(Markdown(data["text"]))
        elif type_ == "tool_call":
            console.print(
                f"[cyan]⏺ {data['name']}[/cyan] "
                f"[dim]{_preview(data['args'], MAX_ARG_PREVIEW)}[/dim]"
            )
        elif type_ == "tool_result":
            style = "dim" if data["ok"] else "red"
            console.print(f"  [{style}]⎿ {_preview(data['output'], MAX_RESULT_PREVIEW)}[/{style}]")
        elif type_ == "warning":
            console.print(f"[yellow]⚠ {data['text']}[/yellow]")
        elif type_ == "exit" and data["reason"] != "completed":
            console.print(f"[yellow]{data['text']}[/yellow]")

    return on_event


def run_once(prompt: str, console: Console) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2
    agent = Agent(
        config,
        LLMClient(config),
        ctx=ToolContext(cwd=Path.cwd(), default_timeout=config.timeout),
        on_event=make_printer(console),
    )
    try:
        agent.run(prompt)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    usage = agent.llm.usage
    console.print(
        f"\n[dim]tokens: {usage.prompt_tokens} prompt / {usage.completion_tokens} "
        f"completion ({usage.calls} API calls)[/dim]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tenon", description="A minimal, from-scratch coding agent."
    )
    parser.add_argument("-p", "--prompt", help="run a single task non-interactively")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    console = Console()
    if args.version:
        from . import __version__

        console.print(__version__)
        return 0
    if args.prompt:
        return run_once(args.prompt, console)
    console.print('Interactive REPL arrives in a later milestone; use -p "task" for now.')
    return 0
