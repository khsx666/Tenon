"""Command-line interface.

M3 delivered the non-interactive `-p` one-shot mode; the full interactive
REPL (prompt_toolkit, interrupts, slash commands) arrives in M6.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .agent import Agent
from .checkpoint import Checkpointer
from .config import ConfigError, load_config
from .llm import LLMClient
from .safety import PermissionMode, SafetyLayer
from .session import SessionLogger
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
        elif type_ == "permission":
            console.print(f"[red]  ✗ {data['name']} denied: {data['reason']}[/red]")
        elif type_ == "warning":
            console.print(f"[yellow]⚠ {data['text']}[/yellow]")
        elif type_ == "masked":
            console.print(f"[dim]  … masked {data['count']} old tool result(s) to save context[/dim]")
        elif type_ == "compact":
            console.print(
                f"[dim]  … context compressed: ~{data['before']} → ~{data['after']} "
                "tokens (est.)[/dim]"
            )
        elif type_ == "exit" and data["reason"] != "completed":
            console.print(f"[yellow]{data['text']}[/yellow]")

    return on_event


def build_agent(
    console: Console,
    mode_text: str,
    resume: bool,
    confirm_fn=None,
) -> tuple[Agent, SessionLogger, list[dict] | None]:
    """Wire config → ctx/safety/checkpoint/session → Agent for one workspace."""
    config = load_config()
    workspace = Path.cwd()
    checkpointer = Checkpointer(workspace)
    ctx = ToolContext(
        cwd=workspace,
        default_timeout=config.timeout,
        overflow_dir=Path(tempfile.gettempdir()) / "tenon-overflow",
        on_file_write=checkpointer.snapshot,
    )
    safety = SafetyLayer(PermissionMode.parse(mode_text), workspace, confirm_fn=confirm_fn)
    messages: list[dict] | None = None
    if resume:
        latest = SessionLogger.latest(workspace)
        if latest is None:
            raise ConfigError(f"no session log found under {workspace}/.sessions")
        session = SessionLogger(workspace, resume_from=latest)
        messages = SessionLogger.replay(latest)
        console.print(f"[dim]resumed {latest.name}: {len(messages)} messages[/dim]")
    else:
        session = SessionLogger(workspace)
    agent = Agent(
        config,
        LLMClient(config),
        ctx=ctx,
        on_event=make_printer(console),
        safety=safety,
        session=session,
    )
    return agent, session, messages


def run_once(prompt: str, console: Console, mode_text: str, resume: bool) -> int:
    try:
        agent, session, messages = build_agent(console, mode_text, resume)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2
    try:
        agent.run(prompt, messages=messages)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    usage = agent.llm.usage
    console.print(
        f"\n[dim]tokens: {usage.prompt_tokens} prompt / {usage.completion_tokens} "
        f"completion ({usage.calls} API calls) · session log: {session.path}[/dim]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tenon", description="A minimal, from-scratch coding agent."
    )
    parser.add_argument("-p", "--prompt", help="run a single task non-interactively")
    parser.add_argument("-c", "--continue", dest="resume", action="store_true",
                        help="resume the most recent session in this directory")
    parser.add_argument("--mode", choices=[m.value for m in PermissionMode], default=None,
                        help="permission mode (default: ask in REPL, auto with -p)")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    console = Console()
    if args.version:
        from . import __version__

        console.print(__version__)
        return 0
    if args.resume and not args.prompt:
        console.print("[red]-c requires -p for now (interactive resume arrives with the REPL).[/red]")
        return 2
    if args.prompt:
        return run_once(args.prompt, console, args.mode or "auto", args.resume)
    console.print('Interactive REPL arrives in a later milestone; use -p "task" for now.')
    return 0
