"""Command-line interface: interactive REPL + non-interactive -p mode.

Input via prompt_toolkit (history), output via rich. Interrupt layers:
Ctrl+C during a turn aborts the turn but keeps the session; Ctrl+C twice
at the prompt (or Ctrl+D) exits. Slash commands: /help /mode /undo /cost
/compact /quit.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from .agent import Agent
from .checkpoint import Checkpointer
from .config import ConfigError, load_config
from .context import COMPRESS_KEEP_TAIL
from .llm import LLMClient
from .safety import PermissionMode, SafetyLayer
from .session import SESSIONS_DIR, SessionLogger
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


def make_confirm(console: Console):
    """Interactive confirmation card: y / N / a(lways this session)."""

    def confirm(prompt: str) -> str:
        console.print(f"[yellow]{prompt}[/yellow]")
        answer = Prompt.ask("[bold]allow? y/N/a[/bold]", default="n", console=console)
        answer = answer.strip().lower()
        return answer if answer in ("y", "a") else "n"

    return confirm


def build_agent(
    console: Console,
    mode_text: str,
    resume: bool,
    confirm_fn=None,
) -> tuple[Agent, SessionLogger, Checkpointer, list[dict] | None]:
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
            raise ConfigError(f"no session log found under {workspace}/{SESSIONS_DIR}")
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
    return agent, session, checkpointer, messages


def run_once(prompt: str, console: Console, mode_text: str, resume: bool) -> int:
    try:
        agent, session, _checkpointer, messages = build_agent(console, mode_text, resume)
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


_HELP = """\
Commands:
  /help            show this help
  /mode [MODE]     show or switch permission mode (read-only | ask | auto-edit | auto)
  /undo            restore files changed by this session's file edits
                   (bash side effects cannot be rolled back)
  /cost            show token usage so far
  /compact         compress older context now
  /quit            exit
Anything else is sent to the agent as a task.
"""


def handle_command(
    text: str,
    agent: Agent,
    checkpointer: Checkpointer,
    console: Console,
) -> bool:
    """Handle a slash command; returns True when the REPL should exit."""
    cmd, _, arg = text.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    if cmd == "/quit":
        return True
    if cmd == "/help":
        console.print(_HELP)
    elif cmd == "/mode":
        if not arg:
            console.print(f"permission mode: {agent.safety.mode.value}")
        else:
            try:
                agent.safety.mode = PermissionMode.parse(arg)
                console.print(f"permission mode → {arg}")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
    elif cmd == "/undo":
        actions = checkpointer.undo()
        console.print("\n".join(actions) if actions else "nothing to undo")
        console.print("[dim]note: bash side effects cannot be rolled back[/dim]")
    elif cmd == "/cost":
        u = agent.llm.usage
        console.print(
            f"tokens: {u.prompt_tokens} prompt / {u.completion_tokens} completion "
            f"({u.calls} API calls)"
        )
    elif cmd == "/compact":
        try:
            agent._compress(keep_tail=COMPRESS_KEEP_TAIL)
        except ValueError:
            console.print("nothing meaningful to compress yet")
    else:
        console.print(f"[red]unknown command {cmd!r}; /help for commands[/red]")
    return False


def run_repl(console: Console, mode_text: str, resume: bool) -> int:
    try:
        agent, session, checkpointer, messages = build_agent(
            console, mode_text, resume, confirm_fn=make_confirm(console)
        )
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2
    history_dir = Path.cwd() / SESSIONS_DIR
    history_dir.mkdir(exist_ok=True)
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_dir / "repl-history.txt"))
    )
    console.print(
        f"[bold]tenon[/bold] interactive · mode={agent.safety.mode.value} · "
        "/help for commands · Ctrl+C interrupts a turn · Ctrl+D exits"
    )
    last_interrupt = 0.0
    while True:
        try:
            text = prompt_session.prompt("tenon> ")
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_interrupt < 1.5:
                break
            last_interrupt = now
            console.print("[dim]press Ctrl+C again to exit (Ctrl+D also works)[/dim]")
            continue
        except EOFError:
            break
        text = text.strip()
        if not text:
            continue
        if text.startswith("/"):
            if handle_command(text, agent, checkpointer, console):
                break
            continue
        try:
            agent.run(text, messages=messages)
        except KeyboardInterrupt:
            console.print("\n[yellow]Turn interrupted — session kept.[/yellow]")
        messages = agent.messages
    console.print(f"[dim]session log: {session.path}[/dim]")
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
    if args.prompt:
        return run_once(args.prompt, console, args.mode or "auto", args.resume)
    return run_repl(console, args.mode or "ask", args.resume)
