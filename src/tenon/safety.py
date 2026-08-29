"""Safety layer: policy (approval) checks — a dimension distinct from
sandboxing (execution isolation, out of scope for this project).

Three software-level mechanisms:
1. deny rules      — sensitive paths (.env/.ssh/...) and writes outside the
                     workspace are refused outright;
2. confirmation    — mutations and bash commands are gated by the
                     permission mode; dangerous command patterns (detected
                     with shell-aware parsing, not pure regex) always
                     escalate to a human, fail-closed;
3. recoverability  — handled separately by checkpoint.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .tools.bash_tool import _command_segments

WORKSPACE_MUTATING_TOOLS = {"write_file", "edit"}
READ_ONLY_TOOLS = {"read_file", "grep", "glob", "todo_write"}
SENSITIVE_NAMES = {".env", ".ssh", ".aws", ".gnupg", ".netrc", ".gitconfig"}


class PermissionMode(Enum):
    READ_ONLY = "read-only"   # no mutations at all
    ASK = "ask"               # confirm every mutation (edits + bash)
    AUTO_EDIT = "auto-edit"   # edits free, bash confirms
    AUTO = "auto"             # everything free except dangerous patterns

    @classmethod
    def parse(cls, text: str) -> "PermissionMode":
        for mode in cls:
            if mode.value == text:
                return mode
        raise ValueError(f"unknown permission mode {text!r}; choose from "
                         f"{[m.value for m in cls]}")


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


class SafetyLayer:
    """Evaluates every tool call before dispatch."""

    def __init__(
        self,
        mode: PermissionMode,
        workspace: Path,
        confirm_fn: Callable[[str], str] | None = None,  # returns "y" | "n" | "a"
    ):
        self.mode = mode
        self.workspace = workspace.resolve()
        self.confirm_fn = confirm_fn or (lambda _prompt: "n")  # fail-closed
        self._always: set[str] = set()   # session-scoped "always allow" memory

    # ---------------- public entry ----------------

    def check(self, name: str, args_json: str) -> Verdict:
        if self.mode is PermissionMode.READ_ONLY and name not in READ_ONLY_TOOLS:
            return Verdict(False, f"{name} is denied in read-only mode")
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError:
            return Verdict(True)  # malformed args: let dispatch produce the structured error
        if not isinstance(args, dict):
            return Verdict(True)

        if name in ("read_file", "write_file", "edit", "grep", "glob"):
            verdict = self._check_paths(name, args)
            if verdict is not None:
                return verdict
        if name == "bash":
            return self._check_bash(args.get("command", ""))
        if name in WORKSPACE_MUTATING_TOOLS and self.mode is PermissionMode.ASK:
            return self._confirm(f"allow {name}?", key=name)
        return Verdict(True)

    # ---------------- paths ----------------

    def _check_paths(self, name: str, args: dict) -> Verdict | None:
        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            return None
        target = Path(raw)
        if not target.is_absolute():
            target = (self.workspace / target)
        target = target.resolve()
        if any(part in SENSITIVE_NAMES or part.startswith(".env") for part in target.parts):
            return Verdict(False, f"{target} is a sensitive path (credentials); access denied")
        if name in WORKSPACE_MUTATING_TOOLS and not self._in_workspace(target):
            return Verdict(False, f"{target} is outside the workspace {self.workspace}; writes denied")
        return None

    def _in_workspace(self, path: Path) -> bool:
        try:
            path.relative_to(self.workspace)
            return True
        except ValueError:
            return False

    # ---------------- bash ----------------

    def _check_bash(self, command: str) -> Verdict:
        danger = self._dangerous_pattern(command)
        if danger is not None:
            # fail-closed: never memorized, always asks a human
            answer = self.confirm_fn(f"DANGEROUS command pattern: {danger}\n{command!r}\nRun anyway?")
            if answer == "y":
                return Verdict(True)
            return Verdict(False, f"command rejected by user ({danger})")
        if self.mode in (PermissionMode.ASK, PermissionMode.AUTO_EDIT):
            head = _head_token(command)
            return self._confirm(f"run bash command?\n{command!r}", key=f"bash:{head}")
        return Verdict(True)

    def _dangerous_pattern(self, command: str) -> str | None:
        segments = _command_segments(command)
        if not segments and command.strip():
            return "unparseable command (fail-closed)"
        heads = [Path(tokens[0]).name for tokens in segments]
        for tokens in segments:
            prog = Path(tokens[0]).name
            flags = {t for t in tokens[1:] if t.startswith("-")}
            targets = [t for t in tokens[1:] if not t.startswith("-")]
            if prog in ("sudo", "doas"):
                return "privilege escalation"
            if prog == "rm" and _has_recursive_flag(flags):
                for target in targets:
                    if target in ("/", "/*", "~", "$HOME", "${HOME}"):
                        return f"recursive delete of {target}"
                    resolved = Path(target).expanduser()
                    if not resolved.is_absolute():
                        resolved = (self.workspace / resolved).resolve()
                    if not self._in_workspace(resolved):
                        return f"recursive delete outside workspace: {target}"
            if prog == "dd" and any(t.startswith("of=/dev/") for t in tokens):
                return "raw write to block device"
            if prog.startswith("mkfs") or prog == "fdisk":
                return "filesystem formatting"
            if prog == "git":
                if "push" in tokens and flags & {"--force", "-f", "--force-with-lease"}:
                    return "force-pushing rewrites shared history"
                if "reset" in tokens and "--hard" in tokens:
                    return "git reset --hard discards uncommitted work"
                if "clean" in tokens and any("f" in f for f in flags):
                    return "git clean -f deletes untracked files"
            if prog in ("shutdown", "reboot", "halt"):
                return "system power operation"
        if {"curl", "wget"} & set(heads) and {"sh", "bash", "zsh"} & set(heads):
            return "piping a network download into a shell"
        return None

    # ---------------- confirmation ----------------

    def _confirm(self, prompt: str, key: str | None) -> Verdict:
        if key is not None and key in self._always:
            return Verdict(True)
        answer = self.confirm_fn(prompt)
        if answer == "a" and key is not None:
            self._always.add(key)
            return Verdict(True)
        if answer in ("y", "a"):
            return Verdict(True)
        return Verdict(False, "rejected by user")


def _has_recursive_flag(flags: set[str]) -> bool:
    if "--recursive" in flags:
        return True
    short = [f[1:] for f in flags if f.startswith("-") and not f.startswith("--")]
    return any("r" in f or "R" in f for f in short)


def _head_token(command: str) -> str:
    segments = _command_segments(command)
    return Path(segments[0][0]).name if segments else command.strip().split(maxsplit=1)[0] if command.strip() else "?"
