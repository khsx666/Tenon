"""Session persistence: append-only JSONL event log + resume.

Every message appended to the conversation is logged as one JSON line
(`{"ts", "type": "message", "message": {...}}`), plus meta events for the
task and the exit. Resuming (`tenon -c`) replays the message events of the
most recent log in the workspace. A write failure warns once and disables
logging — it never interrupts the session.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SESSIONS_DIR = ".sessions"


class SessionLogger:
    def __init__(self, workspace: Path, resume_from: Path | None = None):
        self.workspace = workspace
        self._disabled = False
        if resume_from is not None:
            self.path = resume_from
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path = workspace / SESSIONS_DIR / f"session-{stamp}-{int(time.time() * 1000) % 100000}.jsonl"

    @staticmethod
    def latest(workspace: Path) -> Path | None:
        directory = workspace / SESSIONS_DIR
        if not directory.is_dir():
            return None
        logs = sorted(directory.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime)
        return logs[-1] if logs else None

    def log(self, type_: str, **data) -> None:
        if self._disabled:
            return
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": type_, **data}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._disabled = True
            print(f"[session] warning: cannot write {self.path}: {exc}; logging disabled")

    def log_message(self, message: dict) -> None:
        self.log("message", message=message)

    def log_meta(self, **data) -> None:
        self.log("meta", **data)

    @staticmethod
    def replay(path: Path) -> list[dict]:
        """Rebuild the message list from a session log (latest wins)."""
        messages: list[dict] = []
        with path.open() as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line
                if record.get("type") == "message":
                    messages.append(record["message"])
        return messages
