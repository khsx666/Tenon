"""Recoverability: per-edit snapshots + /undo, plus an informational git
checkpoint.

File-tool side effects are fully reversible: the first time a task touches
a file, its original bytes (or its non-existence) are captured, so /undo
restores every file the task modified and deletes every file it created.
Bash side effects cannot be rolled back — stated plainly in the docs and
in /undo output. If the workspace is a git repo, the HEAD sha at task
start is recorded so the user can also `git reset --hard <sha>` manually.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class Checkpointer:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._snapshots: dict[Path, bytes | None] = {}  # path -> original bytes (None = did not exist)
        self.head_sha = self._read_head()

    def _read_head(self) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=self.workspace,
            )
            return proc.stdout.strip() or None if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def snapshot(self, path: Path) -> None:
        """ctx.on_file_write hook: capture pre-write state on first touch."""
        if path not in self._snapshots:
            self._snapshots[path] = path.read_bytes() if path.is_file() else None

    @property
    def touched_files(self) -> list[Path]:
        return list(self._snapshots)

    def undo(self) -> list[str]:
        """Restore every snapshotted file; returns human-readable actions."""
        actions: list[str] = []
        for path, data in reversed(list(self._snapshots.items())):
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                    actions.append(f"deleted {path}")
                else:
                    path.write_bytes(data)
                    actions.append(f"restored {path}")
            except OSError as exc:
                actions.append(f"FAILED to restore {path}: {exc}")
        self._snapshots.clear()
        return actions
