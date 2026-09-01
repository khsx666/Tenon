"""Agent loop: a hand-written ReAct loop with multi-layer termination.

The loop itself is deliberately minimal: a message list, one LLM call per
turn, tool dispatch, results paired back by tool_call_id. The main exit is
"the model stopped calling tools"; max_turns, repeated-action detection,
and user interrupt are independent deterministic backstops — loop exit is
never left to the model's own discipline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .config import Config
from .context import (
    COMPRESS_KEEP_TAIL,
    cap_tool_message,
    compress,
    estimate_tokens,
    mask_old_tool_results,
)
from .llm import ContextLengthExceeded, LLMClient
from .prompts import build_system_prompt
from .tools import TOOL_REGISTRY, Tool, ToolContext, ToolResult, all_schemas, dispatch

MAX_LOOP_REPEATS = 3  # identical tool-call batches tolerated before intervening


def _fingerprint(tool_calls) -> str:
    """Canonical fingerprint of one batch of tool calls (loop detection)."""
    parts = []
    for tc in tool_calls:
        try:
            args = json.dumps(json.loads(tc.arguments_json), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            args = tc.arguments_json
        parts.append(f"{tc.name}:{args}")
    return "|".join(parts)


class Agent:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        registry: dict[str, Tool] | None = None,
        ctx: ToolContext | None = None,
        on_event: Callable[..., None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        safety=None,    # SafetyLayer, duck-typed; None = no approval layer
        session=None,   # SessionLogger, duck-typed; None = no persistence
    ):
        self.config = config
        self.llm = llm
        self.registry = registry or TOOL_REGISTRY
        self.schemas = all_schemas() if registry is None else [t.schema() for t in registry.values()]
        self.ctx = ctx or ToolContext(cwd=Path.cwd(), default_timeout=config.timeout)
        self.on_event = on_event or (lambda type_, **data: None)
        self.should_stop = should_stop or (lambda: False)
        self.safety = safety
        self.session = session
        self.messages: list[dict] = []

    def run(self, task: str, messages: list[dict] | None = None) -> str:
        self.task = task
        if messages is None:
            self.messages = []
            self._append({"role": "system", "content": build_system_prompt(self.ctx)})
            self._append({"role": "user", "content": task})
        else:
            self.messages = messages  # resumed session: replayed verbatim
            self._append({"role": "user", "content": task})
        if self.session is not None:
            self.session.log_meta(event="task_start", task=task)
        last_fingerprint: str | None = None
        repeats = 0
        try:
            for _turn in range(self.config.max_turns):
                if self.should_stop():
                    return self._finish("interrupted", "Stopped by user interrupt.")
                self._refresh_system_prompt()
                self._manage_context()
                try:
                    resp = self.llm.chat(self.messages, tools=self.schemas)
                except ContextLengthExceeded:
                    # Recovery path: compress aggressively and keep going.
                    try:
                        self._compress(keep_tail=4)
                        continue
                    except Exception:
                        return self._finish(
                            "context_overflow",
                            "Stopped: the conversation exceeded the model's context "
                            "window and could not be compressed further.",
                        )
                self._append(resp.as_message())
                if resp.text:
                    self.on_event("text", text=resp.text)
                if not resp.tool_calls:
                    return self._finish("completed", resp.text)

                fingerprint = _fingerprint(resp.tool_calls)
                repeats = repeats + 1 if fingerprint == last_fingerprint else 0
                last_fingerprint = fingerprint
                if repeats >= MAX_LOOP_REPEATS:
                    if repeats >= MAX_LOOP_REPEATS + 2:
                        return self._finish(
                            "loop_detected",
                            "Stopped: the model repeated the same tool call(s) "
                            "without making progress.",
                        )
                    warning = (
                        f"You have repeated the exact same tool call(s) {repeats + 1} times "
                        "in a row without progress. Take a different approach, or explain "
                        "what is blocking you."
                    )
                    self._append({"role": "user", "content": f"[loop detector] {warning}"})
                    self.on_event("warning", text=warning)

                for call in resp.tool_calls:
                    self.on_event("tool_call", name=call.name, args=call.arguments_json)
                    verdict = self.safety.check(call.name, call.arguments_json) if self.safety else None
                    if verdict is None or verdict.allowed:
                        result = dispatch(self.registry, call.name, call.arguments_json, self.ctx)
                    else:
                        result = ToolResult(False, f"error=permission: {verdict.reason}")
                        self.on_event("permission", name=call.name, reason=verdict.reason)
                    self._append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": cap_tool_message(result.output),  # L1 hard cap
                    })
                    self.on_event(
                        "tool_result",
                        name=call.name,
                        ok=result.ok,
                        output=result.output,
                        truncated=result.truncated,
                    )
        except KeyboardInterrupt:
            # A hard interrupt mid-turn can leave assistant tool_calls without
            # their results; stub them so the history stays protocol-valid.
            repaired = self._repair_pending_tool_calls()
            if repaired:
                self.on_event(
                    "warning",
                    text=f"marked {repaired} pending tool call(s) as interrupted",
                )
            raise
        return self._finish(
            "max_turns",
            f"Stopped: reached the max_turns limit ({self.config.max_turns}).",
        )

    def _repair_pending_tool_calls(self) -> int:
        """Stub results for tool calls that never got one (e.g. interrupt hit
        mid-dispatch). Idempotent; returns how many were stubbed."""
        answered = {m["tool_call_id"] for m in self.messages if m.get("role") == "tool"}
        pending = [
            tc["id"]
            for m in self.messages
            for tc in (m.get("tool_calls") or [])
            if tc.get("id") not in answered
        ]
        for call_id in pending:
            self._append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "[interrupted: this tool call was cancelled before producing a result]",
            })
        return len(pending)

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system message in place; only its dynamic tail changes,
        keeping the cached prefix stable."""
        self.messages[0] = {"role": "system", "content": build_system_prompt(self.ctx)}

    def _append(self, message: dict) -> None:
        """Append to the conversation and the session log, atomically enough."""
        self.messages.append(message)
        if self.session is not None:
            self.session.log_message(message)

    def _manage_context(self) -> None:
        """L2 every turn; L3 when the estimated size crosses the budget."""
        masked = mask_old_tool_results(self.messages)
        if masked:
            self.on_event("masked", count=masked)
        if estimate_tokens(self.messages) > self.config.context_token_budget:
            try:
                self._compress(keep_tail=COMPRESS_KEEP_TAIL)
            except ValueError:
                pass  # nothing meaningful to compress yet

    def _compress(self, keep_tail: int) -> None:
        before = estimate_tokens(self.messages)
        self.messages = compress(self.messages, self.llm, self.task, keep_tail=keep_tail)
        self.on_event(
            "compact",
            before=before,
            after=estimate_tokens(self.messages),
        )

    def _finish(self, reason: str, text: str) -> str:
        if self.session is not None:
            self.session.log_meta(event="exit", reason=reason)
        self.on_event("exit", reason=reason, text=text)
        return text
