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
from .llm import ContextLengthExceeded, LLMClient
from .prompts import build_system_prompt
from .tools import TOOL_REGISTRY, Tool, ToolContext, all_schemas, dispatch

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
    ):
        self.config = config
        self.llm = llm
        self.registry = registry or TOOL_REGISTRY
        self.schemas = all_schemas() if registry is None else [t.schema() for t in registry.values()]
        self.ctx = ctx or ToolContext(cwd=Path.cwd(), default_timeout=config.timeout)
        self.on_event = on_event or (lambda type_, **data: None)
        self.should_stop = should_stop or (lambda: False)
        self.messages: list[dict] = []

    def run(self, task: str) -> str:
        self.messages = [
            {"role": "system", "content": build_system_prompt(self.ctx)},
            {"role": "user", "content": task},
        ]
        last_fingerprint: str | None = None
        repeats = 0
        for _turn in range(self.config.max_turns):
            if self.should_stop():
                return self._finish("interrupted", "Stopped by user interrupt.")
            self._refresh_system_prompt()
            try:
                resp = self.llm.chat(self.messages, tools=self.schemas)
            except ContextLengthExceeded:
                # M4 wires real compression behind this branch.
                return self._finish(
                    "context_overflow",
                    "Stopped: the conversation exceeded the model's context window.",
                )
            self.messages.append(resp.as_message())
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
                self.messages.append({"role": "user", "content": f"[loop detector] {warning}"})
                self.on_event("warning", text=warning)

            for call in resp.tool_calls:
                self.on_event("tool_call", name=call.name, args=call.arguments_json)
                result = dispatch(self.registry, call.name, call.arguments_json, self.ctx)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.output,
                })
                self.on_event(
                    "tool_result",
                    name=call.name,
                    ok=result.ok,
                    output=result.output,
                    truncated=result.truncated,
                )
        return self._finish(
            "max_turns",
            f"Stopped: reached the max_turns limit ({self.config.max_turns}).",
        )

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system message in place; only its dynamic tail changes,
        keeping the cached prefix stable."""
        self.messages[0] = {"role": "system", "content": build_system_prompt(self.ctx)}

    def _finish(self, reason: str, text: str) -> str:
        self.on_event("exit", reason=reason, text=text)
        return text
