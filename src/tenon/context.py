"""Context management: three layers of defense against context rot.

L1  before entering context — tool outputs are capped (head+tail); bash
    spills full output to an overflow file (wired in the tools layer).
L2  between turns       — tool results older than the most recent
    `MASK_KEEP_RECENT` are masked with a placeholder. Messages are never
    deleted: roles, order, and tool_call_id pairing stay intact, which
    keeps the cache prefix as stable as possible. Masking is nearly free
    compared with LLM summarization, so it runs every turn.
L3  near the budget     — the early conversation is summarized by the LLM
    and rebuilt as: system + task/summary + recent tail. The tail is cut
    only at protocol-safe boundaries (never inside a tool_call/result
    pair). Also used as the recovery path for context_length_exceeded.

Token sizes are estimated with a chars/4 heuristic — good enough for a
trigger threshold, and documented as an estimate.
"""
from __future__ import annotations

import json

MASK_KEEP_RECENT = 8        # tool results that stay fully visible
COMPRESS_KEEP_TAIL = 12     # messages kept verbatim when compressing
TOOL_MESSAGE_CAP = 30_000   # hard cap for any single tool message (chars)
MASKED_PLACEHOLDER = "[masked: earlier tool output removed to save context]"


def cap_tool_message(content: str) -> str:
    """L1: hard-cap a tool message, keeping head and tail."""
    if len(content) <= TOOL_MESSAGE_CAP:
        return content
    half = TOOL_MESSAGE_CAP // 2
    return (
        f"{content[:half]}\n[... {len(content) - TOOL_MESSAGE_CAP} chars truncated; "
        "page with offset or grep the overflow file ...]\n"
        f"{content[-half:]}"
    )


def mask_old_tool_results(messages: list[dict], keep_recent: int = MASK_KEEP_RECENT) -> int:
    """L2: replace old tool outputs with a placeholder, in place. Idempotent.

    Returns the number of messages masked on this call.
    """
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    stale = tool_positions[:-keep_recent] if len(tool_positions) > keep_recent else []
    masked = 0
    for i in stale:
        if messages[i].get("content") != MASKED_PLACEHOLDER:
            messages[i] = {**messages[i], "content": MASKED_PLACEHOLDER}
            masked += 1
    return masked


def estimate_tokens(messages: list[dict]) -> int:
    """Rough chars/4 estimate over everything that goes on the wire."""
    chars = 0
    for m in messages:
        for key in ("content", "reasoning_content"):  # both go on the wire
            value = m.get(key)
            if isinstance(value, str):
                chars += len(value)
        if m.get("tool_calls"):
            chars += len(json.dumps(m["tool_calls"]))
    return chars // 4


def find_safe_cut(messages: list[dict], min_tail: int) -> int:
    """First index of the kept tail, never inside a tool_call/result pair.

    Starts `min_tail` messages from the end and walks backwards past any
    leading tool messages to their assistant parent, so the tail always
    begins at a message whose follow-ups are fully contained in it.
    """
    i = max(1, len(messages) - min_tail)
    while i > 1 and messages[i].get("role") == "tool":
        i -= 1
    return i


_SUMMARY_INSTRUCTION = """\
Summarize this coding-agent conversation transcript so the agent can
continue working. Capture, densely and factually (bullet points, <= 400
words): the original task; what has been accomplished; key file paths and
their current state; important decisions; errors hit and how they were
resolved; what remains to be done.

Transcript:
"""


def _render_transcript(messages: list[dict], max_chars: int = 40_000) -> str:
    parts: list[str] = []
    used = 0
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc.get('function', {}).get('name', '?')}({tc.get('function', {}).get('arguments', '')[:200]})"
                for tc in m["tool_calls"]
            )
            content = (content + " " if content else "") + f"[calls: {calls}]"
        if len(content) > 1000:
            content = content[:500] + "\n[...]\n" + content[-500:]
        entry = f"{role}: {content}"
        if used + len(entry) > max_chars:
            parts.append("[... older messages elided ...]")
            break
        parts.append(entry)
        used += len(entry)
    return "\n".join(parts)


def compress(messages: list[dict], llm, task: str,
             keep_tail: int = COMPRESS_KEEP_TAIL) -> list[dict]:
    """L3: summarize the early conversation and rebuild the message list.

    Layout after compression: system (unchanged) + one user message holding
    the original task and the summary + the verbatim recent tail.
    Raises ValueError when there is nothing meaningful to compress.
    """
    cut = find_safe_cut(messages, keep_tail)
    if cut <= 2:
        raise ValueError("nothing meaningful to compress")
    system_msg, middle, tail = messages[0], messages[2:cut], messages[cut:]
    summary = llm.chat([
        {"role": "user", "content": _SUMMARY_INSTRUCTION + _render_transcript(middle)}
    ]).text
    head = {
        "role": "user",
        "content": (
            f"Original task: {task}\n\n"
            "[context compressed] Summary of earlier work (details were "
            f"summarized to free context space; the recent conversation "
            f"follows verbatim):\n{summary}"
        ),
    }
    return [system_msg, head, *tail]
