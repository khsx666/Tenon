"""Context management tests: L1 cap, L2 masking, L3 compression. No network."""
import pytest

from tenon.context import (
    MASKED_PLACEHOLDER,
    TOOL_MESSAGE_CAP,
    cap_tool_message,
    compress,
    estimate_tokens,
    find_safe_cut,
    mask_old_tool_results,
)
from tenon.llm import AssistantMessage


class SummaryLLM:
    def __init__(self, text="summary of earlier work"):
        self.text = text
        self.seen = []

    def chat(self, messages, tools=None, stream=False):
        self.seen.append(messages)
        return AssistantMessage(_raw={"role": "assistant", "content": self.text},
                                text=self.text)


def _tool_pair(call_id, output):
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "bash", "arguments": "{}"}}],
    }
    result = {"role": "tool", "tool_call_id": call_id, "content": output}
    return assistant, result


# ---------------- L1 ----------------

def test_cap_tool_message_keeps_head_and_tail():
    big = "H" * TOOL_MESSAGE_CAP + "M" * 5000 + "T" * TOOL_MESSAGE_CAP
    capped = cap_tool_message(big)
    assert len(capped) <= TOOL_MESSAGE_CAP + 200
    assert capped.startswith("H" * 100)
    assert capped.rstrip().endswith("T" * 100)
    assert "truncated" in capped
    assert cap_tool_message("small") == "small"


# ---------------- L2 ----------------

def test_masking_keeps_recent_and_is_idempotent():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(10):
        a, r = _tool_pair(f"c{i}", f"output {i} " + "x" * 100)
        messages += [a, r]
    masked = mask_old_tool_results(messages, keep_recent=3)
    assert masked == 7
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs[:7]] == [MASKED_PLACEHOLDER] * 7
    assert tool_msgs[-1]["content"].startswith("output 9")
    # ids and roles untouched — protocol stays valid
    assert [m["tool_call_id"] for m in tool_msgs] == [f"c{i}" for i in range(10)]
    # second run masks nothing new
    assert mask_old_tool_results(messages, keep_recent=3) == 0


# ---------------- L3 ----------------

def test_find_safe_cut_never_splits_tool_pair():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(5):
        a, r = _tool_pair(f"c{i}", f"out{i}")
        messages += [a, r]
    cut = find_safe_cut(messages, min_tail=3)
    assert messages[cut]["role"] != "tool"
    # every tool_call in the tail has its result in the tail
    tail = messages[cut:]
    ids_answered = {m["tool_call_id"] for m in tail if m["role"] == "tool"}
    for m in tail:
        for tc in m.get("tool_calls") or []:
            assert tc["id"] in ids_answered


def test_estimate_tokens_scales_with_content():
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "x" * 4000}]
    assert estimate_tokens(big) > estimate_tokens(small)
    assert estimate_tokens(big) == 1000


def test_compress_rebuilds_head_and_tail():
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for i in range(10):
        a, r = _tool_pair(f"c{i}", f"out{i}")
        messages += [a, r]
    llm = SummaryLLM()
    rebuilt = compress(messages, llm, "TASK", keep_tail=4)
    assert rebuilt[0]["role"] == "system" and rebuilt[0]["content"] == "SYS"
    assert rebuilt[1]["role"] == "user"
    assert "TASK" in rebuilt[1]["content"] and "summary of earlier work" in rebuilt[1]["content"]
    assert len(rebuilt) == 2 + 4
    # summarizer saw a transcript mentioning tool outputs
    assert "out0" in llm.seen[0][-1]["content"]


def test_compress_refuses_when_nothing_to_cut():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    with pytest.raises(ValueError):
        compress(messages, SummaryLLM(), "t", keep_tail=12)
