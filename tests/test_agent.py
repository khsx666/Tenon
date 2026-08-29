"""Agent loop tests with a scripted mock LLM — no network required."""
import pytest

from tenon.agent import Agent
from tenon.config import Config
from tenon.llm import AssistantMessage, ToolCall
from tenon.tools import ToolContext


def text_msg(text: str) -> AssistantMessage:
    return AssistantMessage(_raw={"role": "assistant", "content": text}, text=text)


def tool_msg(calls: list[ToolCall]) -> AssistantMessage:
    raw_calls = [
        {"id": c.id, "type": "function",
         "function": {"name": c.name, "arguments": c.arguments_json}}
        for c in calls
    ]
    return AssistantMessage(
        _raw={"role": "assistant", "content": "", "tool_calls": raw_calls},
        text="",
        tool_calls=calls,
    )


class MockLLM:
    """Plays back a fixed script of assistant messages, recording what it saw."""

    def __init__(self, script: list[AssistantMessage]):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, stream=False):
        self.seen.append([dict(m) for m in messages])
        assert self.script, "mock script exhausted"
        return self.script.pop(0)


@pytest.fixture
def config():
    return Config(api_key="x", base_url="http://x", model="x", max_turns=10)


def make_agent(config, llm, tmp_path):
    return Agent(config, llm, ctx=ToolContext(cwd=tmp_path))


def test_main_exit_when_no_tool_calls(config, tmp_path):
    llm = MockLLM([text_msg("all done")])
    assert make_agent(config, llm, tmp_path).run("task") == "all done"
    assert len(llm.seen) == 1


def test_tool_roundtrip_pairs_results_by_id(config, tmp_path):
    call = ToolCall(id="call_1", name="bash", arguments_json='{"command": "echo hi"}')
    llm = MockLLM([tool_msg([call]), text_msg("finished")])
    assert make_agent(config, llm, tmp_path).run("task") == "finished"
    second = llm.seen[1]
    assistant = second[-2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    result = second[-1]
    assert result["role"] == "tool" and result["tool_call_id"] == "call_1"
    assert "hi" in result["content"]


def test_parallel_tool_calls_each_get_paired_result(config, tmp_path):
    calls = [
        ToolCall(id="c1", name="bash", arguments_json='{"command": "echo a"}'),
        ToolCall(id="c2", name="bash", arguments_json='{"command": "echo b"}'),
    ]
    llm = MockLLM([tool_msg(calls), text_msg("done")])
    assert make_agent(config, llm, tmp_path).run("task") == "done"
    tail = llm.seen[1][-2:]
    assert [m["tool_call_id"] for m in tail] == ["c1", "c2"]


def test_max_turns_backstop(config, tmp_path):
    config.max_turns = 3
    call = ToolCall(id="c", name="glob", arguments_json='{"pattern": "*.py"}')
    llm = MockLLM([tool_msg([call])] * 3)
    answer = make_agent(config, llm, tmp_path).run("task")
    assert "max_turns" in answer
    assert len(llm.seen) == 3


def test_loop_detection_warns_then_breaks(config, tmp_path):
    config.max_turns = 20
    call = ToolCall(id="c", name="bash", arguments_json='{"command": "echo same"}')
    llm = MockLLM([tool_msg([call])] * 20)
    agent = make_agent(config, llm, tmp_path)
    answer = agent.run("task")
    assert "repeated the same tool call" in answer
    warnings = [m for m in agent.messages
                if m["role"] == "user" and "loop detector" in m["content"]]
    assert warnings
    assert len(llm.seen) < 20


def test_tool_error_is_fed_back_not_raised(config, tmp_path):
    bad = ToolCall(id="b", name="nonexistent_tool", arguments_json="{}")
    llm = MockLLM([tool_msg([bad]), text_msg("recovered")])
    assert make_agent(config, llm, tmp_path).run("task") == "recovered"
    result = llm.seen[1][-1]
    assert result["role"] == "tool" and "unknown_tool" in result["content"]


def test_interrupt_hook_stops_loop(config, tmp_path):
    call = ToolCall(id="c", name="bash", arguments_json='{"command": "true"}')
    llm = MockLLM([tool_msg([call])] * 10)
    agent = Agent(config, llm, ctx=ToolContext(cwd=tmp_path),
                  should_stop=lambda: True)
    assert "interrupt" in agent.run("task").lower()
    assert len(llm.seen) == 0
