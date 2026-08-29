"""Unit tests for the tool layer: edit semantics, dispatch fault-tolerance
chain, bash behavior, search tools, todo list. No network required."""
import json

import pytest

from tenon.tools import TOOL_REGISTRY, Tool, ToolContext, ToolResult, dispatch


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


def call(ctx, name, **args):
    return dispatch(TOOL_REGISTRY, name, json.dumps(args), ctx)


# ---------------- edit / write_file / read_file ----------------

def test_edit_unique_match(ctx, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world\nbye\n")
    assert call(ctx, "read_file", path="a.txt").ok
    r = call(ctx, "edit", path="a.txt", old_string="world", new_string="tenon")
    assert r.ok and "1 occurrence" in r.output
    assert f.read_text() == "hello tenon\nbye\n"


def test_edit_no_match_with_did_you_mean(ctx, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("    def main():\n        pass\n")
    call(ctx, "read_file", path="a.txt")
    # wrong indentation on the second line → no exact match, but a near miss exists
    r = call(ctx, "edit", path="a.txt", old_string="def main():\n    pass", new_string="def run():\n    pass")
    assert not r.ok and "not found" in r.output
    assert "line(s) 1" in r.output  # did-you-mean points at the near miss


def test_edit_multiple_matches_requires_replace_all(ctx, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x = 1\ny = 1\n")
    call(ctx, "read_file", path="a.txt")
    r = call(ctx, "edit", path="a.txt", old_string="= 1", new_string="= 2")
    assert not r.ok and "2 locations" in r.output
    r = call(ctx, "edit", path="a.txt", old_string="= 1", new_string="= 2", replace_all=True)
    assert r.ok and "2 occurrence" in r.output
    assert f.read_text() == "x = 2\ny = 2\n"


def test_edit_identical_strings_rejected(ctx, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("same\n")
    call(ctx, "read_file", path="a.txt")
    r = call(ctx, "edit", path="a.txt", old_string="same", new_string="same")
    assert not r.ok and "identical" in r.output


def test_edit_requires_prior_read(ctx, tmp_path):
    (tmp_path / "a.txt").write_text("data\n")
    r = call(ctx, "edit", path="a.txt", old_string="data", new_string="new")
    assert not r.ok and "read-before-write" in r.output


def test_write_file_creates_and_enforces_read_before_overwrite(ctx, tmp_path):
    r = call(ctx, "write_file", path="sub/dir/new.txt", content="fresh\n")
    assert r.ok
    assert (tmp_path / "sub/dir/new.txt").read_text() == "fresh\n"
    # overwriting a file the model itself wrote is allowed — it knows the content
    r = call(ctx, "write_file", path="sub/dir/new.txt", content="v2\n")
    assert r.ok


def test_write_file_blind_overwrite_rejected(ctx, tmp_path):
    (tmp_path / "existing.txt").write_text("someone else's content\n")
    r = call(ctx, "write_file", path="existing.txt", content="blind overwrite")
    assert not r.ok and "read it first" in r.output
    call(ctx, "read_file", path="existing.txt")
    r = call(ctx, "write_file", path="existing.txt", content="after read\n")
    assert r.ok


def test_read_file_line_numbers_offset_and_truncation(ctx, tmp_path):
    f = tmp_path / "many.txt"
    f.write_text("".join(f"line {i}\n" for i in range(1, 51)))
    r = call(ctx, "read_file", path="many.txt", offset=10, limit=5)
    assert r.ok and r.truncated and "offset=15" in r.output
    lines = r.output.splitlines()
    assert lines[0].startswith("10\tline 10")
    assert lines[4].startswith("14\tline 14")
    r = call(ctx, "read_file", path="many.txt", offset=1, limit=10)
    assert r.truncated and "of 50" in r.output and "offset=11" in r.output
    r = call(ctx, "read_file", path="many.txt", offset=49, limit=10)
    assert r.ok and not r.truncated  # reached end of file
    r = call(ctx, "read_file", path="many.txt", offset=51)
    assert not r.ok and "beyond end" in r.output


def test_read_file_missing_and_directory(ctx, tmp_path):
    assert not call(ctx, "read_file", path="nope.txt").ok
    r = call(ctx, "read_file", path=".")
    assert not r.ok and "directory" in r.output


# ---------------- dispatch fault-tolerance chain ----------------

def test_dispatch_unknown_tool(ctx):
    r = dispatch(TOOL_REGISTRY, "nonsense", "{}", ctx)
    assert not r.ok and "unknown_tool" in r.output and "read_file" in r.output


def test_dispatch_repairs_trailing_comma(ctx, tmp_path):
    (tmp_path / "a.txt").write_text("hi\n")
    raw = '{ "path": "a.txt", }'
    r = dispatch(TOOL_REGISTRY, "read_file", raw, ctx)
    assert r.ok and "1\thi" in r.output


def test_dispatch_bad_json_returns_structured_error(ctx):
    r = dispatch(TOOL_REGISTRY, "read_file", "{not json at all", ctx)
    assert not r.ok and "bad_arguments" in r.output and "invalid JSON" in r.output


def test_dispatch_missing_required_and_wrong_type(ctx):
    r = dispatch(TOOL_REGISTRY, "read_file", "{}", ctx)
    assert not r.ok and "missing required parameter 'path'" in r.output
    r = dispatch(TOOL_REGISTRY, "read_file", json.dumps({"path": 42}), ctx)
    assert not r.ok and "must be string" in r.output


class _BoomTool(Tool):
    name = "boom"
    description = "always raises unexpectedly"
    parameters = {"type": "object", "properties": {}}

    def run(self, args, ctx):
        raise RuntimeError("kaboom")


def test_dispatch_internal_error_never_raises(ctx):
    registry = {**TOOL_REGISTRY, "boom": _BoomTool()}
    r = dispatch(registry, "boom", "{}", ctx)
    assert not r.ok and "error=internal" in r.output and "kaboom" in r.output


# ---------------- bash ----------------

def test_bash_echo_and_exit_code(ctx):
    r = call(ctx, "bash", command="echo hello")
    assert r.ok and "hello" in r.output
    r = call(ctx, "bash", command="echo oops >&2; exit 3")
    assert not r.ok and "Exit code: 3" in r.output and "oops" in r.output


def test_bash_cwd_persists(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    call(ctx, "bash", command="cd sub")
    assert ctx.cwd == (tmp_path / "sub").resolve()
    r = call(ctx, "bash", command="pwd")
    assert r.ok and r.output.strip().endswith("/sub")


def test_bash_timeout(ctx):
    r = call(ctx, "bash", command="sleep 5", timeout=1)
    assert not r.ok and "timeout" in r.output


def test_bash_interactive_rejected(ctx):
    r = call(ctx, "bash", command="vim a.txt")
    assert not r.ok and "interactive" in r.output
    r = call(ctx, "bash", command="python")
    assert not r.ok and "interactive" in r.output
    r = call(ctx, "bash", command="python3 -c 'print(1 + 1)'")
    assert r.ok and "2" in r.output


def test_bash_output_truncation(ctx):
    r = call(ctx, "bash", command="python3 -c \"print('x' * 60000)\"")
    assert r.ok and r.truncated and "chars truncated" in r.output


# ---------------- grep / glob / todo ----------------

def test_grep_match_and_invalid_regex(ctx, tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("bar = 2\n")
    r = call(ctx, "grep", pattern="def foo")
    assert r.ok and "a.py:1:def foo():" in r.output
    r = call(ctx, "grep", pattern="nothing_matches_this")
    assert r.ok and "no matches" in r.output
    r = call(ctx, "grep", pattern="[unclosed")
    assert not r.ok and "invalid regex" in r.output


def test_glob_bare_pattern_matches_any_depth(ctx, tmp_path):
    (tmp_path / "deep/nest").mkdir(parents=True)
    (tmp_path / "deep/nest/x.py").write_text("")
    (tmp_path / "top.txt").write_text("")
    r = call(ctx, "glob", pattern="*.py")
    assert r.ok and "x.py" in r.output and "top.txt" not in r.output


def test_todo_write_validation(ctx):
    r = call(ctx, "todo_write", todos=[
        {"content": "task a", "status": "in_progress"},
        {"content": "task b", "status": "pending"},
    ])
    assert r.ok and "[>] task a" in r.output
    r = call(ctx, "todo_write", todos=[
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "in_progress"},
    ])
    assert not r.ok and "only one" in r.output
    r = call(ctx, "todo_write", todos=[{"content": "a", "status": "bogus"}])
    assert not r.ok and "status" in r.output
