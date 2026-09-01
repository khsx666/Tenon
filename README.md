# Tenon

A minimal coding agent built from scratch — it talks to an LLM through the native
tool-calling API and autonomously reads/writes files and runs commands to complete
programming tasks in your terminal.

No agent frameworks. The agent loop, context management, tool execution, output
parsing, loop termination and error handling are all hand-written on top of the
`openai` Python client only.

## How it works

```
user ←→ cli.py (REPL / -p one-shot; rich rendering; confirmation cards)
          │
          ▼
      agent.py — ReAct loop: LLM call → tool dispatch → result pairing → termination
       ┌──────┼──────────┬───────────┐
       ▼      ▼          ▼           ▼
   llm.py  tools/    context.py  safety.py
  retry &  7 local   truncate /  permission
  parsing  tools     mask /      modes &
                     summarize   fail-closed gate
```

Each turn, in order: check the interrupt flag → rebuild the system prompt (static
prefix stays byte-stable) → mask stale tool results → call the LLM with all tool
schemas → append the assistant message **verbatim** → if there are no tool calls,
the task is done; otherwise check the repeated-action fingerprint, run each call
through the safety layer and the dispatch fault-tolerance chain, and pair every
result back with its `tool_call_id`.

## Features

- **Hand-rolled ReAct agent loop** with multi-layer termination: no-tool-call exit,
  max turns, repeated-action fingerprint circuit breaker, user interrupt, and
  context-overflow recovery via compression
- **7 carefully-designed local tools**: read_file / write_file / edit
  (exact string replacement) / bash / grep / glob / todo_write
- **Three-layer context management**: output truncation with overflow files →
  stale-result masking → LLM summarization, with a cache-friendly append-only
  message layout
- **Structured error feedback**: every tool failure is rendered by deterministic
  code (error type + location + actionable hint) and fed back to the model;
  a failing tool never kills the session
- **Safety layer**: four permission modes, sensitive-path deny rules, workspace
  confinement for writes, fail-closed dangerous-command detection, file snapshots
  with `/undo`
- **Session persistence**: append-only JSONL event log, `-c` resume
- **Interactive REPL** (prompt_toolkit + rich) and one-shot `-p` mode

## Tools

| Tool | Behavior worth knowing |
|------|------------------------|
| `read_file` | Line numbers, `offset`/`limit` paging, line+byte caps, hints on truncation |
| `write_file` | Creates parent dirs; refuses to overwrite a file not read this session |
| `edit` | Exact string replacement; unique match required unless `replace_all`; did-you-mean hints on miss. No line-number edits, no diffs |
| `bash` | Persistent cwd across calls, 120s timeout (max 600s), ~30KB middle truncation, full output spilled to an overflow file, exit code reported, interactive commands rejected with alternatives |
| `grep` | ripgrep when available, built-in regex fallback; result cap with refinement hints |
| `glob` | Bare patterns like `*.py` match at any depth; results sorted by recency |
| `todo_write` | Externalized task list; exactly one item may be `in_progress` |

## Context management

1. **Before entering context** — tool outputs are capped (head+tail); oversized bash
   output is written to an overflow file whose path is fed back.
2. **Between turns** — tool results older than the most recent 8 are masked with a
   placeholder. Nothing is deleted, so roles, ordering and tool-call pairing stay
   protocol-valid and the cache prefix stays as stable as possible.
3. **Near the budget** — earlier turns are summarized by the LLM and rebuilt as
   system + task/summary + a verbatim recent tail, cut only at protocol-safe
   boundaries. Hitting `context_length_exceeded` triggers aggressive compression
   instead of aborting.

## Safety

Approval (policy) is treated as a dimension separate from sandboxing (execution);
Tenon implements the software-level policy layer:

- **Permission modes**: `read-only` (no mutations) → `ask` (confirm every edit and
  command) → `auto-edit` (edits free, commands confirmed) → `auto` (free except
  dangerous patterns). `-p` defaults to `auto`; the REPL defaults to `ask`.
- **Deny rules**: sensitive paths (`.env`, `.ssh`, …) are never read or written;
  writes are confined to the workspace.
- **Fail-closed command gate**: commands are parsed with shell lexing (not plain
  regex) and dangerous patterns — `sudo`, recursive deletes outside the workspace,
  `dd` to devices, `git push --force`, network-pipe-to-shell, … — always escalate
  to a human, are never remembered by the `a`(lways) answer, and are refused when
  no one can answer.
- **Recoverability**: every file is snapshotted before its first write in a task;
  `/undo` restores modified files and deletes created ones. Bash side effects
  cannot be rolled back — stated plainly.

## Setup

Requires Python 3.10+ and an OpenAI-compatible endpoint whose model supports
native tool calling.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your endpoint credentials
pip install pytest     # only needed to run the test suite (tests/)
```

Configuration (environment variables win over `.env`):

| Variable | Meaning |
|----------|---------|
| `TENON_API_KEY` | API key (never committed, never printed) |
| `TENON_BASE_URL` | OpenAI-compatible chat-completions endpoint |
| `TENON_MODEL` | Model name on that endpoint |

## Usage

```bash
python -m tenon                 # interactive REPL (asks before edits/commands)
python -m tenon --mode auto     # same, with fewer confirmations
python -m tenon -p "task"       # one-shot, non-interactive
python -m tenon -c              # resume the last session in this directory
```

In the REPL: `/help` lists the slash commands (`/mode /undo /cost /compact /quit`),
Ctrl+C interrupts the current turn, Ctrl+D exits. Every run writes a session log
under `.sessions/` and shows token usage on exit.

## Testing

```bash
pytest tests/    # 60 unit tests: tools, dispatch chain, agent loop, context, safety
```

The agent-loop tests use a scripted mock LLM, so the suite runs fully offline.

## Project layout

```
src/tenon/
├── cli.py          # REPL + one-shot entry, rendering, confirmation cards
├── agent.py        # the ReAct loop and its termination layers
├── llm.py          # OpenAI-compatible client: retry/backoff, parsing, usage
├── config.py       # env/.env loading; the key never appears in errors
├── context.py      # truncation / masking / summarization
├── prompts.py      # system prompt: static section first, dynamic tail last
├── safety.py       # permission modes, path rules, dangerous-command gate
├── checkpoint.py   # pre-edit snapshots and /undo
├── session.py      # append-only JSONL session log and replay
└── tools/          # base (ToolResult, dispatch chain) + the seven tools
```

## Design notes

- **Minimal loop, deep guardrails.** The agent loop itself is deliberately trivial;
  the engineering depth lives in termination layers, error feedback, context
  management and safety.
- **Exact-string-replacement editing.** LLMs count lines unreliably and diffs add
  distracting context lines; exact match turns editing into a verifiable search
  problem, with structured miss/ambiguity feedback.
- **Errors are generated by code, not judged by the model.** Structured feedback
  (type + location + hint) makes self-repair converge; verification is pushed
  toward external oracles (tests), not language-only reflection.
- **Append-only history.** Assistant messages replay verbatim, tool results pair by
  id, stale results are masked rather than deleted — keeping both the protocol and
  the prompt cache happy.
