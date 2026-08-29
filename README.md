# Tenon

A minimal coding agent built from scratch — — it talks to an LLM through the native
tool-calling API and autonomously reads/writes files and runs commands to complete
programming tasks in your terminal.

No agent frameworks. The agent loop, context management, tool execution, output
parsing, loop termination and error handling are all hand-written on top of the
`openai` Python client only.

## Features

- **Hand-rolled ReAct agent loop** with multi-layer termination (no-tool-call exit,
  max turns, cost budget, user interrupt, repeated-action detection)
- **7 carefully-designed local tools**: read_file / write_file / edit
  (exact string replacement) / bash / grep / glob / todo_write
- **Three-layer context management**: output truncation → old-result masking →
  LLM summarization, with cache-friendly append-only message layout
- **Safety layer**: permission modes, path whitelist, command approval gate,
  git checkpoints with /undo
- **Test-driven self-repair**: the agent runs tests, reads failures and fixes
  its own edits (budgeted retries)
- **Interactive REPL** (prompt_toolkit + rich) and one-shot `-p` mode

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenAI-compatible endpoint credentials
```

## Usage

```bash
python -m tenon            # interactive REPL
python -m tenon -p "create a CLI todo app with tests and run them"
python -m tenon -c         # resume last session
```
