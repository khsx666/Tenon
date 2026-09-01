# Tenon

中文 | [English](#english)

一个从零实现的命令行编程智能体：通过大模型的原生 tool calling 接口，自主地读写文件、
执行命令，在终端里完成你交给它的编程任务。

不使用任何 agent 框架/SDK。Agent 循环、上下文管理、工具执行、输出解析、循环终止、
错误处理全部手写，运行时只依赖 `openai` 客户端库、`rich`、`prompt_toolkit` 和标准库。

## 它是怎么工作的

```
用户 ←→ cli.py（REPL / -p 单发；rich 渲染；确认卡）
          │
          ▼
      agent.py —— ReAct 循环：调 LLM → 分发工具 → 结果配对回写 → 终止判断
       ┌──────┼──────────┬───────────┐
       ▼      ▼          ▼           ▼
   llm.py  tools/    context.py  safety.py
  重试与   7 个本地   截断 /      权限模式 +
  解析     工具       遮蔽 /      危险命令
                     摘要压缩     fail-closed 门
```

每一轮的精确顺序：检查中断标志 → 重建 system prompt（静态前缀保持字节级稳定）→
遮蔽过期的工具结果 → 带全部工具 schema 调 LLM → assistant 消息**原样**入史 →
没有 tool_calls 则任务完成；否则做重复动作指纹检查，每个调用先过安全层、再过
dispatch 容错链执行，每个结果按 `tool_call_id` 配对回写。

## 特性

- **手写 ReAct 单循环 + 多层终止**：主出口（模型停止调用工具）、最大轮次、
  重复动作指纹熔断、用户中断、上下文溢出时压缩恢复——全部是确定性代码兜底
- **7 个精心设计的本地工具**：read_file / write_file / edit（精确字符串替换）/
  bash / grep / glob / todo_write
- **上下文三级防御**：输出截断（超长溢出入文件）→ 旧结果遮蔽 → LLM 摘要压缩；
  消息组织 append-only，尽量命中 prompt 缓存
- **结构化错误反馈**：工具失败由确定性代码生成（错误类型 + 定位 + 可操作线索）
  回喂模型；任何工具失败都不会杀死会话
- **安全层**：四档权限模式、敏感路径 deny、写操作工作区约束、危险命令
  fail-closed 检测、文件编辑前快照 + `/undo`
- **会话持久化**：append-only JSONL 事件日志，`-c` 恢复上次会话
- **交互式 REPL**（prompt_toolkit + rich）与单发 `-p` 模式

## 工具一览

| 工具 | 值得说明的行为 |
|------|----------------|
| `read_file` | 带行号、`offset`/`limit` 分页、行数+字节双上限、截断时给出续读提示 |
| `write_file` | 自动建父目录；本会话没读过的文件拒绝覆写（先读后写契约） |
| `edit` | 精确字符串替换：默认唯一匹配、`replace_all` 开关、失配时给相似行提示。刻意不用行号编辑和 diff |
| `bash` | 工作目录跨调用持久、默认 120s 超时（上限 600s）、输出 ~30KB 中段截断且完整版落溢出文件、退出码正常回喂、交互式命令拒绝并给替代写法 |
| `grep` | 有 ripgrep 用 ripgrep，没有则用内置正则兜底；条数上限 + 细化提示 |
| `glob` | `*.py` 这类裸模式匹配任意深度；结果按最近修改排序 |
| `todo_write` | 外化任务清单（注意力锚点）；同一时刻只允许一个 `in_progress` |

## 上下文管理

1. **进上下文之前**：工具输出硬帽截断（保头保尾）；bash 超长输出的完整版写入
   溢出文件并把路径回喂给模型。
2. **轮次推进中**：最近 8 条之外的旧 tool result 被替换成占位符。消息不删除，
   角色、顺序、tool_call 配对保持协议合法，缓存前缀尽量稳定。
3. **接近预算时**：LLM 摘要早期轮次，重建为 system + 任务/摘要 + 最近若干条原样，
   切割点只在协议安全边界（绝不拆散 tool_call/result 配对）。真正触发
   `context_length_exceeded` 时走激进压缩恢复，而不是直接终止任务。

## 安全

审批（策略）与沙箱（执行）是两个独立维度；本项目实现软件策略层：

- **权限四档**：`read-only`（禁一切变更）→ `ask`（编辑和命令逐次确认，默认）→
  `auto-edit`（编辑自由、命令确认）→ `auto`（除危险模式外自由）。
  `-p` 非交互模式默认 `auto`。
- **deny 规则**：敏感路径（`.env`、`.ssh` 等）任何工具都不读不写；
  写操作限制在工作区内。
- **危险命令 fail-closed**：命令经 shell 词法解析（不是纯正则），命中危险模式——
  `sudo`、工作区外的递归删除、`dd` 写设备、`git push --force`、网络下载管道进
  shell 等——一律升级人工确认，且不被 `a`（本会话总是允许）记住；无人可答时拒绝。
- **可恢复性**：任务中每个文件第一次被写/改前自动快照；`/undo` 恢复被修改的文件、
  删除被创建的文件。bash 的副作用无法回滚，如实说明。

## 安装

需要 Python 3.10+ 和一个支持原生 tool calling 的 OpenAI 兼容端点。

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你的端点凭据
pip install pytest     # 仅运行测试套件时需要（tests/）
```

配置（环境变量优先于 `.env`）：

| 变量 | 含义 |
|------|------|
| `TENON_API_KEY` | API key（绝不入库、绝不打印） |
| `TENON_BASE_URL` | OpenAI 兼容的 chat completions 端点 |
| `TENON_MODEL` | 该端点上的模型名 |

## 使用

```bash
python -m tenon                 # 交互式 REPL（编辑/命令前会询问）
python -m tenon --mode auto     # 同上，减少确认
python -m tenon -p "任务"       # 单发模式，非交互
python -m tenon -c              # 恢复当前目录的最近一次会话
```

REPL 内：`/help` 列出全部内置命令（`/mode /undo /cost /compact /quit`）；
Ctrl+C 中断当前轮、会话保留；Ctrl+D 退出。每次运行都会在 `.sessions/` 落一份
会话日志，退出时显示 token 用量。

## 测试

```bash
pytest tests/    # 60 个单元测试：工具、容错链、agent 循环、上下文、安全
```

agent 循环的测试使用脚本化的 mock LLM，整个套件完全离线可跑。

## 项目结构

```
src/tenon/
├── cli.py          # REPL + 单发入口、渲染、确认卡
├── agent.py        # ReAct 循环与终止层
├── llm.py          # OpenAI 兼容客户端：重试退避、解析、用量统计
├── config.py       # 环境变量/.env 加载；key 从不出现在错误信息里
├── context.py      # 截断 / 遮蔽 / 摘要压缩
├── prompts.py      # system prompt：静态段前置、动态段后置
├── safety.py       # 权限模式、路径规则、危险命令门
├── checkpoint.py   # 编辑前快照与 /undo
├── session.py      # append-only JSONL 会话日志与重放
└── tools/          # 基座（ToolResult、dispatch 容错链）+ 七个工具
```

## 设计取舍（为什么这么做的简短版）

- **极简循环，深度在护栏。** Agent 循环本身刻意保持平凡；工程量集中在终止条件、
  错误反馈、上下文管理和安全上——这些才是 agent 实际失效的地方。
- **精确字符串替换编辑。** LLM 数行号不可靠、diff 的 context 行干扰生成；
  精确匹配把编辑变成可验证的搜索问题，失配/歧义都有结构化反馈。
- **错误由代码生成，不让模型自评。** 结构化反馈（类型 + 定位 + 线索）才能让
  自我修复收敛；验证依赖外部 oracle（测试），而不是纯语言反思。
- **append-only 历史。** assistant 消息原样回放、tool 结果按 id 配对、旧结果
  只遮蔽不删除——协议正确性和 prompt 缓存两头都占。

---

<a id="english"></a>

## English

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
