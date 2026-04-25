# Gemini Delegation

Reference for Claude agents on how and when to delegate tasks to Gemini
via `container/scripts/gemini-ask.sh`. Applies to Claude orchestrators only
— Gemini must ignore this doc.

## When to delegate to Gemini

**Default to Gemini for:**
- Any web research or documentation lookup (always prefer Gemini over WebSearch/WebFetch)
- Reading/analyzing more than ~10 files or ~1000 lines of unfamiliar code
- Summarizing or reviewing large outputs (logs, eval results, datasets)
- Tasks where you would otherwise need to load many files into your own context

**Handle directly when:**
- The task needs tight tool-call loops (file edits, running commands, iterating on errors)
- The task involves writing or modifying files (use worktree isolation if delegating)
- The full context is already in scope and delegation overhead isn't worth it

## How to delegate

**Read-only / analysis (no file modifications):**
```bash
bash container/scripts/gemini-ask.sh "task description"
# optional flags:
#   --pro               explicitly use gemini-3.1-pro-preview (same as default)
#   --auto              use the Gemini CLI default model (omits -m flag)
#   --flash             use gemini-3-flash-preview
#   --silence-timeout N seconds of silence before declaring a hang (default 300)
```

**Choosing the model:**
- Default / `--pro` (`gemini-3.1-pro-preview`): use for complex reasoning, large context, deep code analysis, multi-step tasks.
- `--auto` (Gemini CLI default, no `-m` flag): use for lightweight tasks where Pro is overkill — short lookups, quick summaries, simple Q&A, single-file reads.
- `--flash` (`gemini-3-flash-preview`): use for fast structured reads — log parsing, metric extraction, large output summarization.

**File-modifying tasks requiring isolation:**
Spawn a Haiku agent with `isolation: "worktree"`. The Haiku agent is a
passthrough only — it must not make decisions. Its sole job:
1. Run `bash container/scripts/gemini-ask.sh [--pro|--auto|--flash] "task"`
2. Relay the full output and exit code verbatim back to the orchestrator

The orchestrator then decides what to do based on the exit code and output.

## Exit codes from gemini-ask.sh
- `0` — success; response in stdout
- `2` — quota exceeded; handle the task directly, do not retry Gemini
- `3` — hung; Gemini stopped producing output; partial work may exist in the worktree

## Worktree state after Haiku returns
- Gemini made no changes → worktree auto-cleaned
- Gemini made partial changes → worktree preserved, branch name returned; orchestrator
  decides whether to continue, retry, or discard
