# OpenCode Delegation

Reference for orchestrator agents on how to delegate tasks to OpenCode
via `container/scripts/opencode-ask.sh`. OpenCode provides access to multiple
AI models (GPT-5.4, GPT-5.3-Codex, MiniMax M2.5, Nemotron Super) from a single interface.

## When to delegate to OpenCode

**Default to OpenCode for:**
- Code writing, refactoring, and implementation tasks (especially with GPT-5.4/5.3-Codex)
- Code review and quality analysis
- Tasks benefiting from a model with different strengths than Gemini/Claude

**Model selection guide:**
- `--gpt5.4`: complex reasoning, code generation, architecture decisions (requires OpenAI quota)
- `--gpt5.3-codex`: code-focused tasks, refactors, detailed implementation (requires OpenAI quota)
- `--minimax-free`: lightweight tasks, summaries, quick Q&A — no API key needed (default)
- `--nemotron-super-free`: alternative free model; strong on structured tasks — no API key needed

## How to delegate

```bash
bash container/scripts/opencode-ask.sh [--model-flag] "task description"

# Model flags:
#   --minimax-free          opencode/minimax-m2.5-free (default — no API key)
#   --nemotron-super-free   opencode/nemotron-3-super-free (no API key)
#   --gpt5.4                openai/gpt-5.4 (requires OPENAI_API_KEY)
#   --gpt5.3-codex          openai/gpt-5.3-codex (requires OPENAI_API_KEY)
#   --model PROVIDER/MODEL  any model by explicit ID
#   --silence-timeout N     seconds of silence before declaring a hang (default 300)
```

## Exit codes from opencode-ask.sh
- `0` — success; response in stdout
- `1` — opencode exited with an error
- `3` — hung; OpenCode stopped producing output

## Notes
- OpenCode reads project files during execution — its response reflects the current repo state.
- Free models (MiniMax, Nemotron) have no API key requirement but may have rate limits.
- GPT models require `OPENAI_API_KEY` to be set in the environment or via `opencode providers`.
