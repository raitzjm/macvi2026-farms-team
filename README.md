# Challenge Arena

Multi-challenge ML workspace for competitive ML benchmarks and workshops.
Each challenge lives under `challenges/{workshop-challenge}/` and manages its own conda environment via bootstrap scripts.

> **MaCVi 2026 (FARMS Team)** — the original repo. Challenge tree is grandfathered at `challenges/embedded-semantic-segmentation/` (migration to the new convention tracked in issue #69).

## Challenge Tracks

| Track | Folder | Status |
|-------|--------|--------|
| MaCVi 2026 — Embedded Semantic Segmentation | `challenges/embedded-semantic-segmentation/` | **COMPLETE** (CVPR 2026) |

### MaCVi 2026 — Embedded Semantic Segmentation
3-class maritime segmentation (sky / water / obstacle) on the [LaRS dataset](https://larss.science.si/).
Model trained in PyTorch/mmsegmentation, exported to ONNX, submitted to challenge server (INT8 quantization via Qualcomm SNPE, deployment via Luxonis RVC4).
Best result: EXP-020 PIDNet-S, Q=52.1 — top 3 among ≥30 FPS entries.

## Repository Structure

```
macvi2026-farms-team/
├── CLAUDE.md                          # Global agent instructions (canonical)
├── README.md                          # This file
├── .gitignore
├── .claude/skills/                    # Agent skills (setup-wandb, launch-training, etc.)
├── container/
│   ├── Dockerfile                     # Vanilla Ubuntu 24.04; no baked envs (lean image)
│   └── scripts/
│       ├── build.sh                   # Build the container image
│       ├── run.sh                     # Run the container
│       ├── gemini-ask.sh              # Delegate tasks to Gemini as a sub-agent
│       └── opencode-ask.sh            # Delegate tasks to OpenCode (GPT, MiniMax, Nemotron)
├── challenges/
│   └── embedded-semantic-segmentation/   # Grandfathered MaCVi 2026 challenge tree
│       ├── CLAUDE.md                  # Challenge-specific agent instructions
│       ├── STATE.md                   # Challenge status tracker
│       ├── bootstrap-macvi.sh         # Create macvi conda env (Phase 1)
│       ├── bootstrap-macvi2.sh        # Create macvi2 conda env (Phase 2+)
│       ├── docs/                      # Reference docs (dataset, env, preprocessing, etc.)
│       ├── experiment-log.md          # Training experiment log (append-only)
│       ├── eval_metrics.md            # Phase 2+ evaluation results
│       ├── submissions.md             # Challenge submission tracker
│       ├── phase1/                    # Phase 1 configs, scripts, report, artifacts
│       ├── phase2/                    # Phase 2 configs, scripts, PHASE2_NOTES.md
│       └── phase3/                    # Phase 3 configs, scripts, phase3_report.md
└── docs/
    ├── multi-challenge-convention.md  # Multi-challenge workspace conventions
    └── agent-reference/               # Delegation docs for Gemini, OpenCode
```

## Container Setup

The development environment runs inside a Docker (or Podman) container.
All training and experimentation is done inside the container.

### Build

```bash
# Docker (default)
./container/scripts/build.sh

# Podman
./container/scripts/build.sh --podman
```

### Run

```bash
# Minimal — manual training work, no agentic features
./container/scripts/run.sh --data-dir /path/to/datasets

# Full setup — agentic work with Claude Code, Gemini, or OpenCode
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh \
  --claude-dir ~/.claude \
  --gemini-dir ~/.gemini \
  --opencode-dir ~/.config/opencode

# Podman
./container/scripts/run.sh --podman --data-dir /path/to/datasets
```

**Run options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--podman` | — | Use Podman instead of Docker |
| `--data-dir DIR` | — | Dataset root; mounted read-only at `/home/farms/data/` |
| `--ssh-dir DIR` | — | SSH directory; mounted read-only at `/home/farms/.ssh/` |
| `--claude-dir DIR` | — | Claude Code config and memory directory |
| `--gemini-dir DIR` | — | Gemini CLI config directory; mounted at `/home/farms/.gemini/` |
| `--opencode-dir DIR` | — | OpenCode config directory; mounted at `/home/farms/.config/opencode/` |
| `--gitconfig FILE` | — | Git user config (name, email) |

The container mounts the repository at `/home/farms/macvi2026-farms-team/`.
Datasets are mounted at `/home/farms/data/` and are **read-only**.

### Agentic work

The following mounts and environment variable are only needed when running
AI assistants inside the container for assisted development work. They are not
required for manually running training scripts.

| What | How | Why |
|------|-----|-----|
| `--claude-dir` | Mounts `~/.claude` into the container | Claude Code stores its settings, credentials, and persistent memory here. Without it, Claude Code cannot authenticate or retain context across sessions. |
| `--ssh-dir` | Mounts `~/.ssh` read-only | Needed for git operations over SSH (push, pull, clone). Without it, git remotes must use HTTPS with a token. |
| `--gitconfig` | Mounts `~/.gitconfig` read-only | Provides git author name and email so commits are attributed correctly. Already has a sensible default (`~/.gitconfig`). |
| `--gemini-dir` | Mounts `~/.gemini` into the container | Required for Gemini CLI authentication and to persist Gemini config across sessions. |
| `--opencode-dir` | Mounts `~/.config/opencode` into the container | Persists OpenCode provider credentials and config across sessions. |
| `MACVI_GH_TOKEN` | Passed as `GH_TOKEN` env var | GitHub CLI (`gh`) uses this to manage issues, PRs, and comments. Required for any agent GitHub interaction. |

Set the token before running:

```bash
export MACVI_GH_TOKEN=<your_github_pat>
./container/scripts/run.sh --data-dir /path/to/datasets --ssh-dir ~/.ssh
```

### Gemini CLI integration

Gemini CLI is installed in the container and can be used as a sub-agent via
`container/scripts/gemini-ask.sh`.

**Context loading:** `.gemini/settings.json` configures Gemini CLI to automatically
load `GEMINI.md` and `CLAUDE.md` files from the project tree at session start,
so project rules and context are in scope for interactive sessions.

To authenticate Gemini CLI, mount your host `~/.gemini` directory:

```bash
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh \
  --gemini-dir ~/.gemini
```

### OpenCode integration

OpenCode is installed in the container and can be used as a multi-model sub-agent via
`container/scripts/opencode-ask.sh`. It supports GPT, MiniMax, Nemotron, and other providers.

```bash
# Default (MiniMax free, no API key needed)
bash container/scripts/opencode-ask.sh "your prompt"

# Named model shortcuts
bash container/scripts/opencode-ask.sh --minimax-free "prompt"
bash container/scripts/opencode-ask.sh --nemotron-super-free "prompt"
bash container/scripts/opencode-ask.sh --gpt5.4 "prompt"
bash container/scripts/opencode-ask.sh --gpt5.3-codex "prompt"

# Any model explicitly
bash container/scripts/opencode-ask.sh --model openai/gpt-5 "prompt"
```

To persist OpenCode credentials (sign in once, reuse across sessions), mount your
host `~/.config/opencode` directory:

```bash
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --opencode-dir ~/.config/opencode
```

Sign in interactively with `opencode` and authenticate each provider via
`opencode providers`. The credentials are then available in all future sessions.

## Model Weights (Git LFS)

Selected Phase 2 checkpoints are committed via Git LFS under
`challenges/embedded-semantic-segmentation/phase2/artifacts/`.
Only the best-checkpoint-per-model is committed; the full training outputs live
locally and are not tracked.

### Phase 2 — Committed checkpoints

Best checkpoints for INT8-stable models are committed via Git LFS under
`challenges/embedded-semantic-segmentation/phase2/artifacts/`.

See [`eval_metrics.md`](challenges/embedded-semantic-segmentation/eval_metrics.md) for the
full results table including INT8 impact analysis.

To use a checkpoint, run the export script:
```bash
python challenges/embedded-semantic-segmentation/phase2/scripts/export_and_eval.py \
  --models pidnets   # or stdc2, bisenetv2
```

## Workflow

- Each challenge has its own bootstrap script(s) — run once to create the conda env inside the container.
- ML envs for MaCVi 2026: `macvi` (Phase 1, mmseg 0.30.0) and `macvi2` (Phase 2+, mmseg 1.2.2).
- All agents follow `CLAUDE.md` (global) and the relevant `challenges/*/CLAUDE.md`.
- Experiments are logged in `challenges/embedded-semantic-segmentation/experiment-log.md`.
- Tasks are tracked via [GitHub Issues](https://github.com/raitzjm/macvi2026-farms-team/issues).
- Agent activity is always tagged `[Agent - <Model Name>]` in commits and issue comments.
