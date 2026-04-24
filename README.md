# MaCVi 2026 — FARMS Team

Codebase for the FARMS team competing in the [MaCVi 2026](https://macvi.org/workshop/cvpr) maritime computer vision challenge at CVPR 2026.

## Challenge Tracks

| Track | Folder | Status |
|-------|--------|--------|
| Embedded Semantic Segmentation | `challenges/embedded-semantic-segmentation/` | Phase 2 (active) |

### Embedded Semantic Segmentation
3-class maritime segmentation (sky / water / obstacle) on the [LaRS dataset](https://larss.science.si/).
Model is trained in PyTorch/mmsegmentation, exported to ONNX, and submitted to the challenge server which handles INT8 quantization (Qualcomm SNPE) and deployment (Luxonis RVC4).
Target: ≥30 FPS at 768×384, maximizing F1 detection score.

## Repository Structure

```
macvi2026-farms-team/
├── AGENTS.md                          # Global rules and practices for AI agents (canonical)
├── CLAUDE.md                          # Claude Code entrypoint; must import AGENTS.md
├── README.md                          # This file
├── .gitignore
├── .claude/skills/                    # Agent skills (setup-wandb, launch-training, etc.)
├── container/
│   ├── Dockerfile                     # CUDA 12.1 base; ML envs: macvi (Phase 1), macvi2 (Phase 2)
│   └── scripts/
│       ├── build.sh                   # Build the container image
│       └── run.sh                     # Run the container
├── challenges/
│   ├── common/lars-analysis/          # LaRS dataset analysis scripts
│   └── embedded-semantic-segmentation/
│       ├── AGENTS.md                  # Challenge-specific context for agents (canonical)
│       ├── CLAUDE.md                  # Canonical one-line stub importing AGENTS.md
│       ├── experiment-log.md          # Training experiment log (append-only)
│       ├── eval_metrics.md            # Phase 2 evaluation results
│       ├── submissions.md             # Challenge submission tracker
│       ├── reference_baselines.md     # Published baselines from challenge website
│       ├── mmsegmentation-macvi/      # Fork of lojzezust/mmsegmentation-macvi (Phase 1)
│       ├── phase1/                    # Phase 1 configs, scripts, report, artifacts
│       └── phase2/                    # Phase 2 configs, scripts, datasets, artifacts
└── docs/
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

# Full setup — agentic work with Claude Code, Codex, or Gemini
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh \
  --codex-dir ~/.codex

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
| `--codex-dir DIR` | — | Codex CLI config directory; mounted at `/home/farms/.codex/` |
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
| `--codex-dir` | Mounts `~/.codex` into the container | Required for Codex CLI authentication, local settings, and session persistence across container recreations. |
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
load `GEMINI.md` and `AGENTS.md` files from the project tree at session start,
so project rules and context are always in scope without passing them in every prompt.
`AGENTS.md` is the canonical context file for all three agents (Claude via a
`CLAUDE.md` stub that imports it, Codex natively, Gemini via the setting above).

To authenticate Gemini CLI, mount your host `~/.gemini` directory:

```bash
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh \
  --gemini-dir ~/.gemini
```

### Codex CLI integration

OpenAI Codex CLI is installed in the container and can be used as a sub-agent
via `container/scripts/codex-ask.sh`.

Codex loads `AGENTS.md` natively as the project instruction file. `CLAUDE.md`
is kept as a stub for Claude Code compatibility.

To authenticate Codex CLI and persist sessions, mount your host `~/.codex` directory:

```bash
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh \
  --codex-dir ~/.codex
```

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

- ML conda environments: `macvi` (Phase 1, mmseg 0.30.0) and `macvi2` (Phase 2, mmseg 1.2.2)
- All agents follow the rules in `AGENTS.md` (global) and the relevant `challenges/*/AGENTS.md`.
- Root `CLAUDE.md` may add Claude-specific instructions, but it must include `@AGENTS.md`; child-directory `CLAUDE.md` files must stay as the canonical one-line stub `@AGENTS.md`.
- Experiments are logged in `challenges/embedded-semantic-segmentation/experiment-log.md`.
- Tasks are tracked via [GitHub Issues](https://github.com/raitzjm/macvi2026-farms-team/issues).
- Agent activity is always tagged `[Agent - <Model Name>]` in commits and issue comments.
