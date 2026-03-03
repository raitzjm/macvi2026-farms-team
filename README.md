# MaCVi 2026 — FARMS Team

Codebase for the FARMS team competing in the [MaCVi 2026](https://macvi.org/workshop/cvpr) maritime computer vision challenge at CVPR 2026.

## Challenge Tracks

| Track | Folder | Status |
|-------|--------|--------|
| Embedded Semantic Segmentation | `challenges/embedded-semantic-segmentation/` | Setup |

### Embedded Semantic Segmentation
3-class maritime segmentation (sky / water / obstacle) on the [LaRS dataset](https://larss.science.si/).
Model is trained in PyTorch/mmsegmentation, exported to ONNX, and submitted to the challenge server which handles INT8 quantization (Qualcomm SNPE) and deployment (Luxonis RVC4).
Target: ≥30 FPS at 768×384, maximizing F1 detection score.

## Repository Structure

```
macvi2026-farms-team/
├── CLAUDE.md                          # Global rules and practices for AI agents
├── README.md                          # This file
├── .gitignore
├── container/
│   ├── Dockerfile                     # CUDA 12.1 base; ML env is conda macvi (Python 3.11)
│   └── scripts/
│       ├── build.sh                   # Build the container image
│       └── run.sh                     # Run the container
├── challenges/
│   └── embedded-semantic-segmentation/
│       ├── CLAUDE.md                  # Challenge-specific context for agents
│       ├── experiment-log.md          # Training experiment log (append-only)
│       └── mmsegmentation-macvi/      # Fork of lojzezust/mmsegmentation-macvi
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

# Full setup — agentic work with Claude Code
./container/scripts/run.sh \
  --data-dir /path/to/datasets \
  --ssh-dir ~/.ssh

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
| `--gitconfig FILE` | — | Git user config (name, email) |

The container mounts the repository at `/home/farms/macvi2026-farms-team/`.
Datasets are mounted at `/home/farms/data/` and are **read-only**.

### Agentic work (Claude Code)

The following mounts and environment variable are only needed when running
Claude Code inside the container for AI-assisted work. They are not required
for manually running training scripts.

| What | How | Why |
|------|-----|-----|
| `--claude-dir` | Mounts `~/.claude` into the container | Claude Code stores its settings, credentials, and persistent memory here. Without it, Claude Code cannot authenticate or retain context across sessions. |
| `--ssh-dir` | Mounts `~/.ssh` read-only | Needed for git operations over SSH (push, pull, clone). Without it, git remotes must use HTTPS with a token. |
| `--gitconfig` | Mounts `~/.gitconfig` read-only | Provides git author name and email so commits are attributed correctly. Already has a sensible default (`~/.gitconfig`). |
| `MACVI_GH_TOKEN` | Passed as `GH_TOKEN` env var | GitHub CLI (`gh`) uses this to manage issues, PRs, and comments. Required for any agent GitHub interaction. |

Set the token before running:

```bash
export MACVI_GH_TOKEN=<your_github_pat>
./container/scripts/run.sh --data-dir /path/to/datasets --ssh-dir ~/.ssh
```

## Workflow

- All agents follow the rules in `CLAUDE.md` (global) and the relevant `challenges/*/CLAUDE.md`.
- Experiments are logged in `challenges/embedded-semantic-segmentation/experiment-log.md`.
- Tasks are tracked via [GitHub Issues](https://github.com/raitzjm/macvi2026-farms-team/issues).
- Agent activity is always tagged `[Agent - <Model Name>]` in commits and issue comments.
