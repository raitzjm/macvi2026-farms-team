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
│   ├── Dockerfile                     # CUDA 12.1 + Python 3.12 dev image
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
# Minimal (no data, no SSH)
./container/scripts/run.sh

# With datasets and SSH key forwarding
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
| `--ssh-dir DIR` | — | SSH directory; mounted read-only (for git over SSH) |
| `--claude-dir DIR` | `~/.claude` | Claude Code config directory |
| `--gitconfig FILE` | `~/.gitconfig` | Git config file |

The container mounts the repository at `/home/farms/macvi2026-farms-team/`.
Datasets are mounted at `/home/farms/data/` and are **read-only**.

### Required environment variable

```bash
export MACVI_GH_TOKEN=<your_github_pat>
```

The token is passed into the container as `GH_TOKEN` for the GitHub CLI.

## Workflow

- All agents follow the rules in `CLAUDE.md` (global) and the relevant `challenges/*/CLAUDE.md`.
- Experiments are logged in `challenges/embedded-semantic-segmentation/experiment-log.md`.
- Tasks are tracked via [GitHub Issues](https://github.com/raitzjm/macvi2026-farms-team/issues).
- Agent activity is always tagged `[Agent - <Model Name>]` in commits and issue comments.
