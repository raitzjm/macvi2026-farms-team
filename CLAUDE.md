# MaCVi 2026 - FARMS Team

## Project
Maritime computer vision challenge. Multiple challenge tracks under challenges/.
Repo: macvi2026-farms-team

## Architecture
```
macvi2026-farms-team/
├── CLAUDE.md              (this file - global rules)
├── README.md
├── container/
│   ├── Dockerfile
│   └── scripts/
│       ├── build.sh       (Docker default; --podman flag for Podman)
│       └── run.sh         (Docker default; --podman flag for Podman)
├── challenges/
│   └── embedded-semantic-segmentation/
│       ├── CLAUDE.md      (challenge-specific context)
│       ├── experiment-log.md
│       ├── mmsegmentation-macvi/  (copied from upstream, tracked directly)
│       ├── configs/
│       └── scripts/
└── docs/
```

## Safety Rules
- NEVER commit dataset files, model weights (.pth, .onnx), or credentials
- NEVER modify files under /home/farms/data/ (mounted read-only datasets)
- NEVER access repositories other than macvi2026-farms-team
- Before any destructive action (deleting files, force push), ask for confirmation

## Logging (CRITICAL)
- The experiment log (experiment-log.md) is for training experiments ONLY
- Before starting a training run, append a timestamped entry describing the
  experiment setup
- After completing a training run, update the entry with results and status
- Each entry must log: config path, dataset, resolution, epochs, final
  metrics (val F1, val mIoU), and git commit hash
- Non-experiment work (installations, refactors, scripts) is tracked through
  git commits only — do not clutter the experiment log with these

## Git Workflow
- Work on main for routine changes
- Branch only when needed for isolation (risky refactors, parallel work)
- Branch naming when used: {type}/{short-description}
  (e.g. exp/baseline-deeplabv3, fix/onnx-export, feat/4class-training)
  Branches should be created from a GitHub issue when one exists.
- Commit with descriptive messages referencing experiment IDs when applicable
- ALL agent commits must start with [agent] tag in the subject line
  (e.g. "[agent] install Python 3.12 and pip build tools")
- In the extended commit body, include the model name that performed the work
  (e.g. "Claude Sonnet 4.6 Agent", "Claude Opus 4.6 Agent")
- Push and open PR for teammate review before merging branches

## GitHub Issues Workflow
- Use GitHub Issues to track tasks, record decisions, and hand off context between agents
- Every issue or comment posted by an agent must begin with:
  `[Agent - <Model Name>]` (e.g. `[Agent - Claude Sonnet 4.6]`)
  so it is always clear when an agent acted using the owner's credentials
- Include a brief rationale for decisions taken directly in the relevant issue
- Sub-issues, milestones, and sprint-style batching are allowed when appropriate
- Close issues (or leave a closing comment) when work is complete
- The GitHub Project board is used for visual task tracking; add issues to it

## Container Development Practice
When installing new system or Python dependencies:
1. Install and test inside the running container first
2. Verify it works (import, run a quick test)
3. Only then update the Dockerfile (or environment file) to make it permanent
4. Group related dependencies into logical commits
5. Notify the user to rebuild the container after Dockerfile changes

## Environment
- Base image: nvidia/cuda:12.1.1-devel-ubuntu22.04
- Python: 3.12
- Datasets mounted read-only at /home/farms/data/ (pass --data-dir to run.sh)
- Training outputs go to challenges/{challenge}/outputs/
- See challenge-specific CLAUDE.md files for task details
- **Agents always run INSIDE the container** — never on the local host system.
  Do not attempt to detect or interact with the host environment.

## Container Scripts
- `container/scripts/build.sh` — builds the image; Docker by default, `--podman` to use Podman
- `container/scripts/run.sh`   — runs the container; Docker by default, `--podman` to use Podman
  Key flags: `--data-dir`, `--ssh-dir`, `--claude-dir`, `--gitconfig`

## Updating Practices
When the user asks to change a workflow, convention, or practice:
1. First ask: "Is this a permanent change to our practices, or just for this one time?"
2. If permanent, update the relevant CLAUDE.md file(s) to reflect the new practice
3. If one-time, proceed without modifying CLAUDE.md

## Communication
- When you need a decision (strategic choice, phase transition, new model),
  ask clearly and wait for a response
- When a task is complete, summarize what was done and what the next step is
- If something fails, report the error and your diagnosis before attempting fixes
