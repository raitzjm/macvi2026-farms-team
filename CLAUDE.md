# MaCVi 2026 - FARMS Team

## Project
Maritime computer vision challenge. Multiple challenge tracks under challenges/.
Repo: macvi2026-farms-team

## Safety Rules
- NEVER commit dataset files or credentials
- NEVER commit model weights (.pth, .onnx) **except** where explicitly documented
  in a challenge-specific CLAUDE.md (e.g. curated experiment artifacts committed via Git LFS)
- NEVER modify files under /home/farms/data/ (mounted read-only datasets)
- NEVER access repositories other than macvi2026-farms-team
- Before any destructive action (deleting files, force push), ask for confirmation

## Secret Scanning
A gitleaks pre-commit hook runs automatically on every commit inside the container
(baked into the image via system-level git config — no manual setup needed).
If the hook blocks a commit:
1. Remove the secret from the file
2. Replace it with a placeholder (e.g. `# removed — use $ENV_VAR_NAME instead`)
3. Notify the user
4. NEVER use `--no-verify` to bypass this check

## Logging (CRITICAL)
- The experiment log (`challenges/embedded-semantic-segmentation/experiment-log.md`) is for training experiments ONLY
- Before starting a training run, append a timestamped entry describing the
  experiment setup
- After completing a training run, update the entry with results and status
- Each entry must log: config path, dataset, resolution, epochs, final
  metrics (val F1, val mIoU), and git commit hash
  (val F1 is reported via post-export eval when the mmseg evaluator does not support it;
  in that case record it in the phase report and note 'see phase report' in the experiment
  log entry)
- Non-experiment work (installations, refactors, scripts) is tracked through
  git commits only — do not clutter the experiment log with these

## Git Workflow
- Work on main for routine changes
- Branch only when needed for isolation (risky refactors, parallel work)
- Branch naming when used: {type}/{short-description}
  (e.g. exp/baseline-deeplabv3, fix/onnx-export, feat/4class-training)
- **Always create branches linked to their GitHub issue** using:
  `gh issue develop <issue-number> --name <branch-name> --checkout`
  This registers a development link so GitHub auto-closes the issue when the PR merges.
  Never use plain `git checkout -b` for issue-tracked work.
- Commit with descriptive messages referencing experiment IDs when applicable
- ALL agent commits must start with [agent] tag in the subject line
  (e.g. "[agent] install Python 3.12 and pip build tools")
- In the extended commit body, include the model name that performed the work
  (e.g. "Claude Sonnet 4.6 Agent", "Claude Opus 4.6 Agent")
- Push and open PR for teammate review before merging branches
- Every PR that resolves an issue must include `Closes #N` in the PR body
  as a fallback in case the development link was not set at branch creation

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
- **All agent and user sessions run inside the container — no exceptions.**
  Never assume you are on the host. If something seems wrong with the environment,
  diagnose within the container rather than assuming a host context.
- Base image: nvidia/cuda:12.1.1-devel-ubuntu22.04
- System Python: Ubuntu 22.04 default (not used for ML work)
- ML training env: conda env `macvi`, Python 3.11 (see challenge CLAUDE.md for details)
- Datasets mounted read-only at /home/farms/data/ (pass --data-dir to run.sh)
- Training outputs go to challenges/{challenge}/outputs/
- See challenge-specific CLAUDE.md files for task details

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
