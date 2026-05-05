# Challenge Arena

## Project
Multi-challenge ML workspace. Each challenge lives at `challenges/{workshop-challenge}/`.
Current challenges: see `README.md` (Challenge Tracks table) for the list of challenge folders.

## Multi-challenge Convention

- Each challenge lives at `challenges/{workshop-challenge}/` (flat, e.g. `challenges/fathomnet-fungiclef/`).
- No year subfolders — year snapshots are pinned via annotated git tags (e.g. `macvi-embedded-semantic-segmentation-2026`).
- Each challenge has a minimal `STATE.md` tracking `current_year` + `status: active|finished`. Exception: `challenges/fathomnet-clef/` intentionally keeps status in its AGENTS.md.
- Each challenge has its own `bootstrap-*.sh` env script(s); envs are NOT baked into the Docker image.
- Grandfathered: the MaCVi 2026 tree lives at `challenges/embedded-semantic-segmentation/` (migration tracked in issue #69).

Full details: `docs/multi-challenge-convention.md`.

## Safety Rules
- NEVER commit dataset files or credentials
- NEVER commit model weights (.pth, .onnx) **except** where explicitly documented
  in a challenge-specific AGENTS.md (e.g. curated experiment artifacts committed via Git LFS)
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

## Git Workflow
- Work on main for routine changes; branch only for isolation (risky refactors, parallel work)
- Branch naming: `{type}/{short-description}` (e.g. `exp/baseline-deeplabv3`, `fix/onnx-export`)
- For issue-tracked branches: `gh issue develop <issue-number> --name <branch-name> --checkout`
  (never use plain `git checkout -b` — the `gh issue develop` form registers the dev link)
- For grouped experiment families that share one decision thread (for example size comparisons or k-fold sweeps), prefer a single branch and a single tracking issue/sub-issue for the whole family; only split branches/issues when the workflow is materially separate (for example a distinct Co-DETR pseudo-labeling track)
- All agent commits: subject starts with `[agent]`; body includes model name (e.g. `Claude Sonnet 4.6 Agent`)
- Push and open PR for teammate review before merging branches
- Every PR resolving an issue must include `Closes #N` in the PR body

## GitHub Issues Workflow
- Every issue or comment posted by an agent must begin with `[Agent - <Model Name>]`
  (e.g. `[Agent - Claude Sonnet 4.6]`) so it is always clear when an agent acted
  using the owner's credentials
- Close issues (or leave a closing comment) when work is complete; use `Closes #N` in PR body

## Container Development Practice
When installing new system or Python dependencies:
1. Install and test inside the running container first
2. Verify it works (import, run a quick test)
3. Only then update the Dockerfile (or environment file) to make it permanent
4. Group related dependencies into logical commits
5. Notify the user to rebuild the container after Dockerfile changes

## Environment
All agent and user sessions run inside the container — no exceptions. Never assume
you are on the host; diagnose within the container if something seems wrong.

Base: vanilla Ubuntu 24.04 — no CUDA in the base image (see `container/Dockerfile`).
Each challenge's env manages its own CUDA, via conda-forge (`cuda-nvcc`,
`cudatoolkit`, `cudnn`) or via prebuilt wheels that bundle the runtime (e.g.
PyTorch `cuXYZ` wheels). The NVIDIA driver is injected at runtime by
`nvidia-container-toolkit`.

ML envs are per-challenge and created by bootstrap scripts in each challenge's
folder — NOT baked into the image (lean image, faster rebuilds).
Datasets read-only at `/home/farms/data/`. Training outputs at `challenges/{challenge}/outputs/`.
See each challenge's AGENTS.md for its specific env bootstrap.

## Container Scripts
- `container/scripts/build.sh` — builds the image; Docker by default, `--podman` to use Podman
- `container/scripts/run.sh`   — runs the container; Docker by default, `--podman` to use Podman
  Key flags: `--data-dir`, `--ssh-dir`, `--claude-dir`, `--claude-json`, `--gemini-dir`, `--codex-dir`, `--opencode-dir`, `--gitconfig`
- `container/scripts/codex-ask.sh` — delegate tasks to OpenAI Codex CLI as a sub-agent
- `container/scripts/gemini-ask.sh` — delegate read-only tasks to Gemini as a sub-agent
- `container/scripts/opencode-ask.sh` — delegate tasks to OpenCode (multi-model: MiniMax, GPT, Nemotron)

## Updating Practices
When the user asks to change a workflow, convention, or practice:
1. First ask: "Is this a permanent change to our practices, or just for this one time?"
2. If permanent, update the relevant AGENTS.md file(s) to reflect the new practice
3. If one-time, proceed without modifying AGENTS.md

## Local Planning Files
- `docs/superpowers/` is local-only working scratch space for plans/specs during a live session
- NEVER commit `docs/superpowers/`
- Clean up finished task-list / plan files when the work is done; remove the directory too if it becomes empty

## Agent Delegation
See `docs/agent-reference/gemini-delegation.md` for how to delegate tasks to Gemini
(via `container/scripts/gemini-ask.sh`) and the Haiku-passthrough pattern for
file-modifying work.
See `docs/agent-reference/opencode-delegation.md` for OpenCode delegation patterns
(via `container/scripts/opencode-ask.sh`).
See `README.md` for Codex CLI integration and `container/scripts/codex-ask.sh`.

## Communication
- When you need a decision (strategic choice, phase transition, new model), ask clearly and wait
- When a task is complete, summarize what was done and what the next step is
- If something fails, report the error and your diagnosis before attempting fixes
