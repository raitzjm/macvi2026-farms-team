# Multi-Challenge Convention

Conventions for organizing multiple ML challenge tracks in this repo.
See also: root `AGENTS.md` for the short summary.

## Layout

Each challenge lives at `challenges/{workshop-challenge}/` — a flat compound
name combining the workshop and challenge track. Examples:

- `challenges/macvi-embedded-semantic-segmentation/`
- `challenges/fathomnet-fungiclef/`

No separate workshop-level subfolder (e.g. no `challenges/macvi/embedded/`).
If we ever run 3+ tracks in the same workshop we can revisit — YAGNI until then.

## Year Handling

No year subfolders inside a challenge directory. A challenge is a long-running
project; annual submissions are milestones, not separate events.

Each year's submission state is pinned via an annotated git tag:

```
git tag -a <challenge-name>-<year> <commit-sha> -m "<submission summary>"
```

Example: `macvi-embedded-semantic-segmentation-2026`

The tag message should include the final score and any key notes about the
submission. To list all challenge tags: `git tag | grep -E '^[a-z].*-[0-9]{4}$'`

## STATE.md Format

Every challenge folder must contain a minimal `STATE.md` with exactly two keys:

```
# <Challenge Name> — State

current_year: <YYYY>
status: active|finished
```

- `active` — currently training/submitting for the current year.
- `finished` — submission cycle for the current year is complete.

Nothing else in STATE.md. All other detail lives in challenge AGENTS.md or docs/.

Exception: `challenges/fathomnet-clef/` intentionally keeps status in `AGENTS.md`.

## Env Bootstrap Pattern

Each challenge manages its own conda environment(s) via `bootstrap-*.sh` scripts
in the challenge folder. Environments are:

- Created inside the challenge directory at `challenges/{challenge}/envs/<name>/`
  using `conda create -p <path>` (path-based, not named)
- **Persistent** — the challenge directory is part of the mounted repo volume on the
  host, so the env survives container recreations. Run the bootstrap script once only.
- Gitignored via `challenges/*/envs/` — never committed (too large, reproducible)
- NOT baked into the Docker image (lean image, faster rebuilds)

Bootstrap scripts are committed to the repo and tracked normally — no special
`.gitignore` handling needed for the scripts themselves, only for `envs/`.

## CUDA is env-scoped, not base-image-scoped

The base image is vanilla Ubuntu 24.04 — no system CUDA, no `nvcc`, no
`/usr/local/cuda`. Each env is responsible for the CUDA stack it needs:

- **Prebuilt wheels (common):** PyTorch `cuXYZ` wheels, mmcv prebuilt wheels,
  `onnxruntime-gpu`, etc. bundle the CUDA runtime libs they need. The env just
  pip-installs them — nothing else required.
- **Source builds:** If an env needs to compile CUDA code (e.g. custom ops),
  install `cuda-nvcc`, `cudatoolkit`, `cudnn` from conda-forge into the env.
  Do this in the bootstrap script, pinned to the CUDA version your framework
  expects.

Two envs in the same container can use different CUDA versions — they don't
interact. The NVIDIA driver is injected at runtime by `nvidia-container-toolkit`
from the host, independent of whatever CUDA userspace the env ships.

Why this matters: when we add a new challenge, we don't touch the Dockerfile
to change the CUDA version. The bootstrap script is the single source of truth
for the challenge's CUDA stack.

## Grandfathered: MaCVi 2026

The MaCVi 2026 embedded semantic segmentation challenge predates this convention
and lives at `challenges/embedded-semantic-segmentation/` (not yet renamed to
`challenges/macvi-embedded-semantic-segmentation/`). Migration is tracked in
issue #69. The 2026 submission state is tagged as
`macvi-embedded-semantic-segmentation-2026`.

## Link back

Root AGENTS.md: `/AGENTS.md` — contains the short summary of this convention.

Root `CLAUDE.md` is the Claude compatibility wrapper for `AGENTS.md`. Only repo-root `CLAUDE.md` may receive Claude-specific additions.
