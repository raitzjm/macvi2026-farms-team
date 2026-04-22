# Multi-Challenge Convention

Conventions for organizing multiple ML challenge tracks in this repo.
See also: root `CLAUDE.md` for the short summary.

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

Nothing else in STATE.md. All other detail lives in challenge CLAUDE.md or docs/.

## Env Bootstrap Pattern

Each challenge manages its own conda environment(s) via `bootstrap-*.sh` scripts
in the challenge folder. Environments are:

- Named per-challenge (e.g. `macvi`, `macvi2`, `fathomnet`)
- Created at `/home/farms/miniforge3/envs/<name>/`
- Ephemeral per container — run the bootstrap script once after each fresh container
- NOT baked into the Docker image (lean image, faster rebuilds)

Bootstrap scripts are committed to the repo. If placed under `challenges/`, they
are gitignored-except (tracked) — no special .gitignore entry needed.

## Grandfathered: MaCVi 2026

The MaCVi 2026 embedded semantic segmentation challenge predates this convention
and lives at `challenges/embedded-semantic-segmentation/` (not yet renamed to
`challenges/macvi-embedded-semantic-segmentation/`). Migration is tracked in
issue #69. The 2026 submission state is tagged as
`macvi-embedded-semantic-segmentation-2026`.

## Link back

Root CLAUDE.md: `/CLAUDE.md` — contains the short summary of this convention.
