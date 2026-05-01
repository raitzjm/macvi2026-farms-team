# FathomNet-CLEF 2026 — Run Tracking Policy

## Purpose

This policy defines how FathomNet training runs, inference runs, checkpoints, submission CSVs, and log artifacts are tracked.

Agents working on any of the following must read and follow this document before making changes:

- training
- inference
- Kaggle kernels
- TensorDock training/inference workflows
- logging or backfilling training results
- logging or backfilling inference results
- restructuring `artifacts/`, `training-log.md`, or `inference-log.md`

## Core Principles

- Keep one canonical config or script path per model family.
- Do not copy a new config file per experiment unless the model family truly forks.
- Every tracked run must record the repo commit SHA plus a resolved config snapshot of the final effective parameters.
- If a config change defines a new tracked experiment, isolate that config change in its own commit before launching the run.
- Human-readable journals live in `training-log.md` and `inference-log.md`.
- Machine-readable run indexes live in per-run `metadata.json` files.

## Artifact Layout

Durable per-run bundles:

- `artifacts/training-runs/T-NNN/`
- `artifacts/inference-runs/I-NNN/`

Heavy payloads stay separate:

- selected checkpoints under `artifacts/checkpoints/T-NNN-.../`
- submission CSVs under `artifacts/submissions/I-NNN-.../`

## Training Run Bundle Contract

Every new tracked training run must create `artifacts/training-runs/T-NNN/` with:

- `metadata.json`
- raw `stdout.log` when available
- resolved config snapshot: `resolved-config.json`, `resolved-config.yaml`, or `resolved-config.py`
- native metrics artifact: `results.csv`, `metrics.jsonl`, or equivalent

Optional but valid extras:

- `framework.log`
- `kaggle-console.json`
- framework-native side metadata

The selected checkpoint is not duplicated into the run bundle. `metadata.json` points to the durable checkpoint path under `artifacts/checkpoints/`.

## Inference Run Bundle Contract

Every new tracked inference run must create `artifacts/inference-runs/I-NNN/` with:

- `metadata.json`
- resolved inference config snapshot, typically `resolved-config.json`
- raw `stdout.log` when available

The submission CSV is not duplicated into the run bundle. `metadata.json` points to the durable CSV path under `artifacts/submissions/`.

## `metadata.json` Contract

`metadata.json` is the machine-readable index for a run bundle. It should identify:

- `kind`: training or inference
- `run_id`
- `track`
- `platform`
- `config_source`
- `git_commit` when known
- `resolved_config`
- `stdout_log` when available
- payload pointer:
  - training: selected checkpoint and metrics artifact
  - inference: submission CSV and checkpoint used
- `source_train_id` for inference runs when applicable

Historical backfills may use `null` for fields that were not preserved originally, but they must say so explicitly with a note field instead of inventing values.

## Logging Policy

- `training-log.md` is the human-readable training journal.
- `inference-log.md` is the human-readable inference and submission journal.
- Every new tracked run must be reflected in the relevant markdown log and in its run bundle.
- The markdown logs may evolve, but their headers must note format changes and effective dates.

## Naming Policy

- FathomNet WandB training runs must log under entity `little_farms`, project `fathomnet2026`.
- Tracked training run names must append the training run ID suffix, for example `yolov11l-baseline-T-004`.
- Tracked inference CSV filenames must append the inference run ID, and the source training run ID when known, for example `submission-I-009-from-T-004.csv`.
- If a training launch fails after creating a WandB run, delete that failed WandB run before retrying.

## Kaggle Policy

Kaggle training scripts must emit durable top-level output files in `/kaggle/working/` so they can be recovered with `kaggle kernels output`.

Required pattern for new tracked Kaggle runs:

- tee console output to `train.log`
- write `metadata.json`
- write resolved config snapshot (`resolved-config.*`)
- write native metrics artifact when available
- write the selected checkpoint and any submission CSV to the expected durable locations

When historical Kaggle raw logs need to be recovered and no explicit `train.log` was written, use:

- `kaggle kernels logs <owner>/<slug>`

This is recovery-only. New kernels should write `train.log` directly.

## TensorDock Policy

- Keep the local workspace and the TensorDock clone synchronized with explicit pull/push steps.
- Before remote training or inference, verify the TensorDock clone is on the intended commit.
- After remote work is committed and pushed, pull it locally before continuing.
- TensorDock dataset roots must be supplied through `FATHOMNET_DATA_DIR`.
- Remote TensorDock launches that should use WandB must explicitly export `WANDB_API_KEY` in the launch command path; non-interactive SSH shells may not inherit late `~/.bashrc` exports.

## LFS Policy

The following are Git LFS tracked for FathomNet:

- curated checkpoints under `artifacts/checkpoints/`
- curated submission CSVs under `artifacts/submissions/`

Run bundles themselves are regular git content unless a future artifact added to them is large enough to justify LFS.

## Historical Backfill Policy

Backfills should preserve provenance without pretending to know missing facts.

Required behavior:

- point to the real durable payloads that exist today
- recover Kaggle console logs when possible
- set missing raw logs or commit SHAs to `null` when they were not preserved
- add note fields explaining why data is missing
