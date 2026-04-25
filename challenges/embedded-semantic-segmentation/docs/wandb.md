# WandB Experiment Tracking

Full WandB configuration for the MaCVi embedded semantic segmentation challenge.
For quick setup, use the `/setup-wandb` skill.

## Package and Key

- **Package**: `wandb==0.25.0` (pinned — required for `wandb_` key prefix patch)
- **API key**: `~/data/wandb_key.txt` (read-only mount — never commit)
- Load via: `source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh`
  (or the equivalent `setup_wandb.sh` in phase2/phase2_5/phase3/scripts/)

## Project Settings

- **Entity**: `little_farms`
- **Project**: `embedded_segmentation_macvi2026`
- **Run naming**: `{phase}_{model_name}` (e.g. `phase3_pidnets_exp020`)

## Gotchas

- `WANDB_DIR` must NOT be on `PYTHONPATH` — the run scripts handle this correctly.
- Failed runs: delete from WandB UI before relaunching to keep the project clean.
- The `wandb_` prefix patch in `wandb/sdk/lib/wbauth/validation.py` must be applied
  after any WandB reinstall. See `docs/environment.md` for the patch details.

## Phase 2 Prediction Image Logging

Phase 2 uses `WandbVisBackend` + `SegVisualizationHook` for prediction overlay images.
Requires `WANDB_API_KEY` to be set AND the `wandb_` prefix patch applied.
Images appear in the WandB "Media" tab per run at every val interval (every 250 iters).

## Phase 2.5 Visualizer Extension

Phase 2.5 adds `MacViSegVisualizer` (subclass of `SegLocalVisualizer`) that marks
GT==255 (ignore) pixels grey in prediction overlays, matching how GT overlays show them.
