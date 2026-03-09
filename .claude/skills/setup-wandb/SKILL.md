# Skill: Setup WandB

## When to use
Before any training run that needs WandB logging (all Phase 2 runs).

## Procedure

1. **Load API key:**
   ```bash
   source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
   ```
   Or inline: `export WANDB_API_KEY=$(tr -d '[:space:]' < ~/data/wandb_key.txt)`

2. **Verify the validation.py patch** (required for `wandb_` prefix API keys):
   Check that `wandb/sdk/lib/wbauth/validation.py` has the early-return for `wandb_` prefix
   before the 40-char length check. If not, patch it:
   ```python
   # Add before the length check:
   if api_key.startswith("wandb_"):
       return True
   ```
   This patch must be reapplied after any `pip install wandb`.

3. **Set WANDB_DIR** (if not handled by run script):
   ```bash
   export WANDB_DIR=challenges/embedded-semantic-segmentation/outputs/phase2
   ```
   CRITICAL: WANDB_DIR must NOT be on PYTHONPATH, or Python will import the
   wandb cache directory instead of the wandb package.

4. **Validate:**
   ```bash
   python -c "import wandb; wandb.login()"
   ```

## Run naming convention
`{phase}_{model_name}` -- e.g. `phase2_stdc2`, `phase2_segformer_mit-b2`

## Entity/Project
- Entity: `little_farms`
- Project: `embedded_segmentation_macvi2026`

## Failed run cleanup
If a training run fails and must be restarted, delete the failed run from WandB first
(`wandb run delete <run-id>` or via WandB web UI) before relaunching.
