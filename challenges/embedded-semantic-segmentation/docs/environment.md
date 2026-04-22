# Environment

Detailed environment setup for the MaCVi embedded semantic segmentation challenge.
For the quick bootstrap summary, see the `## Env Bootstrap` section in the challenge CLAUDE.md.

## Two Conda Environments

### `macvi` — Phase 1 (mmseg 0.30.0 fork)

- Path: `/home/farms/miniforge3/envs/macvi/`
- Python 3.11 (conda-forge), CUDA 12.1
- PyTorch 2.1.2+cu121
- mmcv-full 1.7.2 (prebuilt wheel — no source build)
- mmsegmentation 0.30.0 fork (`lojzezust/mmsegmentation-macvi`)
  installed non-editable at `/opt/macvi-env/mmsegmentation-macvi/`
- WandB 0.25.0 (pinned — required for `wandb_` key prefix patch)

Created by: `bootstrap-macvi.sh`

#### Patches baked in (setuptools<70, numpy<2, mmcv scatter patch `_functions.py`)

#### Patches NOT baked in (must reapply after reinstall):
1. **DDP eval fix** — `mmcv/parallel/distributed.py` line ~160:
   change `self._use_replicated_tensor_module` →
   `getattr(self, '_use_replicated_tensor_module', False)`
   (fixes DDP eval under PyTorch 2.1.2)
2. **WandB key prefix patch** — `wandb/sdk/lib/wbauth/validation.py`:
   add early-return for `wandb_` prefix before the 40-char length check
   (required for new wandb API key format starting with `wandb_`)

### `macvi2` — Phase 2+ (mmseg 1.2.2 stock)

- Path: `/home/farms/miniforge3/envs/macvi2/`
- Python 3.11, PyTorch 2.1.2+cu121
- mmcv 2.1.0, mmsegmentation 1.2.2 (stock pip install)
- Additional deps vs `macvi`: `ftfy`, `regex`
  (required by mmseg tokenizer at import time)
- Tools at: `/home/farms/miniforge3/envs/macvi2/lib/python3.11/site-packages/mmseg/.mim/tools/`
- WandB 0.25.0 (pinned — same patch required)

Created by: `bootstrap-macvi2.sh`

#### Patches NOT baked in (must reapply after reinstall):
1. **WandB key prefix patch** — same as `macvi` env above.

## WandB Setup

- Package: `wandb==0.25.0` (pinned)
- API key: `~/data/wandb_key.txt` (read-only mount — never commit)
- Load via: `source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh`
  (or the equivalent setup_wandb.sh in phase2/phase2_5/phase3/scripts/)
- Entity/Project: `little_farms` / `embedded_segmentation_macvi2026`
- Run naming: `{phase}_{model_name}` (e.g. `phase3_pidnets_exp020`)
- `WANDB_DIR` must NOT be on `PYTHONPATH` — run scripts handle this
- Failed runs: delete from WandB before relaunching to keep project clean
- For setup, use the `/setup-wandb` skill.

## Run Commands

```bash
# Phase 1 (macvi env)
conda run --no-capture-output -n macvi python ...

# Phase 2+ (macvi2 env)
conda run --no-capture-output -n macvi2 torchrun ...
# or activate env directly: conda activate macvi2
```
