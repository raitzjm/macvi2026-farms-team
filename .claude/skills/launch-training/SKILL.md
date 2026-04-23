# Skill: Launch Phase 2 Training

## When to use
When starting a Phase 2 training run (mmseg 1.2.2, macvi2 conda env).

## Pre-flight checklist

1. **WandB setup**: run `/setup-wandb` skill first (or verify WANDB_API_KEY is set)
2. **LaRS symlinks**: the run script rebuilds these automatically, but verify
   `/home/farms/lars_linked/` exists if running manually
3. **Conda env**: `macvi2` (NOT `macvi` -- that is Phase 1)

## Launch command
```bash
# From repo root, inside container
source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh <model>
```

## Available models
| Model name | Optimizer | Config |
|---|---|---|
| stdc2 | SGD | stdc2_768x384_lars_phase2.py |
| pidnets | SGD | pidnets_768x384_lars_phase2.py |
| bisenetv2 | SGD | bisenetv2_768x384_lars_phase2.py |
| pidnetm | SGD | pidnetm_768x384_lars_phase2.py |
| segformer_mit-b2 | AdamW | segformer_mit-b2_768x384_lars_phase2.py |
| segnext_t | AdamW | segnext_t_768x384_lars_phase2.py |
| segnext_s | AdamW | segnext_s_768x384_lars_phase2.py |
| unet | AdamW | unet_s5d16_768x384_lars_phase2.py |
| stdc2_adamw | AdamW | stdc2_adamw_768x384_lars_phase2.py |
| bisenetv2_adamw | AdamW | bisenetv2_adamw_768x384_lars_phase2.py |
| pidnets_adamw | AdamW | pidnets_adamw_768x384_lars_phase2.py |
| segnext_t_uperhead | AdamW | segnext_t_uperhead_768x384_lars_phase2.py |

## Output directory
`challenges/embedded-semantic-segmentation/outputs/phase2/{model_name}/`
(set automatically by run_phase2.sh)

## DDP details
- 2 GPUs via torchrun (--nproc_per_node=2)
- Batch sizes and LRs are pinned per model (see phase2/AGENTS.md tables)
