# Experiment Log - Embedded Semantic Segmentation

## Format
Each entry follows this structure:
- **ID**: EXP-NNN
- **Date**: YYYY-MM-DD HH:MM UTC
- **Phase**: current phase
- **Status**: PLANNED | IN PROGRESS | COMPLETED | FAILED | ABANDONED
- **Branch/Commit**: git reference
- **Config**: path to config file
- **Dataset**: what data was used
- **Resolution**: training resolution
- **Epochs**: number of epochs
- **Results (val)**: F1, mIoU, and any other relevant metrics
- **Notes**: observations, comparisons, issues

## Comparison Matrix

| ID | Model | Data | Res | Aug | Val F1 | Val mIoU | Quant F1 | FPS | Commit |
|----|-------|------|-----|-----|--------|----------|----------|-----|--------|
| -- | --    | --   | --  | --  | --     | --       | --       | --  | --     |

---

## Entries

(entries will be appended below by the agent)

---

## EXP-001
- **ID**: EXP-001
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: IN PROGRESS
- **Branch/Commit**: exp/phase1-baselines @ 26c181b
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/stdc2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2607, val: 200)
- **Resolution**: 768×384
- **Iters**: 10 000 (eval every 2 000; save best mIoU)
- **Hardware**: RTX A6000 48 GB
- **bs / lr**: 174 / 0.145 (SGD, poly, warmup 1k iters; linear-scaled from ref bs=12 lr=0.01)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_stdc2
- **Results (val)**: TBD
- **Notes**: First Phase 1 run. Corrected preprocessing vs fun_experiments (reflect pad, keep_ratio=True, test Pad added).

---

## EXP-002
- **ID**: EXP-002
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: IN PROGRESS
- **Branch/Commit**: exp/phase1-baselines @ 26c181b
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/bisenetv2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2607, val: 200)
- **Resolution**: 768×384
- **Iters**: 10 000 (eval every 2 000; save best mIoU)
- **Hardware**: RTX A6000 48 GB
- **bs / lr**: 152 / 1.9 (SGD, poly, warmup 1k iters; linear-scaled from ref bs=4 lr=0.05)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_bisenetv2
- **Results (val)**: TBD
- **Notes**: High scaled LR (1.9) — monitor warmup stability. Runs after EXP-001.

---

## EXP-003
- **ID**: EXP-003
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: IN PROGRESS
- **Branch/Commit**: exp/phase1-baselines @ 26c181b
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/segnext_t_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2607, val: 200)
- **Resolution**: 768×384
- **Iters**: 10 000 (eval every 2 000; save best mIoU)
- **Hardware**: RTX A6000 48 GB
- **bs / lr**: 75 / 0.00006 (AdamW, poly, warmup 1.5k iters; LR fixed per AdamW rule)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_segnext_t
- **Results (val)**: TBD
- **Notes**: Runs after EXP-002.

---

## EXP-004
- **ID**: EXP-004
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: IN PROGRESS
- **Branch/Commit**: exp/phase1-baselines @ 26c181b
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/segformer_mit-b2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2607, val: 200)
- **Resolution**: 768×384
- **Iters**: 10 000 (eval every 2 000; save best mIoU)
- **Hardware**: RTX A6000 48 GB
- **bs / lr**: 45 / 0.00006 (AdamW, poly, warmup 1.5k iters; LR fixed per AdamW rule)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_segformer_mit-b2
- **Results (val)**: TBD
- **Notes**: Accuracy upper-bound reference (above 30 FPS threshold). Runs after EXP-003.