# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (obstacle=0, water=1, sky=2) on LaRS dataset.
We train and export an ONNX model. The challenge platform handles INT8
quantization (Qualcomm SNPE) and deployment (Luxonis RVC4) - we do NOT do this.
Our job: train model -> export ONNX -> submit ONNX to challenge server.
Winner: highest F1 detection score (FPS tiebreaker). Must achieve >=30 FPS at 768x384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Submission Results

Server result files (frames.csv, segments.json, summary.json) are downloaded from the
challenge platform after each submission and stored at:
  `~/data/submission_results/<submission_id>_results/`

This path is a read-only mount inside the container (passed via `--data-dir` to run.sh).
When tracking a new submission, copy the result files into:
  `challenges/embedded-semantic-segmentation/submissions/<id>_<model>_<exp>/results/`

The `summary.json` fields map to submissions.md columns as follows:
- Q → Q score  |  WE_acc → μ (waterline estimation accuracy)
- Pr, Re, F1 → detection metrics  |  mIoU → segmentation quality
- FPS is NOT in summary.json — leave as TBD unless provided separately

## Dataset
- LaRS: /home/farms/data/lars_dataset/ (train: 2605, val: 198, test: 1205 images)
- Test set has NO segmentation masks
- External data allowed but must be disclosed
- img_dir: {split}/images/, ann_dir: {split}/semantic_masks/
- Symlink tree at /home/farms/lars_linked/ must be rebuilt each container session (not persistent)

## Workspace
- mmsegmentation-macvi/: copied from github.com/lojzezust/mmsegmentation-macvi
- Training resolution: 768x384 (CRITICAL - always train at target resolution)

## Key Dependencies
- Conda env `macvi` at /home/farms/miniforge3/envs/macvi/ (on PATH by default)
- Python 3.11 (conda-forge), CUDA 12.1
- PyTorch 2.1.2+cu121, mmcv-full 1.7.2 (prebuilt wheel — no source build)
- mmsegmentation 0.30.0 installed non-editable at /opt/macvi-env/mmsegmentation-macvi/
- All baked into the Docker image — no post-start setup needed
- Required patches baked in: setuptools<70, numpy<2, mmcv scatter patch (_functions.py)
- **NOT baked in (must reapply after reinstall)**:
  - `mmcv/parallel/distributed.py` line ~160: `self._use_replicated_tensor_module` →
    `getattr(self, '_use_replicated_tensor_module', False)` (DDP eval fix for PyTorch 2.1.2)
  - `wandb/sdk/lib/wbauth/validation.py`: add early-return for `wandb_` prefix before the
    40-char check (required for new wandb API key format starting with `wandb_`)

## WandB Experiment Tracking

- **API key**: `~/data/wandb_key.txt` (read-only mount — never commit). Load via
  `source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh`
- **Entity/Project**: `little_farms` / `embedded_segmentation_macvi2026`
- **Run naming**: `{phase}_{model_name}` (e.g. `phase3_pidnets_exp020`)
- **WANDB_DIR**: must NOT be on `PYTHONPATH` — run scripts handle this
- **Failed runs**: delete from WandB before relaunching to keep project clean
- For setup, use the `/setup-wandb` skill.

## Preprocessing Requirements (CRITICAL — must match challenge server)

The challenge server preprocesses images before feeding them to our ONNX:
1. Resize with **bilinear interpolation, aspect ratio preserved** → fits within 384×768
2. **Center** the resized image in the 384×768 canvas
3. Fill remaining space with **mirror padding** (not zero padding)
4. Normalize with ImageNet mean/std (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Required mmseg pipeline settings:**
- Test: `Resize(keep_ratio=True)` + `Pad(padding_mode='reflect')` with centering
- Train: `Pad(padding_mode='reflect')` (centering not needed in train)

## Config Philosophy

- **Architecture-specific** (optimizer, LR, losses, heads): left untouched per model
- **Shared** (resolution, pipeline, schedule, dataset paths): standardised across models
- **LR scaling**: SGD uses linear scaling rule; AdamW keeps LR fixed. See issue #9.
- **SyncBN**: required for BatchNorm models (STDC2, BiSeNetV2) when training with DDP
- **Runner type**: `IterBasedRunner` (mmseg 0.30.0) / iter-based (mmseg 1.2.2)

## Experiment Phases
Phase 1: COMPLETE. Short validation runs (1.5k iters, 4 models). Preprocessing validated.
Phase 2: COMPLETE. Full training on mmseg 1.2.2 (7 models, INT8 eval). See `phase2/CLAUDE.md`.
Phase 2.5: COMPLETE. Pipeline fix verification (keep_ratio, de-padding). See `phase2_5/CLAUDE.md`.
Phase 3: COMPLETE. CopyPaste augmentation + architecture experiments (PIDChad-S, SegChad).
          Best result: EXP-020 PIDNet-S, Q=52.1 (server). See `phase3/phase3_report.md`.
Phase 4: Not pursued (external data used as augmentation source in Phase 3 instead).

## ONNX Export Requirements
- Output: argmaxed class map, shape 1×1×384×768, dtype int64
- Opset version: 13 (server uses SNPE which requires opset 13)
- See fun_experiments/{model}/export_onnx.py for reference implementations

## Output Directory Convention

Training live outputs (checkpoints, logs) go to:
  `challenges/embedded-semantic-segmentation/outputs/{phase}/{model_name}/`

Pass this as `--work-dir` to the train script. Examples:
  `outputs/phase1/stdc2/`
  `outputs/phase2/stdc2/`

This keeps phases isolated and prevents checkpoint collisions. The `outputs/` tree
is gitignored — only curated artifacts are committed (see Phase 1 Artifact Archival).

## Artifact Archival

After each training run completes, use the `/archive-experiment` skill.
It handles ONNX export, pth vs ONNX precision validation, artifact commit, and
experiment-log.md update. INT8 quantization evaluation is deferred to Phase 2.

### Phase 2 committed checkpoints

Best checkpoints for INT8-stable models are committed to Git LFS under
`phase2/artifacts/{model}/best_mIoU_iter_NNNN.pth`.

| Model | Checkpoint | Val mIoU | Val F1 | Q | INT8-stable? |
|-------|-----------|----------|--------|---|--------------|
| pidnets (EXP-008) | best_mIoU_iter_4750.pth | 93.25% | 76.17% | 71.03% | ✓ |
| pidnetm (EXP-016) | best_mIoU_iter_3250.pth | 92.33% | 72.82% | 67.23% | ✓ |
| stdc2 (EXP-007) | best_mIoU_iter_2750.pth | 90.69% | 68.92% | 62.50% | ✓ |
| bisenetv2 (EXP-009) | best_mIoU_iter_6000.pth | 91.09% | 69.06% | 62.91% | ✓ |
| segnext_t_uperhead (EXP-015) | best_mIoU_iter_4250.pth | 92.48% | 80.21% | 74.18% | ✗ (−78%) |
| segnext_s (EXP-010) | best_mIoU_iter_4750.pth | 92.40% | 72.33% | 66.83% | ✗ (−82%) |
| segnext_t (EXP-006) | best_mIoU_iter_4250.pth | 91.79% | 71.74% | 65.85% | ✗ (−67%) |

SegNeXt models (EXP-006/010/015) committed to LFS for reference only — INT8 catastrophic
failure (MSCAN large-kernel depthwise attention). See `eval_metrics.md` for full results.

## Current Status
All training phases complete. Team placed **top 3** on the challenge leaderboard.
Best submission: EXP-020 PIDNet-S (Q=52.1, mIoU=94.4%, F1=55.2%).
Now preparing the challenge report for CVPR 2026.

## Key Reports & Documents
| Document | Path | Description |
|----------|------|-------------|
| Phase 1 Report | `phase1/phase1_report.md` | Baseline validation (4 models, 1.5k iters), preprocessing checks, architecture rankings |
| Phase 3 Report | `phase3/phase3_report.md` | CopyPaste augmentation, PIDChad-S, SegChad experiments and server results |
| Submissions Tracker | `submissions.md` | All challenge submissions with Q, mIoU, F1, FPS, and analysis links |
| Experiment Log | `experiment-log.md` | Append-only log of all training runs (EXP-001 through EXP-022) |
| Eval Metrics | `eval_metrics.md` | Phase 2 FP32 & INT8 ONNX evaluation results for all models |
| Reference Baselines | `reference_baselines.md` | Published challenge baseline results for comparison |
| LaRS Analysis | `../common/lars-analysis/` | Failure analysis, spatial/whole-image OOD reports (PDFs) |
| External Data Analysis | `../common/external-dataset-analysis/` | WaterScenes, ROSEBUD, MULTIAQUA_night analysis |
| Preliminary Strategy | `../../docs/Embedded_Semantic_Seg_Challenge_MaCVi2026_Preliminary_Report.pdf` | Initial strategy report |

## Key References
- Challenge: macvi.org/workshop/cvpr/challenges/usv_seg_emb
- Starter repo: github.com/lojzezust/mmsegmentation-macvi
