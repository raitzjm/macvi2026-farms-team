# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (obstacle=0, water=1, sky=2) on LaRS dataset.
Train model → export ONNX → submit to challenge server. Winner: highest F1
detection score (FPS tiebreaker). Must achieve ≥30 FPS at 768×384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Current Status
All training phases complete. Best submission: EXP-020 PIDNet-S (Q=52.1,
mIoU=94.4%, F1=55.2%). Placed top 3 among ≥30 FPS entries. CVPR 2026
report submitted. Status: **finished** (see `STATE.md`).

## Logging (CRITICAL)
- `experiment-log.md` is for training runs ONLY — non-experiment work goes in git commits
- Before a training run: append a timestamped entry with experiment setup
- After a training run: update the entry with results and status
- Each entry must log: config path, dataset, resolution, epochs, val F1, val mIoU, git commit hash
  (if mmseg evaluator doesn't report F1, record in the phase report and note 'see phase report')

## Env Bootstrap

Two conda envs are used for this challenge. Both created by bootstrap
scripts (NOT baked into the image):

- `bootstrap-macvi.sh`  — mmseg 0.30.0 fork, Phase 1.
- `bootstrap-macvi2.sh` — mmseg 1.2.2 stock, Phase 2+.

Envs live at `/home/farms/miniforge3/envs/macvi{,2}/` and are ephemeral per
container. Run the bootstrap script once after each fresh container.

Full dependency details and patch list: `docs/environment.md`.

## Experiment Phases
- Phase 1: COMPLETE. Short validation runs (1.5k iters, 4 models). Preprocessing validated.
- Phase 2: COMPLETE. Full training on mmseg 1.2.2 (7 models, INT8 eval). See `phase2/PHASE2_NOTES.md`.
- Phase 2.5: COMPLETE. Pipeline fix verification (keep_ratio, de-padding). See `phase2_5/PHASE2_5_NOTES.md`.
- Phase 3: COMPLETE. CopyPaste augmentation + PIDChad-S, SegChad. Best: EXP-020 Q=52.1. See `phase3/phase3_report.md`.

Preprocessing requirements (CRITICAL): `docs/preprocessing.md`.
Config philosophy: `docs/config-philosophy.md`.
ONNX export requirements: `docs/onnx-export.md`.
Output directory convention and artifact archival: `docs/output-convention.md`.
WandB tracking details: `docs/wandb.md`.
Submission results layout: `docs/submission-results.md`.
Dataset paths and symlink notes: `docs/dataset.md`.

## Key Reports & Documents
| Document | Path | Description |
|----------|------|-------------|
| Phase 1 Report | `phase1/phase1_report.md` | Baseline validation (4 models, 1.5k iters), preprocessing checks, architecture rankings |
| Phase 2 Results | `eval_metrics.md` | FP32 & INT8 ONNX evaluation results for all Phase 2 models |
| Phase 3 Report | `phase3/phase3_report.md` | CopyPaste augmentation, PIDChad-S, SegChad experiments and server results |
| Submissions Tracker | `submissions.md` | All challenge submissions with Q, mIoU, F1, FPS, and analysis links |
| Experiment Log | `experiment-log.md` | Append-only log of all training runs (EXP-001 through EXP-022) |
| Reference Baselines | `reference_baselines.md` | Published challenge baseline results for comparison |
| Phase 2 Notes | `phase2/PHASE2_NOTES.md` | Phase 2 env, configs, batch sizes, training launch |
| Phase 2.5 Notes | `phase2_5/PHASE2_5_NOTES.md` | Pipeline fix details, verification methodology |
| LaRS Analysis | `../common/lars-analysis/` | Failure analysis, spatial/whole-image OOD reports (PDFs) |
| External Data Analysis | `../common/external-dataset-analysis/` | WaterScenes, ROSEBUD, MULTIAQUA_night analysis |
| Preliminary Strategy | `../../docs/Embedded_Semantic_Seg_Challenge_MaCVi2026_Preliminary_Report.pdf` | Initial strategy report |

## Key References
- Challenge: macvi.org/workshop/cvpr/challenges/usv_seg_emb
- Starter repo: github.com/lojzezust/mmsegmentation-macvi
