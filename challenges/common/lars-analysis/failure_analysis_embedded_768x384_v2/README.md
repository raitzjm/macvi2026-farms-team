# Phase 2/2.5 Failure Analysis — Embedded Semantic Segmentation (768×384)

Per-image segmentation failure analysis for all Phase 2 and Phase 2.5 ONNX models on the
LaRS validation set (198 images).

**Challenge:** Embedded Semantic Segmentation (≥30 FPS on Luxonis RVC4)
**Inference resolution:** 768×384 — the embedded challenge target resolution.
**Issue:** #39

## Models

| Model | Phase | Exp | Role | ONNX |
|---|---|---|---|---|
| **SegFormer MiT-B2** | P2 | EXP-005 | Upper-bound reference (> 30 FPS limit) | `onnx_exports/phase2/segformer_mit-b2/` |
| **SegNeXt-T + UPerHead + BoundaryLoss** | P2 | EXP-015 | Highest FP32 F1; INT8-unstable | `onnx_exports/segnext_t_uperhead/` (local) |
| **PIDNet-S** | P2 | EXP-008 | Best INT8-stable model | `onnx_exports/phase2/pidnets/` |
| **SegNeXt-S** | P2 | EXP-010 | Larger MSCAN backbone | `onnx_exports/phase2/segnext_s/` |
| **SegNeXt-T** | P2 | EXP-006 | Standard LightHamHead | `onnx_exports/phase2/segnext_t/` |
| **BiSeNetV2** | P2 | EXP-009 | SGD training | `onnx_exports/phase2/bisenetv2/` |
| **STDC2** | P2 | EXP-007 | Phase 2 baseline | `onnx_exports/phase2/stdc2/` |
| **STDC2** | P2.5 | EXP-017 | Pipeline fix (+1.99% Q) | `onnx_exports/phase2_5/phase2_5/stdc2/` |
| **PIDNet-S** | P2.5 | EXP-018 | Pipeline fix (−6.30% Q regression) | `onnx_exports/phase2_5/phase2_5/pidnets/` |

All ONNX are FP32 argmax format (output: 1×1×384×768 int64). 8 models pre-exported to
`/home/farms/data/onnx_exports/` (read-only mount). segnext_t_uperhead requires
`export_onnx.py` (see below).

## Files

```
failure_analysis_embedded_768x384_v2/
├── README.md                  ← this file
├── setup.md                   ← environment install log
├── GOTCHAS.md                 ← failures, fixes, methodology decisions
├── export_onnx.py             ← exports segnext_t_uperhead ONNX from LFS checkpoint
├── infer_and_eval.py          ← inference + per-image metric computation
├── build_report.py            ← figure generator + PDF report builder
├── per_image_metrics.csv      ← per-image metrics, all 198 images × 9 models
├── onnx_exports/              ← local ONNX output (gitignored except segnext_t_uperhead)
├── figures/
│   ├── model_ranking.png              ← horizontal bar chart: mean obstacle IoU ranking
│   ├── iou_distributions.png          ← violin plots per class × model
│   ├── failure_overlap.png            ← breadth distribution + pairwise co-failure heatmap
│   ├── worst_cases_{model}.png        ← worst 20 images gallery (one per model)
│   └── sequence_breakdown.png         ← mean obstacle IoU per sequence
└── logs/
    └── inference.log
```

Report: `../reports/failure_analysis_embedded_768x384_v2_report.pdf`

## How to reproduce

```bash
# 0. Ensure you are in the macvi/macvi2 conda env as appropriate (see steps below)
# 1. Pull LFS files (checkpoint for segnext_t_uperhead)
git lfs pull

# 2. Export segnext_t_uperhead ONNX (macvi2 env — mmseg 1.2.2 required)
conda run --no-capture-output -n macvi2 \
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/export_onnx.py

# 3. Run inference (macvi env — generates per_image_metrics.csv and inference.log)
conda run --no-capture-output -n macvi \
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/infer_and_eval.py

# 4. Generate figures and PDF report (macvi env)
conda run --no-capture-output -n macvi \
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/build_report.py
```

## Methodology vs v1

| Aspect | v1 (Phase 1) | v2 (Phase 2/2.5) |
|--------|-------------|-----------------|
| ONNX type | Logits (1,3,384,768) float32 | Argmax (1,1,384,768) int64 |
| Post-inference | Bilinear upsample logits → argmax | Crop reflect-pad → nearest-neighbour resize |
| Preprocessing | Simple resize to 768×384 | Keep-ratio resize → center reflect-pad |
| Execution provider | CPU only | CUDA if available, fallback CPU |
| Val dataset path | `/home/farms/lars_linked/val/` (symlink) | `/home/farms/data/datasets/lars_dataset/` (direct) |
| Models | 3 (Phase 1) | 9 (Phase 2 + Phase 2.5) |
| Failure overlap viz | 3-circle Venn diagram | Breadth distribution + pairwise co-failure heatmap |

## Metrics

- **IoU**: TP / (TP + FP + FN), per class
- **Dice / F1**: 2·TP / (2·TP + FP + FN), per class
- **Pixel accuracy (per class)**: TP / (TP + FN) = per-class recall
- **mIoU**: mean IoU over all 3 classes (all 198 images have all 3 classes)
- **Ignore index**: GT pixels with value 255 are excluded from all computations
- **Failure threshold**: obstacle IoU < 0.5
