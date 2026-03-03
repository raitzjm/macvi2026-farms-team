# Phase 1 Failure Analysis

Per-image segmentation failure analysis for the three Phase 1 ONNX models on the
LaRS validation set (198 images). Requested by Rahul in issue #12 for "Dice, mIoU,
Accuracy — per patch metrics — all the metrics for all the classes."

## Models

| Model | Role | ONNX |
|---|---|---|
| **STDC2** | Phase 1 baseline | `phase1/stdc2/stdc2_logits.onnx` |
| **SegNeXt-T** | Embedded candidate, still climbing | `phase1/segnext_t/segnext_t_logits.onnx` |
| **SegFormer MiT-B2** | Upper-bound reference (> 30 FPS limit) | `phase1/segformer_mit-b2/segformer_mit-b2_logits.onnx` |

BiSeNetV2 is excluded — training was unstable and no ONNX was committed.

## Files

```
failure_analysis/
├── README.md                  ← this file
├── setup.md                   ← environment install log
├── GOTCHAS.md                 ← failures, fixes, methodology decisions
├── infer_and_eval.py          ← inference + per-image metric computation
├── build_report.py            ← figure generator + PDF report builder
├── failure_analysis.ipynb     ← interactive Plotly notebook
├── per_image_metrics.csv      ← per-image metrics, all 198 images × 3 models
├── figures/
│   ├── iou_distributions.png        ← violin plots per class × model
│   ├── overlap_upset.png            ← cross-model failure Venn diagram
│   ├── worst_cases_stdc2.png        ← worst 20 images gallery
│   ├── worst_cases_segnext_t.png
│   ├── worst_cases_segformer.png
│   └── sequence_breakdown.png       ← mean obstacle IoU per sequence
└── logs/
    └── inference.log
```

Report: `../reports/failure_analysis_report.pdf`

## How to reproduce

```bash
# 0. Ensure you are in the macvi conda env
# 1. Pull LFS files
git lfs pull

# 2. Rebuild symlink tree (required each container session)
for split in train val test; do
  mkdir -p /home/farms/lars_linked/$split
  ln -sfn /home/farms/data/lars_dataset/img_dir/$split/images /home/farms/lars_linked/$split/images
  ln -sfn /home/farms/data/lars_dataset/ann_dir/$split/semantic_masks /home/farms/lars_linked/$split/semantic_masks 2>/dev/null || true
done

# 3. Run inference (generates per_image_metrics.csv and inference.log)
cd /home/farms/macvi2026-farms-team
python challenges/common/lars-analysis/failure_analysis/infer_and_eval.py

# 4. Generate figures and PDF report
python challenges/common/lars-analysis/failure_analysis/build_report.py
```

## Class ordering

obstacle=0, water=1, sky=2  (verified and corrected in PR #11)

## Metrics

- **IoU**: TP / (TP + FP + FN), per class
- **Dice / F1**: 2·TP / (2·TP + FP + FN), per class
- **Pixel accuracy (per class)**: TP / (TP + FN) = per-class recall
- **mIoU**: mean IoU over all 3 classes (all 198 images have all 3 classes)
- **Ignore index**: GT pixels with value 255 are excluded from all computations

## Preprocessing note

Images are simply resized to 768×384 (bilinear, no keep-ratio, no padding) to match
the preprocessing used in `export_and_eval.py`. This ensures per-image metrics aggregate
to approximately the Phase 1 reported mIoU values.
