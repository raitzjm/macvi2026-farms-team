# Submission #16537 — PIDNet-S SGD Analysis

**Date:** 2026-03-05
**Model:** PIDNet-S SGD (EXP-008, `phase2/artifacts/pidnets/best_mIoU_iter_4750.pth`)
**Server rank:** 36/67 overall (~7th among ≥30 FPS submissions)

## Server Metrics

| Metric | Value |
|--------|-------|
| Q (primary) | 49.46 |
| mIoU | 92.82% |
| F1 (instance) | 53.28% |
| Precision | 52.51% |
| Recall | 54.09% |
| WE Accuracy | 67.47% |
| WE IoU | 42.23% |
| FPS (server) | 67.6 |

**Test set:** LaRS test set, 1203 images.

The mIoU is solid (92.82%) but instance-level F1 is weak (53.28%).
Q is the product — improving F1 is the higher-leverage lever.

## Instance Detection Breakdown

Total: TP=4453, FP=4028, FN=3780

### FP breakdown by obstacle area

| Size bucket | Count | % of FPs |
|-------------|-------|----------|
| Tiny (<100 px) | 2209 | **54.8%** |
| Small (100–1k px) | 1029 | 25.5% |
| Medium (1k–10k px) | 563 | 14.0% |
| Large (>10k px) | 227 | 5.6% |

> **55% of false alarms are sub-100 px blobs.** These are likely small water-texture
> artifacts that the model misclassifies as obstacle. Post-processing minimum-size
> filtering (e.g. discard predicted obstacle connected components <100 px) could
> eliminate more than half of all FPs at negligible TP cost.

### FN breakdown by obstacle area

| Size bucket | Count | % of FNs |
|-------------|-------|----------|
| Tiny (<100 px) | 1756 | 46.5% |
| Small (100–1k px) | 1669 | 44.2% |
| Medium (1k–10k px) | 327 | 8.7% |
| Large (>10k px) | 28 | 0.7% |

> **91% of missed obstacles are <1000 px.** Small obstacle recall is the core
> weakness. The model is well-calibrated for large obstacles (only 0.7% of FNs).

## Per-Category Recall

| Category | TP | FN | Recall |
|----------|----|----|--------|
| Float | 17 | 3 | **85.0%** |
| Row boat | 311 | 90 | 77.6% |
| Paddle board | 96 | 62 | 60.8% |
| Boat/ship | 3191 | 2363 | 57.5% |
| Other | 263 | 208 | 55.8% |
| Swimmer | 209 | 182 | 53.5% |
| Buoy | 282 | 582 | 32.6% |
| Animal | 84 | 290 | **22.5%** |

Buoys and animals are severely underdetected. Both tend to be small and visually
ambiguous (buoys blend with water; animals are rare in training data).

## Per-Source Breakdown

| Source | N frames | TP | FP | FN | Recall | Precision |
|--------|----------|----|----|----|--------|-----------|
| SMD | 35 | 240 | 25 | 61 | 79.7% | **90.6%** |
| MIT | 58 | 217 | 186 | 29 | **88.2%** | 53.8% |
| Davimar | 137 | 365 | 618 | 96 | 79.2% | 37.1% |
| InHouse | 448 | 2149 | 1636 | 1645 | 56.6% | 56.8% |
| ORCA | 127 | 76 | 214 | 63 | 54.7% | 26.2% |
| YouTube | 398 | 1406 | 1349 | 1886 | 42.7% | 51.0% |

- **SMD**: best precision (90.6%) — clean controlled environments.
- **MIT**: best recall (88.2%) but many FPs — close-range, clear obstacles but noisy background.
- **Davimar & ORCA**: worst precision — open-water with many small wave/texture FPs.
- **YouTube**: worst recall (42.7%) — diverse lighting and viewpoints; long-tail of small obstacles.
- **InHouse**: largest source (37% of test); balanced but absolute FN count is highest.

## Key Findings

1. **Tiny FP problem dominates precision.** 2209 of 4028 FPs are <100 px. A simple
   morphological area filter on the obstacle segmentation mask would remove most of
   these without touching recall for medium/large obstacles.

2. **Small obstacle recall is the ceiling for F1.** 91% of FNs are <1k px. The model
   is not learning to segment tiny blobs reliably. Strategies: higher input resolution,
   auxiliary small-object detection head, loss weighting by inverse obstacle size.

3. **WE metrics are mediocre** (WE_acc=67.5%, WE_IoU=42.2%). The water-edge boundary
   at the waterline is imprecise. This does not directly affect Q but indicates the
   model struggles with the water/obstacle boundary, which is related to small-obstacle
   detection near the waterline.

4. **Buoy recall is critically low (32.6%).** Buoys are the second most common
   obstacle type (TP+FN=864) but only 1 in 3 is detected. They are typically round,
   small, and appear at the waterline — compounding the small-object problem.

5. **Animal recall is 22.5%.** Rare class, likely underrepresented in training.

## Actionable Items

| Priority | Action | Expected gain | Issue |
|----------|--------|---------------|-------|
| High | Post-process: discard obstacle CC components <100 px | +~10–15% Precision | — |
| High | Train at higher resolution (e.g. 1024×512 or 1280×640) | Better small-obstacle recall | — |
| Medium | Longer training / SegFormer (better val mIoU=71.85% Q) | Baseline F1 improvement | — |
| Medium | Augmentation: small obstacle oversampling / copy-paste | Buoy + animal recall | — |
| Low | Instance detection head (Mask R-CNN style) | Better instance-level F1 | — |

## Raw Data

- `results/summary.json` — aggregate server metrics
- `results/frames.csv` — per-image IoU, WE, TP/FP/FN (1203 rows)
- `results/segments.json` — per-obstacle instance records with area, diagonal, coverage
