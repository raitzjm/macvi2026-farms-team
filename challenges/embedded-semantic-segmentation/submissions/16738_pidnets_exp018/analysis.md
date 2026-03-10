# Submission #16738 — PIDNet-S EXP-018 Analysis

**Date:** 2026-03-10
**Model:** PIDNet-S (EXP-018, Phase 2.5)
**FPS:** TBD (not present in summary.json)

## Server Metrics

| Metric | Value |
|--------|-------|
| Q (primary) | 38.86 |
| mIoU | 86.53% |
| F1 (instance) | 44.91% |
| Precision | 36.13% |
| Recall | 59.32% |
| WE Accuracy | 61.11% |
| WE IoU | 32.31% |
| FPS (server) | TBD |

**IoU per class:**

| Class | IoU |
|-------|-----|
| Obstacle | 77.19% |
| Water | 88.73% |
| Sky | 93.66% |

**Test set:** LaRS test set, 1203 images.

The mIoU has fallen sharply to 86.53% (vs. 92.82% for EXP-008). The instance-level
precision has collapsed to 36.13%, which is the primary driver of the low Q score
(38.86 vs. 49.46 for EXP-008). Recall is modestly improved (59.3% vs. 54.1%) but
does not come close to compensating for the precision loss.

## Instance Detection Breakdown

Total: TP=4884, FP=8633, FN=3349

> **FP explosion warning.** EXP-018 fires 8633 false alarms vs. 4028 for EXP-008 —
> a +114% increase. The model is dramatically more trigger-happy, indicating a
> significant shift in the decision boundary, likely caused by a Phase 2.5 configuration
> change (e.g. altered loss weights, small-object focus, or augmentation strategy)
> that traded precision for recall at a severe net cost to Q.

### FP breakdown by obstacle area

| Size bucket | Count | % of FPs |
|-------------|-------|----------|
| Tiny (<100 px) | 4224 | **48.9%** |
| Small (100–1k px) | 2240 | 25.9% |
| Medium (1k–10k px) | 1449 | 16.8% |
| Large (>10k px) | 720 | 8.3% |

> **49% of false alarms are sub-100 px blobs.** Tiny FPs remain the single largest
> contributor — and their absolute count has nearly doubled compared to EXP-008
> (4224 vs. 2209). Notably, medium and large FPs have also increased substantially,
> suggesting the FP explosion is not limited to texture noise but affects all size
> scales.

### FN breakdown by obstacle area

| Size bucket | Count | % of FNs |
|-------------|-------|----------|
| Tiny (<100 px) | 1260 | 37.6% |
| Small (100–1k px) | 1526 | 45.6% |
| Medium (1k–10k px) | 482 | 14.4% |
| Large (>10k px) | 81 | 2.4% |

> **83% of missed obstacles are <1000 px.** Small obstacle recall remains the core
> recall weakness. Total FN count is slightly lower than EXP-008 (3349 vs. 3780),
> consistent with the higher recall (59.3% vs. 54.1%), but the improvement is modest
> relative to the precision cost.

## Per-Category Recall

| Category | TP | FN | Recall |
|----------|----|----|--------|
| Paddle board | 299 | 102 | **74.6%** |
| Buoy | 14 | 6 | 70.0% |
| Float | 3952 | 2073 | 65.6% |
| Row boat | 94 | 64 | 59.5% |
| Other | 177 | 214 | 45.3% |
| Boat/ship | 314 | 550 | 36.3% |
| Animal | 34 | 340 | **9.1%** |

Buoy recall has improved dramatically relative to EXP-008 (70.0% vs. 32.6%), and
paddle board recall is also higher. However, boat/ship recall has fallen sharply
(36.3% vs. 57.5% for EXP-008), and animal recall remains critically low at 9.1%.
Float is the dominant category by volume (TP+FN=6025) and its 65.6% recall sets
the floor for overall recall.

Note: Swimmer category had zero entries in the segments data for this submission.

## Per-Source Breakdown

| Source | N frames | TP | FP | FN | Recall | Precision |
|--------|----------|----|----|----|--------|-----------|
| SMD | 35 | 195 | 204 | 106 | 64.8% | 48.9% |
| MIT | 58 | 200 | 298 | 46 | 81.3% | 40.2% |
| Davimar | 137 | 358 | 1083 | 103 | 77.7% | 24.8% |
| InHouse | 448 | 2048 | 3478 | 1746 | 54.0% | 37.1% |
| ORCA | 127 | 57 | 546 | 82 | 41.0% | **9.5%** |
| YouTube | 364 | 2009 | 2835 | 1259 | 61.5% | 41.5% |

- **MIT**: best recall (81.3%) but precision is poor (40.2%) — close-range scenes
  generate many spurious detections under EXP-018's more aggressive detection regime.
- **SMD**: precision has dropped sharply (48.9% vs. 90.6% for EXP-008) — the model
  is generating FPs even in the controlled SMD environment.
- **ORCA**: precision is nearly zero (9.5%) — only 57 TPs against 546 FPs. This source
  appears to be the worst-affected by the FP explosion; open-water low-contrast scenes
  are triggering an enormous number of false alarms.
- **Davimar**: precision similarly poor (24.8%) — another open-water source hit hard
  by the precision collapse.
- **InHouse**: largest source (37% of test); absolute FP count of 3478 dominates the
  total FP budget.
- **YouTube**: second-largest contributor to FPs (2835); recall is 61.5%, marginally
  better than EXP-008's 42.7%, but at severe precision cost.

## Regression vs. EXP-008 (#16537)

| Metric | EXP-008 (#16537) | EXP-018 (#16738) | Delta |
|--------|-----------------|-----------------|-------|
| Q | 49.46 | 38.86 | **−10.60** |
| mIoU | 92.82% | 86.53% | −6.29 pp |
| F1 (instance) | 53.28% | 44.91% | −8.37 pp |
| Precision | 52.51% | 36.13% | **−16.38 pp** |
| Recall | 54.09% | 59.32% | +5.23 pp |
| WE Accuracy | 67.47% | 61.11% | −6.36 pp |
| WE IoU | 42.23% | 32.31% | −9.92 pp |
| TP | 4453 | 4884 | +431 |
| FP | 4028 | 8633 | **+4605 (+114%)** |
| FN | 3780 | 3349 | −431 |

EXP-018 is a strict regression on every metric except recall. The FP explosion (+4605,
+114%) is the dominant failure mode. The small improvement in TP (+431) and recall
(+5.2 pp) does not compensate: F1 and Q are both substantially worse. mIoU has also
regressed significantly (−6.3 pp), indicating the segmentation quality has degraded
beyond the instance-level effects. WE metrics have deteriorated as well, suggesting
the water/obstacle boundary quality is worse.

The pattern — higher recall, much worse precision, mIoU regression, FP explosion —
is consistent with a training configuration that encourages the model to predict
obstacle more aggressively (e.g. upweighted obstacle loss, focal loss parameters
favouring recall, aggressive small-object augmentation). The configuration change
produced the intended recall increase but caused an unacceptable false alarm rate
across all scene types and sizes.

## Key Findings

1. **Precision collapse is the primary driver of low Q.** At 36.1%, precision is 16 pp
   below EXP-008. The Q formula is sensitive to F1, and F1 is dominated by precision
   when precision << recall. Recovering precision to EXP-008 levels is the single
   highest-leverage action.

2. **FP explosion is not limited to tiny blobs.** While 49% of FPs are tiny (<100 px),
   the medium (1449) and large (720) FP counts have also increased sharply relative to
   EXP-008 (563 and 227 respectively). The model is not just noisier on texture — it
   is generating large spurious obstacle regions in open-water scenes.

3. **ORCA source is catastrophically affected** (9.5% precision). This is a canary
   for open-water FP generation. Any configuration producing this behaviour should be
   considered unsubmittable for production.

4. **mIoU regression (−6.3 pp) confirms segmentation quality loss.** This is not just
   a post-processing artefact; the pixel-level obstacle map has degraded. The obstacle
   IoU (77.19% vs. an implied ~93%+ for EXP-008) confirms heavy over-prediction in the
   obstacle class.

5. **Recall improvement is insufficient to offset precision loss.** The +5 pp recall
   gain (+431 TPs) is real but marginal. EXP-018 does show that the model can detect
   more obstacles when pushed — this headroom should be exploited via better calibration
   rather than aggressive threshold relaxation.

6. **WE metrics regressed.** WE_acc fell 6.4 pp and WE_IoU fell 9.9 pp, indicating
   the water-edge boundary is less precise. This compounds the small-obstacle recall
   problem and suggests the model's waterline localisation has degraded.

## Actionable Items

| Priority | Action | Expected gain | Notes |
|----------|--------|---------------|-------|
| High | Revert to EXP-008 config as baseline; isolate the EXP-018 delta | Identify root cause of FP explosion | Compare configs diff |
| High | Post-process: discard obstacle CC components <100 px | Eliminate ~4200 tiny FPs; target Pr >50% | Apply to any candidate model |
| High | Tune confidence threshold / loss weights to recover Precision ≥50% | Q recovery toward EXP-008 level | Do not accept Pr <40% in any submission |
| Medium | Investigate ORCA source FP pattern separately | Fix open-water hallucination | ORCA Pr=9.5% is unacceptable |
| Medium | If EXP-018 change was focal loss / class-weight tuning, try intermediate values | Balance Pr/Re trade-off | Target Pr≈Re for max F1 |
| Medium | Improve small obstacle recall via resolution increase or auxiliary head | +5–10 pp recall without Pr sacrifice | Phase 3 action |
| Low | Animal recall (9.1%) — targeted augmentation or copy-paste | Niche improvement | Low frequency class |

## Raw Data

- `results/summary.json` — aggregate server metrics
- `results/frames.csv` — per-image IoU, WE, TP/FP/FN (1203 rows)
- `results/segments.json` — per-obstacle instance records with area, diagonal, coverage
