# Submission #16597 — STDC2 SGD Analysis

**Date:** 2026-03-10
**Model:** STDC2 SGD (EXP-007, `phase2/artifacts/stdc2/best_mIoU_iter_2750.pth`)
**FPS:** TBD (not reported in summary.json)

## Server Metrics

| Metric | Value |
|--------|-------|
| Q (primary) | 45.77 |
| mIoU | 89.82% |
| IoU obstacle | 82.97% |
| IoU water | 92.62% |
| IoU sky | 93.86% |
| F1 (instance) | 50.96% |
| Precision | 46.15% |
| Recall | 56.88% |
| WE Accuracy | 61.40% |
| WE IoU | 32.33% |

**Test set:** LaRS test set, 1203 images.

mIoU is weaker than PIDNet-S (89.82% vs 92.82%) and instance-level F1 is also lower
(50.96% vs 53.28%). Precision is notably poor (46.15% vs 52.51% for PIDNet-S), meaning
STDC2 generates proportionally more false alarms. Q is −3.7 points below PIDNet-S
(45.77 vs 49.46). Both levers — Precision and Recall — are weaker, but Precision is
the larger drag given the high absolute FP count.

## Instance Detection Breakdown

Total: TP=4683, FP=5464, FN=3550

### FP breakdown by obstacle area

| Size bucket | Count | % of FPs |
|-------------|-------|----------|
| Tiny (<100 px) | 2886 | **52.8%** |
| Small (100–1k px) | 1345 | 24.6% |
| Medium (1k–10k px) | 847 | 15.5% |
| Large (>10k px) | 386 | 7.1% |

> **53% of false alarms are sub-100 px blobs.** This is essentially the same pattern
> as PIDNet-S (54.8%). STDC2 has 1436 more FPs in absolute terms (5464 vs 4028), spread
> across all size buckets, but the tiny-FP fraction is structurally identical. Post-processing
> minimum-size filtering (e.g. discard predicted obstacle connected components <100 px) would
> eliminate the majority of FPs at negligible TP cost — and the gain applies equally to both
> architectures.

### FN breakdown by obstacle area

| Size bucket | Count | % of FNs |
|-------------|-------|----------|
| Tiny (<100 px) | 1426 | 40.2% |
| Small (100–1k px) | 1524 | 42.9% |
| Medium (1k–10k px) | 528 | 14.9% |
| Large (>10k px) | 72 | 2.0% |

> **83% of missed obstacles are <1000 px.** This is a moderate improvement in the
> small-obstacle FN share compared to PIDNet-S (91% of FNs <1k px), but the absolute
> FN count in the small bucket is higher for STDC2 (1524 vs 1669 — comparable). The
> medium and large FN counts are significantly worse for STDC2 (528 vs 327 medium,
> 72 vs 28 large), indicating that STDC2 also misses more non-tiny obstacles — a
> broader recall deficit, not just a small-object problem.

## Per-Category Recall

| Category | TP | FN | Recall |
|----------|----|----|--------|
| Buoy | 14 | 6 | **70.0%** |
| Float | 3881 | 2144 | 64.4% |
| Paddle board | 247 | 154 | 61.6% |
| Row boat | 94 | 64 | 59.5% |
| Other | 133 | 258 | 34.0% |
| Boat/ship | 229 | 635 | **26.5%** |
| Animal | 85 | 289 | **22.7%** |

> **Note:** "Swimmer" category had zero entries in the segments data.
> "Float" in the raw data likely maps to small floating debris; "Boat/ship" corresponds
> to the large vessel category (see #16537 for category mapping interpretation).

Boat/ship recall is catastrophically low at 26.5% — compared to 57.5% for PIDNet-S,
this is a regression of more than 30 percentage points on the largest and most safety-
critical obstacle class. Note: raw TP/FN counts are not directly comparable across
analyses because category_id mapping is inconsistent between submissions (see note below);
recall percentages are the valid cross-model comparison. This recall collapse is the
primary driver of STDC2's lower Q.

Animal recall (22.7%) is comparable to PIDNet-S (22.5%) — both models effectively fail
on this rare class.

Buoy recall (70.0%) is surprisingly the strongest category and a reversal from PIDNet-S
(32.6%), though the absolute counts are very small (TP=14, FN=6) so this may not be
statistically robust.

## Per-Source Breakdown

| Source | N frames | TP | FP | FN | Recall | Precision |
|--------|----------|----|----|----|--------|-----------|
| SMD | 35 | 212 | 70 | 89 | 70.4% | 75.2% |
| MIT | 58 | 214 | 229 | 32 | 87.0% | 48.3% |
| Davimar | 137 | 333 | 771 | 128 | 72.2% | 30.2% |
| InHouse | 448 | 1992 | 2257 | 1802 | 52.5% | 46.9% |
| ORCA | 127 | 32 | 165 | 107 | **23.0%** | **16.2%** |
| YouTube | 364 | 1878 | 1853 | 1390 | 57.5% | 50.3% |

- **SMD**: reasonable precision (75.2%) but recall has dropped vs PIDNet-S (70.4% vs 79.7%) —
  controlled environments where STDC2 should perform well but misses more obstacles.
- **MIT**: best recall (87.0%), close to PIDNet-S (88.2%), but very low precision (48.3% vs
  53.8%) — close-range clear obstacles come through but background FPs are higher.
- **Davimar**: precision collapses to 30.2% (vs 37.1% for PIDNet-S) — open-water texture FPs
  are worse for STDC2.
- **ORCA**: extremely poor on both axes — 23.0% recall and 16.2% precision. PIDNet-S achieved
  54.7% recall on ORCA. This is a severe regression. ORCA frames likely feature challenging
  lighting, glare, or unusual viewpoints where STDC2's shallower feature extraction fails.
  The low precision (16.2%) also means that what STDC2 does predict in ORCA scenes is mostly
  noise.
- **YouTube**: recall improved versus PIDNet-S (57.5% vs 42.7%) — the diverse YouTube
  distribution suits STDC2 slightly better, or the PIDNet-S YouTube frames had more
  hard small obstacles. Precision is comparable (50.3% vs 51.0%).
- **InHouse**: largest source (37% of test); recall (52.5%) and precision (46.9%) are both
  weaker than PIDNet-S (56.6% / 56.8%). InHouse drives the most absolute FNs (1802).

## Key Findings

1. **Tiny FP problem is structurally identical to PIDNet-S.** 52.8% of STDC2 FPs are
   <100 px (vs 54.8% for PIDNet-S). However, STDC2 has 1436 more FPs in total, so the
   absolute tiny-FP count is higher (2886 vs 2209). Minimum-area post-processing is the
   highest-priority precision lever for both architectures.

2. **Boat/ship recall is the defining regression vs PIDNet-S.** STDC2 detects only 26.5%
   of vessel instances vs 57.5% for PIDNet-S — a collapse of ~30 pp on the dominant
   obstacle category. This is the single largest contributor to STDC2's lower Q score.

3. **ORCA source is effectively non-functional.** 23.0% recall and 16.2% precision
   on ORCA versus 54.7% / 26.2% for PIDNet-S. The ORCA distribution (presumably challenging
   open-sea conditions) exposes a major robustness gap in STDC2's shallow backbone.

4. **WE metrics are substantially weaker.** WE_acc=61.40% and WE_IoU=32.33% versus
   67.47% / 42.23% for PIDNet-S. Water-edge boundary quality is worse, indicating
   that STDC2's segmentation of the water/obstacle interface is less precise — consistent
   with a shallower architecture lacking the detailed boundary refinement of PIDNet-S's
   dual-resolution head.

5. **Small obstacle recall issue persists.** 83% of FNs are <1k px, same structural
   problem as PIDNet-S (91%). Higher input resolution and small-object augmentation
   remain necessary regardless of architecture choice.

## Comparison to PIDNet-S (#16537)

| Metric | STDC2 (#16597) | PIDNet-S (#16537) | Delta |
|--------|---------------|-------------------|-------|
| Q | 45.77 | 49.46 | **−3.69** |
| mIoU | 89.82% | 92.82% | −3.00 pp |
| F1 | 50.96% | 53.28% | −2.32 pp |
| Precision | 46.15% | 52.51% | −6.36 pp |
| Recall | 56.88% | 54.09% | +2.79 pp |
| WE Accuracy | 61.40% | 67.47% | −6.07 pp |
| WE IoU | 32.33% | 42.23% | −9.90 pp |
| FP total | 5464 | 4028 | +1436 |
| FN total | 3550 | 3780 | −230 |

STDC2 has marginally better recall (+2.79 pp) but substantially worse precision
(−6.36 pp). The Q gap of −3.69 points is driven by the precision collapse and the
weaker mIoU, not by STDC2 missing more obstacles overall. The key levers to close
the gap are: (1) reduce FPs via area filtering, (2) address the Boat/ship detection
failure (likely requires better feature capacity for medium-to-large objects), and
(3) fix the ORCA robustness gap.

## Actionable Items

| Priority | Action | Expected gain | Issue |
|----------|--------|---------------|-------|
| High | Post-process: discard obstacle CC components <100 px | +~10–15% Precision | — |
| High | Prefer PIDNet-S over STDC2 for further Phase 2/3 work | Baseline Q +3.7 points | — |
| Medium | Investigate ORCA failure mode (data analysis, visualisation) | Diagnose Boat/ship + ORCA gap | — |
| Medium | Train at higher resolution (e.g. 1024×512 or 1280×640) | Better small-obstacle recall | — |
| Low | Augmentation: small obstacle oversampling / copy-paste | Buoy + animal recall | — |

## Raw Data

- `results/summary.json` — aggregate server metrics
- `results/frames.csv` — per-image IoU, WE, TP/FP/FN (1203 rows)
- `results/segments.json` — per-obstacle instance records with area, diagonal, coverage
