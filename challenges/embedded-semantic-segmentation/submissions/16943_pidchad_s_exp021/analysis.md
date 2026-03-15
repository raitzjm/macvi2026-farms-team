# Submission #16943 — PIDChad-S EXP-021 Analysis

## Server Results

| Metric | Value |
|--------|-------|
| Q | 46.40 |
| mIoU | 92.88% |
| F1 | 49.95% |
| Pr | 46.32% |
| Re | 54.20% |
| WE_acc (μ) | 66.11% |
| WE_IoU | 40.27% |
| FP | 5170 |
| FN | 3771 |
| TP | 4462 |

## Comparison to EXP-020 (PIDNet-S, Sub #16909)

| Metric | EXP-020 | EXP-021 | Δ |
|--------|---------|---------|---|
| Q | 52.11 | 46.40 | **−5.71** |
| mIoU | 94.36% | 92.88% | −1.48% |
| F1 | 55.22% | 49.95% | −5.27% |
| Pr | 54.10% | 46.32% | −7.78% |
| Re | 56.40% | 54.20% | −2.20% |
| FP | 3939 | 5170 | +1231 |
| FN | 3590 | 3771 | +181 |

## Key Observations

1. **Precision collapse** (54.1% → 46.3%): The AFFM/FPPM structural additions and aggressive LSJ (0.1–2.0) caused a false positive explosion (+1231 FPs). The attention-gated fusion was intended to suppress water-reflection FPs but had the opposite effect on the test set.

2. **Recall also dropped** (56.4% → 54.2%): Despite more aggressive small-scale augmentation (LSJ 0.1), the model did not improve recall. The architectural changes may have destabilized the detail branch.

3. **mIoU regression** (94.4% → 92.9%): Segmentation quality dropped, particularly in sky class (visible in local eval: 90.77% → 88.25% sky IoU).

4. **Multiple confounding variables**: EXP-021 changed architecture (AFFM+FPPM), LSJ range (0.5→0.1), boundary loss weight (20→25), and training length (15k→25k) simultaneously. Cannot isolate which change caused the regression.

See `phase3/phase3_report.md` for full Phase 3 analysis.
