# Challenge Submissions — Embedded Semantic Segmentation

Challenge: [MaCVi 2026 Embedded Segmentation](https://macvi.org/workshop/cvpr/challenges/usv_seg_emb)
Primary metric: **Q = mIoU × F1 / 100** (higher is better; FPS tiebreaker)
FPS requirement: ≥30 FPS to be eligible for ranking

## Submissions

| Sub ID | Date | Model | EXP | Q | μ | Pr | Re | F1 | mIoU | FPS | Rank | Notes | Analysis |
|--------|------|-------|-----|---|---|----|----|----|------|-----|------|-------|----------|
| #16537 | 2026-03-05 | PIDNet-S SGD | EXP-008 | 49.5 | 67.5 | 52.5 | 54.1 | 53.3 | 92.8 | 67.6 | 36/67 (~7th ≥30FPS) | Phase 2 full training, 5k iters; gap to top mainly mIoU (92.8 vs 95-96% for leaders) | [analysis](submissions/16537_pidnets_sgd/analysis.md) |
| #16597 | 2026-03-10 | STDC2 SGD | EXP-007 | 45.8 | 61.4 | 46.2 | 56.9 | 51.0 | 89.8 | TBD | TBD | Phase 2 full training; worse mIoU and F1 vs PIDNet-S; FP count high (5464); ORCA recall collapsed (23%) | [analysis](submissions/16597_stdc2_sgd/analysis.md) |
| #16738 | 2026-03-10 | PIDNet-S | EXP-018 | 38.9 | 61.1 | 36.1 | 59.3 | 44.9 | 86.5 | TBD | TBD | Phase 2.5 experiment; precision collapse (36.1%), FP explosion (8633 vs 4028 baseline); recall gain doesn't compensate | [analysis](submissions/16738_pidnets_exp018/analysis.md) |
| #16820 | 2026-03-12 | STDC2 SGD | EXP-017 | 44.8 | 65.8 | 49.2 | 47.3 | 48.3 | 92.9 | TBD | TBD | Phase 2.5 re-train with pipeline fixes (keep_ratio=True, centered pad, de-padding); improved mIoU over EXP-007 baseline (#16597) | [analysis](submissions/16820_stdc2_exp017/analysis.md) |
| #16864 | 2026-03-12 | PIDNet-S | EXP-019 | 41.6 | 64.2 | 39.6 | 54.6 | 45.9 | 90.7 | TBD | TBD | Phase 3 CopyPaste + EnhPMD; Success: recovered from EXP-018 FP explosion (-1783 FPs); +2.76% Q-score and +4.12% mIoU vs EXP-018 baseline | [analysis](submissions/16864_pidnets_exp019/analysis.md) |
