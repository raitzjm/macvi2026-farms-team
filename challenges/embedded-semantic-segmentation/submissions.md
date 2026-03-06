# Challenge Submissions — Embedded Semantic Segmentation

Challenge: [MaCVi 2026 Embedded Segmentation](https://macvi.org/workshop/cvpr/challenges/usv_seg_emb)
Primary metric: **Q = mIoU × F1 / 100** (higher is better; FPS tiebreaker)
FPS requirement: ≥30 FPS to be eligible for ranking

## Submissions

| Sub ID | Date | Model | EXP | Q | μ | Pr | Re | F1 | mIoU | FPS | Rank | Notes |
|--------|------|-------|-----|---|---|----|----|----|------|-----|------|-------|
| #16537 | 2026-03-05 | PIDNet-S SGD | EXP-008 | 49.5 | 67.5 | 52.5 | 54.1 | 53.3 | 92.8 | 67.6 | 36/67 (~7th ≥30FPS) | Phase 2 full training, 5k iters; gap to top mainly mIoU (92.8 vs 95-96% for leaders) |
