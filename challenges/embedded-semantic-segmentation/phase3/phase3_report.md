# Phase 3 Report: Augmentation and Architecture Enhancements

## 1. Executive Summary
Phase 3 of the Embedded Semantic Segmentation challenge focused on improving the overall Q score (mIoU × F1) through data augmentations and architectural enhancements. Building upon the solid INT8-stable PIDNet-S baseline established in Phase 2 (EXP-008), this phase introduced CopyPaste augmentations utilizing external datasets (WaterScenes, ROSEBUD, MULTIAQUA_night) to address obstacle detection (EXP-020). We further experimented with an upgraded architecture, PIDChad-S (EXP-021), featuring Attention-based Feature Fusion Module (AFFM) and Fast Pyramid Pooling Module (FPPM) to mitigate water-reflection false positives and improve waterline estimation. 

While EXP-020 successfully improved our leaderboard standing with a new best Q score of 52.11%, EXP-021's architectural additions exhibited regression on the server, highlighting the difficulty of translating structural changes to the challenge platform.

## 2. Phase 3 Goals
The primary goal for Phase 3 was to surpass the Phase 2 baseline Q score (EXP-008: Q=49.46) by enhancing obstacle recall and reducing false positives, without compromising the INT8 quantization stability required for the Luxonis RVC4 deployment target.

Key objectives:
- **Improve Obstacle Detection:** Increase F1 score via targeted CopyPaste augmentations from external datasets.
- **Reduce False Positives:** Suppress water-reflection misclassifications using structural modifications to the PIDNet architecture.
- **Maintain INT8 Stability:** Ensure any new operations remain safe for Qualcomm SNPE INT8 quantization.

## 3. Training Setup
- **Base Dataset:** LaRS (train: 2605, val: 198)
- **External Data (Augmentation):** CopyPasteObstacles utilizing `copypaste_candidates.db` sourced from WaterScenes, ROSEBUD, and MULTIAQUA_night.
- **Augmentation Pipeline:** 
  - `CopyPasteObstacles` (n_paste=1-3, prob_apply=0.7) implemented with a shared RAM cache to minimize data loading bottlenecks (`data_time` reduced from ~0.85s to ~0.53s).
  - `EnhancedPhotoMetricDistortion` for color/lighting robustness.
  - Aggressive Large Scale Jittering (LSJ) added in later experiments to capture small-scale signal.
  - `GenerateEdgeMasked` to resolve edge target conflicts in reflect-padded regions.
- **Hardware:** 2× NVIDIA RTX A6000 48 GB GPUs (DDP, `torchrun --standalone --nproc_per_node=2`)
- **Resolution:** 768×384 (Keep-ratio + centered reflect-pad matching server preprocessing).

## 4. EXP-020: PIDNet-S Phase 3 Definitive
EXP-020 served as the definitive augmentation run, rectifying the "augmentation shock" observed in the initial Phase 3 attempt (EXP-019).

**Configuration:**
- Model: PIDNet-S (identical to Phase 2 EXP-008).
- Schedule: 15,000 iterations (extended from 8k/5k) with a 2,500 iteration warmup.
- Optimizer: SGD, lr=0.133, bs=160/GPU (total 320).

**Local Eval (LaRS Val Proxy):**
- **FP32 ONNX:** mIoU = 92.74%, F1 = 75.78%, Q = 70.28%
- **INT8 (ORT Static):** mIoU = 92.32%, F1 = 76.29%, Q = 70.44%
- *INT8 Stability:* Excellent (ΔmIoU = -0.42%, ΔF1 = +0.51%). Maintains ~14-16 FPS locally.

**Server Results (Submission #16909):**
- **mIoU:** 94.36% (vs 92.82% in EXP-008)
- **F1:** 55.22% (vs 53.28% in EXP-008)
- **Q Score:** 52.11 (vs 49.46 in EXP-008)
- **Precision:** 54.10%, **Recall:** 56.39%
- **Errors:** FP = 3939, FN = 3590

**Outcome vs EXP-008:** +2.65% Q score on the server. The extended schedule and CopyPaste augmentations successfully lifted both mIoU and F1 without degrading device performance.

## 5. EXP-021: PIDChad-S (AFFM+FPPM)
EXP-021 aimed to combat the high false positive rate (water reflections) and boost waterline estimation by introducing "PIDChad-S".

**Architectural Changes vs PIDNet-S:**
1. **AFFM (Attention-based Feature Fusion Module):** Added before LightBag. The I-branch context generates a spatial attention map that gates the P-branch (detail), suppressing local texture details in open water regions.
2. **FPPM (Fast Pyramid Pooling Module):** Replaced cascaded PAPPM with parallel AvgPool2d branches at scales (1, 2, 3, 6, 12). This provided a wider global anchor for waterline estimation. Both modules strictly utilized INT8-safe operations (1×1 Conv, BN, ReLU, AvgPool2d).

**Configuration Diffs vs EXP-020:**
- Aggressive LSJ ratio range (0.1, 2.0) to improve recall on tiny obstacles (<100px).
- Obstacle class upweighted (1.5×) in semantic losses.
- BoundaryLoss weight increased from 20.0 to 25.0.
- Extended Schedule: 25,000 iterations (5,000 warmup).

**Server Results (Submission #16943):**
- **mIoU:** 92.88% (vs 94.36% in EXP-020)
- **F1:** 49.95% (vs 55.22% in EXP-020)
- **Q Score:** 46.40 (vs 52.11 in EXP-020)
- **Precision:** 46.32%, **Recall:** 54.20%
- **Errors:** FP = 5170, FN = 3771

**Outcome vs EXP-020:** Significant regression on the server (-5.71% Q score). While recall was moderately maintained (54.20%), precision collapsed (46.32%) as False Positives surged from 3,939 to 5,170. The AFFM/FPPM additions and aggressive LSJ failed to translate to actual performance gains on the test server.

## 6. Server vs Local Gap Analysis
A persistent challenge remains the discrepancy between local LaRS validation metrics and server test set metrics. 

| Metric | EXP-020 Local INT8 | EXP-020 Server | Gap (Absolute) |
|--------|--------------------|----------------|----------------|
| mIoU   | 92.32%             | 94.36%         | +2.04%         |
| F1     | 76.29%             | 55.22%         | -21.07%        |
| Q      | 70.44%             | 52.11%         | -18.33%        |

*Analysis:*
- The server test set has a significantly harsher evaluation for object detection (F1 drops by ~21%). 
- Local val F1 overestimates real-world challenge performance, likely due to test set domain shifts or smaller average obstacle sizes that our local evaluation does not penalize as heavily.
- However, server mIoU actually *exceeded* local mIoU for EXP-020, suggesting our padding/preprocessing pipeline is highly accurate and generalizing well to the test set's background distributions.

## 7. Progression Table
Q score progression across Phase 2 and Phase 3 submissions (Server Metrics):

| Submission | EXP     | Model       | Notes                                      | Q Score | mIoU  | F1    |
|------------|---------|-------------|--------------------------------------------|---------|-------|-------|
| #16537     | EXP-008 | PIDNet-S    | Phase 2 Baseline                           | 49.46   | 92.82 | 53.28 |
| #16864     | EXP-019 | PIDNet-S    | Phase 3 Initial (Aug Shock)                | 41.60   | 90.70 | 45.90 |
| #16909     | EXP-020 | PIDNet-S    | Phase 3 Definitive (15k + CopyPaste)       | **52.11** | **94.36** | **55.22** |
| #16943     | EXP-021 | PIDChad-S   | AFFM + FPPM + Aggressive LSJ               | 46.40   | 92.88 | 49.95 |

## 8. Key Findings & Lessons Learned
1. **CopyPaste Works (When Warmed Up):** Introducing external obstacle crops via `CopyPasteObstacles` is highly effective (EXP-020 beat EXP-008), but models suffer "augmentation shock." A long warmup (2,500 iters) and extended schedule (15,000 iters) are strictly necessary for the model to digest the injected data.
2. **RAM Caching is Essential:** Heavy augmentations crippled dataloading. Implementing a shared RAM cache for the CopyPaste database saved substantial training time.
3. **Architectural Tweaks are Risky:** The theoretical improvements of PIDChad-S (AFFM context gating and FPPM global anchors) drastically worsened the false positive rate on the server. Small local proxy gains do not consistently survive test-set distribution shifts.
4. **Aggressive LSJ Caveats:** The (0.1, 2.0) jitter scale used in EXP-021, combined with the new modules, likely caused the network to hallucinate obstacles in noisy textures, destroying precision.

## 9. Recommendations for Next Steps
- **Revert to PIDNet-S:** Abandon the AFFM/FPPM structural changes and return to the standard PIDNet-S backbone used in EXP-020.
- **Tune Loss Weighting:** With EXP-020 as the new baseline, perform a controlled sweep on `OhemCrossEntropy` class weights (e.g., upweighting the obstacle class) without aggressive LSJ.
- **Explore QAT for Better Architectures:** If INT8 stability remains the bottleneck for stronger models (like SegNeXt), invest in Quantization-Aware Training (QAT) to recover their performance rather than trying to heavily modify PIDNet-S.
- **Refine Local Validation:** Construct a local validation subset or metric that better correlates with the server F1 gap, potentially by heavily weighting small-obstacle metrics.

## 10. Appendix
### INT8 Stability Verification (Local Proxy)
*Based on ONNXruntime static quantization, QUInt8, 50-image calibration.*

| Model / EXP            | FP32 mIoU | INT8 mIoU | ΔmIoU   | FP32 F1 | INT8 F1 | ΔF1     | INT8-Stable |
|------------------------|-----------|-----------|---------|---------|---------|---------|-------------|
| PIDNet-S (EXP-008)     | 93.25%    | 93.43%    | +0.18%  | 76.17%  | 75.66%  | -0.51%  | Yes         |
| PIDNet-S (EXP-020)     | 92.74%    | 92.33%    | -0.41%  | 75.78%  | 76.35%  | +0.56%  | Yes         |
| PIDChad-S (EXP-021)    | 91.64%    | 91.65%    | +0.00%  | 75.94%  | 75.73%  | -0.22%  | Yes         |

All Phase 3 models are INT8-stable (≤0.41% mIoU drop). CNN backbones with BatchNorm remain robust to quantization.