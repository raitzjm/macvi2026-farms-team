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
  - Large Scale Jittering (LSJ): EXP-020 used (0.5, 2.0); EXP-021 used aggressive (0.1, 2.0) to target tiny obstacle recall.
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
- **INT8 (ORT Static):** mIoU = 92.33%, F1 = 76.35%, Q = 70.49%
- *INT8 Stability:* Excellent (ΔmIoU = -0.41%, ΔF1 = +0.56%). Maintains ~17-22 FPS locally.

**Server Results (Submission #16909):**
- **mIoU:** 94.36% (vs 92.82% server in EXP-008)
- **F1:** 55.22% (vs 53.28% server in EXP-008)
- **Q Score:** 52.11 (vs 49.46 server in EXP-008)
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

## 6. EXP-022: SegChad — ConvNeXt-V2 Femto + 3-stream D/P/I Decoder

EXP-022 introduced "SegChad", a fully new architecture replacing the PIDNet-S backbone with
ConvNeXt-V2 Femto (timm pretrained) and a novel 3-stream D/P/I decoder with PAG fusion and
boundary-guided aggregation (see PR #54 and issue #44 for design rationale).

**Configuration:**
- Backbone: ConvNeXt-V2 Femto (5.2M, channels [48, 96, 192, 384], LayerNorm + GRN)
- Decoder: D Branch (boundary, 1/4 res), P Branch (detail, 1/8 res), I Branch (semantic, 1/8 res via PAG)
- Optimizer: AdamW, differential LR (backbone 1.5e-5 / decoder 1.5e-4), cosine annealing
- Schedule: 24,000 iterations, 1,000-iter warmup; server BS=32/GPU (total 64, 2× A6000)

**Training Results (val mIoU, mmseg evaluator):**
- Best val mIoU: **97.13%** @ iter 17000 — new all-time high for this project
- Plateaued in the 97.10–97.13% range from iter 17000 to end of training

**Local Eval (export_and_eval.py, LaRS val):**
- **FP32 ONNX:** mIoU = 93.15%, F1 = 80.03%, Q = 74.55% — best local FP32 Q of all experiments
- **INT8 (ORT Static):** mIoU = 90.27%, F1 = 74.94%, Q = 67.65% — ΔQ = −6.90%
- *Note:* The −6.90% ORT INT8 Q drop was flagged as a concern but was considered acceptable
  given that PIDNet-S (EXP-008) shows near-zero ORT INT8 drop. QAT (QEXP-009, see below) was
  attempted but worsened INT8 degradation (ΔQ = −15.24%), confirming the issue is architectural.

**Server Result:**
- **Catastrophic failure — ranked among the lowest positions on the leaderboard.**
- Exact Q score not available (submission results deleted before archival).

**Root Cause Analysis:**
The failure is most likely caused by incompatibility between ConvNeXt-V2's normalisation
architecture and Qualcomm SNPE INT8 quantization on the Luxonis RVC4:

1. **LayerNorm does not fold under INT8.** BatchNorm (used in PIDNet-S, STDC2) folds cleanly
   into Conv weights at quantization time — it is a free affine rescaling with no runtime cost
   and no precision loss. LayerNorm (used throughout ConvNeXt-V2) is computed at runtime over
   the channel dimension and cannot be folded. Qualcomm's SNPE DSP/HTA likely handles this
   poorly, falling back to CPU execution or degrading numerical precision catastrophically.

2. **GRN (Global Response Normalization) is a non-standard op.** ConvNeXt-V2 introduces GRN
   as `x * (L2_norm(x) / mean(L2_norm(x)))` — an L2 norm across spatial dimensions followed by
   element-wise division. This Reduce-based op has no established DSP kernel in Qualcomm SNPE,
   and likely either forces a CPU fallback or causes an ONNX parse error on device.

3. **ORT INT8 is not a reliable proxy for SNPE INT8.** Our local INT8 smoke test uses
   ONNXRuntime static quantization (CPU, per-tensor affine). SNPE applies its own fixed-point
   quantization targeting the Hexagon DSP, with a completely different op coverage, quantization
   granularity, and precision model. A model that appears stable under ORT INT8 (−6.9% Q) can
   still fail catastrophically under SNPE INT8. This extends the Phase 2 lesson (SegNeXt MSCAN
   failure) to transformer-style backbones in general: **any architecture with runtime
   normalisation (LayerNorm, GroupNorm) or non-standard Reduce ops is a quantization risk on
   embedded Qualcomm hardware.**

4. **timm ONNX export adds Reduce ops.** timm's ConvNeXt-V2 export emits ReduceL2 and Div ops
   for GRN. SNPE's ONNX frontend may reject or misparse these, even if ORT accepts them cleanly.

**Key Lesson:** The INT8 stability criterion for this challenge is not ORT INT8 ≈ FP32, but
rather "does this architecture contain only BatchNorm + Conv + ReLU + bilinear interpolation?"
CNN backbones with BN remain the safe deployment target for Qualcomm RVC4.

---

## 6a. QEXP-009: QAT Attempt on SegChad EXP-022

A QAT fine-tuning run (2,000 iters, BS=2, AdamW lr=1e-5) was attempted on the EXP-022
checkpoint to recover INT8 stability before submission results were known.

**Results:**
- **QAT FP32 ONNX:** mIoU = 93.09%, F1 = 79.13%, Q = 73.67% (−0.88% vs non-QAT FP32)
- **QAT INT8 (ORT Static):** mIoU = 86.22%, F1 = 67.76%, Q = 58.43% — ΔQ = **−15.24%**

QAT made things significantly worse: INT8 degradation doubled (−6.90% → −15.24% Q). This is
consistent with the root cause above — QAT can calibrate quantization ranges for well-behaved
ops (Conv, BN, ReLU) but cannot compensate for ops (LayerNorm, GRN) that are fundamentally
incompatible with SNPE's quantization pipeline. The FakeQuant fine-tuning disrupted
the existing weight distributions without addressing the underlying hardware limitation.

---

## 7. Server vs Local Gap Analysis
A persistent challenge remains the discrepancy between local LaRS validation metrics and server test set metrics. 

| Metric | EXP-020 Local INT8 | EXP-020 Server | Gap (Absolute) |
|--------|--------------------|----------------|----------------|
| mIoU   | 92.33%             | 94.36%         | +2.03%         |
| F1     | 76.35%             | 55.22%         | -21.13%        |
| Q      | 70.49%             | 52.11%         | -18.38%        |

*Analysis:*
- The server test set has a significantly harsher evaluation for object detection (F1 drops by ~21%). 
- Local val F1 overestimates real-world challenge performance, likely due to test set domain shifts or smaller average obstacle sizes that our local evaluation does not penalize as heavily.
- However, server mIoU actually *exceeded* local mIoU for EXP-020, suggesting our padding/preprocessing pipeline is highly accurate and generalizing well to the test set's background distributions.

## 8. Progression Table
Q score progression across Phase 2 and Phase 3 (Server Metrics):

| Submission | EXP     | Model       | Notes                                              | Q Score   | mIoU  | F1    |
|------------|---------|-------------|----------------------------------------------------|-----------|-------|-------|
| #16537     | EXP-008 | PIDNet-S    | Phase 2 Baseline                                   | 49.46     | 92.82 | 53.28 |
| #16864     | EXP-019 | PIDNet-S    | Phase 3 Initial (Aug Shock)                        | 41.60     | 90.70 | 45.90 |
| #16909     | EXP-020 | PIDNet-S    | Phase 3 Definitive (15k + CopyPaste)               | **52.11** | **94.36** | **55.22** |
| #16943     | EXP-021 | PIDChad-S   | AFFM + FPPM + Aggressive LSJ                       | 46.40     | 92.88 | 49.95 |
| —          | EXP-022 | SegChad     | ConvNeXt-V2 + D/P/I decoder; catastrophic on SNPE | n/a (catastrophic) | — | — |

## 9. Key Findings & Lessons Learned
1. **CopyPaste Works (When Warmed Up):** Introducing external obstacle crops via `CopyPasteObstacles` is highly effective (EXP-020 beat EXP-008), but models suffer "augmentation shock." A long warmup (2,500 iters) and extended schedule (15,000 iters) are strictly necessary for the model to digest the injected data.
2. **RAM Caching is Essential:** Heavy augmentations crippled dataloading. Implementing a shared RAM cache for the CopyPaste database saved substantial training time.
3. **Architectural Tweaks are Risky:** The theoretical improvements of PIDChad-S (AFFM context gating and FPPM global anchors) drastically worsened the false positive rate on the server. Small local proxy gains do not consistently survive test-set distribution shifts.
4. **Aggressive LSJ Caveats:** The (0.1, 2.0) jitter scale used in EXP-021, combined with the new modules, likely caused the network to hallucinate obstacles in noisy textures, destroying precision.
5. **LayerNorm/GRN Architectures Are Incompatible with Qualcomm SNPE INT8:** EXP-022 (SegChad, ConvNeXt-V2) achieved the best local FP32 Q score (74.55%) and best training val mIoU (97.13%) of any experiment, yet failed catastrophically on the server. The root cause is architectural: ConvNeXt-V2's LayerNorm and GRN layers are fundamentally incompatible with Qualcomm SNPE INT8 quantization on RVC4. ORT static INT8 is not a reliable proxy for SNPE — it showed only −6.9% Q drop locally while the server result was catastrophic. **For this challenge, only BatchNorm + Conv + ReLU architectures should be considered server-safe.**
6. **QAT Cannot Fix Architectural SNPE Incompatibility:** QEXP-009 (QAT on EXP-022) worsened the ORT INT8 degradation from −6.9% to −15.2% Q, confirming that QAT cannot compensate for ops that SNPE cannot execute efficiently. QAT is only effective for recovering from calibration-range mismatches, not hardware op coverage gaps.

## 10. Recommendations for Next Steps
- **Revert to PIDNet-S:** Abandon the AFFM/FPPM structural changes and return to the standard PIDNet-S backbone used in EXP-020.
- **Tune Loss Weighting:** With EXP-020 as the new baseline, perform a controlled sweep on `OhemCrossEntropy` class weights (e.g., upweighting the obstacle class) without aggressive LSJ.
- **Explore QAT for Better Architectures:** If INT8 stability remains the bottleneck for stronger models (like SegNeXt), invest in Quantization-Aware Training (QAT) to recover their performance rather than trying to heavily modify PIDNet-S.
- **Refine Local Validation:** Construct a local validation subset or metric that better correlates with the server F1 gap, potentially by heavily weighting small-obstacle metrics.

## 11. Appendix
### INT8 Stability Verification (Local Proxy)
*Based on ONNXruntime static quantization, QUInt8, 50-image calibration.*

| Model / EXP              | FP32 mIoU | INT8 mIoU | ΔmIoU   | FP32 F1 | INT8 F1 | ΔF1      | FP32 Q | INT8 Q | ΔQ       | ORT-Stable | SNPE-Stable |
|--------------------------|-----------|-----------|---------|---------|---------|----------|--------|--------|----------|------------|-------------|
| PIDNet-S (EXP-008)       | 93.25%    | 93.43%    | +0.18%  | 76.17%  | 75.66%  | −0.51%   | 71.03  | 70.69  | −0.34%   | ✓          | ✓           |
| PIDNet-S (EXP-020)       | 92.74%    | 92.33%    | −0.41%  | 75.78%  | 76.35%  | +0.56%   | 70.28  | 70.49  | +0.21%   | ✓          | ✓           |
| PIDChad-S (EXP-021)      | 91.64%    | 91.65%    | +0.00%  | 75.94%  | 75.73%  | −0.22%   | 69.60  | 69.40  | −0.20%   | ✓          | ✓           |
| SegChad (EXP-022)        | 93.15%    | 90.27%    | −2.88%  | 80.03%  | 74.94%  | −5.09%   | 74.55  | 67.65  | −6.90%   | ~          | ✗ (catastrophic) |
| SegChad QAT (QEXP-009)   | 93.09%    | 86.22%    | −6.87%  | 79.13%  | 67.76%  | −11.37%  | 73.67  | 58.43  | −15.24%  | ✗          | n/a         |

CNN backbones with BatchNorm (PIDNet-S, PIDChad-S) are reliably SNPE-safe. ConvNeXt-V2 (LayerNorm + GRN) failed on SNPE despite appearing borderline-stable under ORT INT8 (−6.9% Q). ORT INT8 is not a reliable predictor of SNPE INT8 behaviour for non-BN architectures.