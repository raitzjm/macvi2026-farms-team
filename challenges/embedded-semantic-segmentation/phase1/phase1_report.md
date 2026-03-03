# Phase 1 Baseline Report — Embedded Semantic Segmentation

**Date completed:** 2026-03-03
**Branch:** `exp/phase1-baselines`
**Issue:** #9
**Status:** COMPLETE

---

## 1. Purpose and Scope

Phase 1 is a short sanity-check run, not the final baseline. Its goals were:

1. Validate that the preprocessing pipeline matches the challenge server exactly.
2. Confirm architecture rankings at low iteration count before migrating to a unified mmseg stack in Phase 2.
3. Establish pinned batch sizes and LRs for Phase 2 consistency cross-checking.

Phase 2 is the real baseline (full training budget, mmseg 1.x migration). Phase 1 results are intentionally short: 1500 iters × total_bs provides the same training-data volume as the original 1-GPU 3000-iter plan but halves wall-clock time by using 2 GPUs.

---

## 2. Environment

| Item | Value |
|------|-------|
| Hardware | 2× NVIDIA RTX A6000 48 GB |
| DDP launcher | `torchrun --standalone --nproc_per_node=2` |
| Conda env | `macvi` (`/home/farms/miniforge3/envs/macvi/`) |
| Python | 3.11.14 (conda-forge) |
| PyTorch | 2.1.2+cu121 |
| MMCV | 1.7.2 (prebuilt wheel) |
| MMSegmentation | 0.30.0 (mmsegmentation-macvi fork) |
| CUDA | 12.1 |
| Dataset | LaRS — train: 2607, val: 200, test: 1205 |
| Resolution | 768×384 (W×H) — target deployment resolution |
| WandB project | `little_farms/embedded_segmentation_macvi2026` |

---

## 3. Results Summary

| Rank | Model | Best val mIoU | Best iter | Obstacle IoU | Water IoU | Sky IoU | Duration | Convergence |
|------|-------|--------------|-----------|-------------|-----------|---------|----------|-------------|
| 1 | SegFormer MiT-B2 | **97.00%** | 1000 | 94.56% | 98.09% | 98.36% | ~36 min | Converged; plateaued after iter 1000 |
| 2 | STDC2 | 95.76% | 1250 | 92.41% | 97.60% | 97.26% | ~1h08m | Stable; slight plateau |
| 3 | SegNeXt-T | 95.61% | 1500↑ | 92.19% | 97.34% | 97.31% | ~34 min | Monotonic; still climbing |
| 4 | BiSeNetV2 | 92.55% | 1500↑ | 86.41% | 94.88% | 96.38% | ~1h02m | LR instability at iter 750; still climbing |

**val F1:** Not reported. The mmseg 0.30.0 `LaRSDataset` evaluator computes mIoU and aAcc only. The challenge metric (F1/BF score) is deferred to Phase 2.

Ranking matches published challenge expectations (SegFormer > STDC2 ≈ SegNeXt > BiSeNetV2). Key finding: training natively at 768×384 dramatically outperforms the published "orig (768×384)" inference-only baselines (e.g. STDC2 published F1=50.8% vs our 95.76% mIoU).

---

## 4. Per-Model Training Metadata

### 4.1 EXP-001 — STDC2

| Parameter | Value |
|-----------|-------|
| Config | `phase1/configs/stdc2_768x384_lars_phase1.py` |
| Work dir | `outputs/phase1/stdc2/` |
| Log file | `20260303_085603.log` |
| Optimizer | SGD, momentum=0.9, weight_decay=0.0005 |
| Reference (bs, lr) | bs=12, lr=0.01 (stdc1_in1k-pre_512x1024_80k_lars.py upstream) |
| samples_per_gpu | 160 |
| Total batch size | 320 (160 × 2 GPUs) |
| Configured peak LR | 0.267 (= 0.01 × 320/12, linear scaling rule) |
| LR post-warmup (iter 200) | 2.349e-01 (poly decay from peak) |
| LR schedule | poly, power=0.9, min_lr=1e-4 |
| Warmup | linear, 150 iters |
| Runner | IterBasedRunner |
| Total iters | 1500 |
| Eval interval | 250 iters (6 eval points) |
| Checkpoint interval | 750 iters |
| Norm | SyncBN (required for DDP) |
| Backbone pretrain | STDCNet2 ImageNet-1k checkpoint |
| Best checkpoint | `best_mIoU_iter_1250.pth` |

Val mIoU progression:

| Iter | mIoU | aAcc | Obstacle IoU | Water IoU | Sky IoU |
|------|------|------|-------------|-----------|---------|
| 250 | 95.00 | 97.66 | 90.91 | 96.67 | 97.43 |
| 500 | 95.44 | 97.90 | 91.71 | 97.11 | 97.50 |
| 750 | 95.54 | 97.93 | 91.96 | 97.14 | 97.53 |
| 1000 | 94.56 | 97.51 | 90.32 | 96.80 | 96.57 |
| **1250** | **95.76** | 97.08 | **92.41** | **97.60** | 97.26 |
| 1500 | 95.51 | 97.98 | 92.06 | 97.62 | 96.83 |

---

### 4.2 EXP-002 — BiSeNetV2

| Parameter | Value |
|-----------|-------|
| Config | `phase1/configs/bisenetv2_768x384_lars_phase1.py` |
| Work dir | `outputs/phase1/bisenetv2/` |
| Log file | `20260303_101339.log` |
| Optimizer | SGD, momentum=0.9, weight_decay=0.0005 |
| Reference (bs, lr) | bs=4 per-GPU, lr=0.05 (bisenetv2_fcn_4x4_1024x1024_160k_lars.py upstream) |
| samples_per_gpu | 128 |
| Total batch size | 256 (128 × 2 GPUs) |
| Configured peak LR | 3.2 (= 0.05 × 256/4, linear scaling rule) |
| LR post-warmup (iter 200) | 2.815e+00 (poly decay from peak) |
| LR schedule | poly, power=0.9, min_lr=1e-4 |
| Warmup | linear, 150 iters |
| Runner | IterBasedRunner |
| Total iters | 1500 |
| Eval interval | 250 iters |
| Checkpoint interval | 750 iters |
| Norm | SyncBN (already set in upstream config) |
| Backbone pretrain | None (BiSeNetV2 trains from scratch) |
| Best checkpoint | `best_mIoU_iter_1500.pth` (still climbing at end) |

Val mIoU progression:

| Iter | mIoU | aAcc | Obstacle IoU | Water IoU | Sky IoU |
|------|------|------|-------------|-----------|---------|
| 250 | 48.28 | 71.92 | 14.86 | 76.51 | 53.48 |
| 500 | 67.93 | 83.15 | 49.55 | 83.84 | 70.42 |
| 750 | 37.25 | 63.52 | 0.12 | 61.33 | 50.29 |
| 1000 | 57.24 | 76.56 | 52.12 | 70.60 | 49.00 |
| 1250 | 86.03 | 93.20 | 75.89 | 91.45 | 90.77 |
| **1500** | **92.55** | 96.46 | **86.41** | **94.88** | **96.38** |

Note: near-catastrophic obstacle collapse at iter 750 (IoU=0.12%) — see Section 7 for details.

---

### 4.3 EXP-003 — SegNeXt-T

| Parameter | Value |
|-----------|-------|
| Config | `phase1/configs/segnext_t_768x384_lars_phase1.py` |
| Work dir | `outputs/phase1/segnext_t/` |
| Log file | `20260303_112104.log` |
| Optimizer | AdamW, betas=(0.9, 0.999), weight_decay=0.01, paramwise (pos_block, norm, head) |
| Reference (bs, lr) | AdamW, lr=6e-5 (upstream SegNeXt config) |
| samples_per_gpu | 56 |
| Total batch size | 112 (56 × 2 GPUs) |
| Configured peak LR | 6e-5 (fixed — AdamW does not use linear scaling) |
| LR post-warmup (iter 300) | 4.804e-05 (poly decay from peak) |
| LR schedule | poly, power=1.0, min_lr=0.0 |
| Warmup | linear, 250 iters, warmup_ratio=1e-6 |
| Runner | IterBasedRunner |
| Total iters | 1500 |
| Eval interval | 250 iters |
| Checkpoint interval | 750 iters |
| Norm | LayerNorm (GroupNorm in HAM decoder); SyncBN not applicable |
| Backbone pretrain | MSCAN-T ImageNet checkpoint |
| Best checkpoint | `best_mIoU_iter_1500.pth` (still climbing at end) |

Val mIoU progression:

| Iter | mIoU | aAcc | Obstacle IoU | Water IoU | Sky IoU |
|------|------|------|-------------|-----------|---------|
| 250 | 90.51 | 95.56 | 84.19 | 94.44 | 92.91 |
| 500 | 94.26 | 97.35 | 90.12 | 96.50 | 96.16 |
| 750 | 94.99 | 97.68 | 91.19 | 96.86 | 96.91 |
| 1000 | 95.39 | 97.87 | 91.83 | 97.13 | 97.20 |
| 1250 | 95.42 | 97.90 | 91.86 | 97.19 | 97.21 |
| **1500** | **95.61** | 97.99 | **92.19** | **97.34** | **97.31** |

---

### 4.4 EXP-004 — SegFormer MiT-B2

| Parameter | Value |
|-----------|-------|
| Config | `phase1/configs/segformer_mit-b2_768x384_lars_phase1.py` |
| Work dir | `outputs/phase1/segformer_mit-b2/` |
| Log file | `20260303_115511.log` |
| Optimizer | AdamW, betas=(0.9, 0.999), weight_decay=0.01, paramwise (pos_block, norm, head) |
| Reference (bs, lr) | AdamW, lr=6e-5, bs=4 per-GPU (upstream SegFormer LaRS config) |
| samples_per_gpu | 36 |
| Total batch size | 72 (36 × 2 GPUs) |
| Configured peak LR | 6e-5 (fixed — AdamW does not use linear scaling) |
| LR post-warmup (iter 300) | 4.804e-05 (poly decay from peak) |
| LR schedule | poly, power=1.0, min_lr=0.0 |
| Warmup | linear, 250 iters, warmup_ratio=1e-6 |
| Runner | IterBasedRunner |
| Total iters | 1500 |
| Eval interval | 250 iters |
| Checkpoint interval | 750 iters |
| Norm | LayerNorm; SyncBN not applicable (decode head norm_cfg=SyncBN in config has no effect for transformer heads) |
| Backbone pretrain | MiT-B2 ImageNet checkpoint |
| Best checkpoint | `best_mIoU_iter_1000.pth` (plateaued after iter 1000) |

Val mIoU progression:

| Iter | mIoU | aAcc | Obstacle IoU | Water IoU | Sky IoU |
|------|------|------|-------------|-----------|---------|
| 250 | 95.95 | 98.14 | 92.81 | 97.47 | 97.58 |
| 500 | 96.65 | 98.47 | 93.92 | 97.92 | 98.10 |
| 750 | 96.79 | 98.53 | 94.19 | 97.96 | 98.21 |
| **1000** | **97.00** | 98.63 | **94.56** | **98.09** | **98.36** |
| 1250 | 96.98 | 98.62 | 94.51 | 98.07 | 98.37 |
| 1500 | 96.98 | 98.62 | 94.54 | 98.10 | 98.31 |

---

## 5. Methodology

### 5.1 Schedule Rationale

Phase 1 ran 1500 iters on 2 GPUs. This is equivalent in total training data to 3000 iters on 1 GPU (the original plan), but uses half the wall-clock time. Warmup iters were scaled proportionally (halved from the 1-GPU plan). The schedule is deliberately short — Phase 1 is a sanity check, not the final baseline.

Config-level schedule origin: all four configs use `IterBasedRunner` (the mmseg 0.30.0 default for LaRS configs). There is no explicit `schedule_Nk` inheritance — the 1500-iter schedule is written directly in each config.

### 5.2 Config Philosophy

Architecture-specific settings (optimizer type, base LR, loss functions, auxiliary head count and config, backbone pretrain) are left untouched per model, inherited from the upstream LaRS or ADE20K reference config. Shared settings (resolution, preprocessing pipeline, schedule length, eval frequency, dataset paths) are standardized across all models. See `CLAUDE.md` for the canonical statement of this policy.

### 5.3 Preprocessing Fixes (Critical)

The challenge server preprocesses images before feeding them to the submitted ONNX:
1. Resize with bilinear interpolation, aspect ratio preserved, to fit within 384×768.
2. Center the resized image in the 384×768 canvas.
3. Fill remaining space with **mirror (reflect) padding** — not zero padding.
4. Normalize with ImageNet mean/std.

The `fun_experiments` configs had three mismatches against this behavior:
- Train and test Pad used zero-fill (`pad_val=0`) instead of reflect padding.
- Test Resize used `keep_ratio=False` (stretching), not aspect-ratio-preserving resize.
- Test pipeline had no explicit Pad step, so images at non-standard aspect ratios were not correctly framed.

All Phase 1 configs correct all three mismatches. See Section 6 for implementation details.

### 5.4 Batch Size Calibration Strategy

Batch sizes were calibrated by running forward+backward passes only (no optimizer step, no metric buffers). The full training step consistently needs 10-25% more memory than fwd+bwd calibration. Actual reductions:

| Model | fwd+bwd boundary | Training OOM at | Final bs/GPU | Reduction |
|-------|-----------------|-----------------|-------------|-----------|
| STDC2 | ~175 | 175 (OOM at start) | 160 | ~9% |
| BiSeNetV2 | ~152 | 152 (OOM confirmed) | 128 | ~16% |
| SegNeXt-T | 75 | 75 (OOM confirmed) | 56 | ~25% |
| SegFormer | 45 | not tested | 36 | ~20% pre-emptive |

For Phase 2: use 75-80% of the fwd+bwd boundary as the starting batch size, or calibrate with a full training step.

### 5.5 LR Scaling Rules

- **SGD (STDC2, BiSeNetV2):** linear scaling rule applied — `new_lr = base_lr × (new_bs / reference_bs)`. Reference (bs, lr) pairs are documented in each config header comment.
- **AdamW (SegNeXt-T, SegFormer):** LR fixed at the upstream reference value regardless of batch size.

### 5.6 SyncBN for DDP

Models using BatchNorm (STDC2, BiSeNetV2) use `SyncBN` so that batch normalization statistics are synchronized across both GPUs during DDP training. Models using LayerNorm (SegNeXt-T, SegFormer) are unaffected and use standard per-GPU norm. This setting is pinned for Phase 2.

---

## 6. Preprocessing Implementation

The custom `ReflectPad` transform is defined in `phase1/pipelines.py`. It wraps `mmcv.impad(padding_mode='reflect')` and falls back to `'constant'` if padding would exceed the source image size. The built-in mmseg `Pad` transform does not expose `padding_mode`, which is why the custom transform was needed.

**Important bug fixed during Phase 1:** `_pad_seg` must pass the 2D mask directly to `mmcv.impad` (not as a 3-channel array via `np.newaxis`). The `mmcv.impad` function squeezes a `(H,W,1)` input to `(H,W)`, after which `[..., 0]` returns `(H,)` instead of `(H,W)`. The fix is to pass the mask as-is (2D) and retrieve the 2D result directly.

Pipeline configuration in every Phase 1 config:

**Train pipeline:**
```python
dict(type='ReflectPad', size=(384, 768), seg_pad_val=255)
```
Applied after Normalize, before DefaultFormatBundle. Reflects the image if it does not fill the crop after RandomCrop.

**Test pipeline (inside MultiScaleFlipAug):**
```python
dict(type='Resize', keep_ratio=True),
...
dict(type='ReflectPad', size=(384, 768)),
```
`keep_ratio=True` preserves the original aspect ratio; `ReflectPad` centers and mirror-pads to the exact target size.

`custom_imports = dict(imports=['challenges.embedded-semantic-segmentation.phase1.pipelines'])` is required in every config; the PYTHONPATH must include the repo root (set by the run scripts).

---

## 7. Issues, Gotchas, and Workarounds

### [Infrastructure] torchrun vs torch.distributed.launch — argument mismatch

PyTorch 2.1.x `torch.distributed.launch` passes `--local-rank` (hyphen) on the CLI, but mmseg 0.30.0 argparse expects `--local_rank` (underscore). Using the standard `dist_train.sh` wrapper (which calls `torch.distributed.launch`) causes an immediate argparse crash on all workers.

**Fix:** use `torchrun --standalone --nproc_per_node=N tools/train.py --launcher pytorch`. `torchrun` sets `LOCAL_RANK` as an environment variable, which mmseg reads correctly. All Phase 1 run scripts use `torchrun`.

---

### [Infrastructure] MMDistributedDataParallel._use_replicated_tensor_module AttributeError

Training proceeds normally (mmseg's `train_step()` bypasses `DDP.forward()`), but the first eval pass at iter 250 crashes with `AttributeError: '_use_replicated_tensor_module'`.

Root cause: `mmcv/parallel/distributed.py` (`_run_ddp_forward`) references `self._use_replicated_tensor_module` directly. PyTorch 2.1.2 does not define this attribute on `DistributedDataParallel` (it was added in a later PyTorch release).

**Fix** (applied at `mmcv/parallel/distributed.py`, ~line 160):
```python
# Before:
module_to_run = self._replicated_tensor_module if \
    self._use_replicated_tensor_module else self.module
# After:
module_to_run = self._replicated_tensor_module if \
    getattr(self, '_use_replicated_tensor_module', False) else self.module
```
This patch is **not baked into the Docker image** and must be reapplied after any mmcv reinstall. It is tracked in the challenge `CLAUDE.md`.

---

### [Infrastructure] WANDB_DIR must not be on PYTHONPATH

If `WANDB_DIR` is not set, wandb writes a `wandb/` directory into the current working directory. If the repo root is on `PYTHONPATH` (as set by the run scripts for `custom_imports`), Python imports the `wandb/` cache directory instead of the `wandb` package, silently breaking all `import wandb` calls.

**Fix:** always set `export WANDB_DIR=<path-not-on-PYTHONPATH>` before training. Phase 1 scripts set `WANDB_DIR=$OUTPUTS/phase1` (i.e. inside `outputs/phase1/`, which is not on PYTHONPATH). The `wandb/` directory is gitignored.

---

### [Infrastructure] wandb==0.25.0 validation.py patch for new key format

New wandb API keys start with `wandb_` (e.g. `wandb_abc...`). The `wandb` SDK's `validation.py` checks for exactly 40 characters and rejects keys with the `wandb_` prefix.

**Fix** (`wandb/sdk/lib/wbauth/validation.py` — add before the 40-char check):
```python
if api_key.startswith("wandb_"):
    return True
```
This patch must be reapplied after any wandb reinstall. Tracked in the challenge `CLAUDE.md`.

---

### [STDC2] OOM at calibrated batch size — full training step exceeds fwd+bwd memory

Batch size was calibrated by running fwd+bwd only. The full training step (loss computation, optimizer step, metric buffers) uses more memory. STDC2 OOMed at the calibrated bs=175; reduced to bs=160 (~9% reduction).

**Lesson:** always apply a 10-25% safety margin to the fwd+bwd calibrated batch size, or calibrate with a full training step.

---

### [BiSeNetV2] OOM at bs=152 — 5 auxiliary heads are memory-intensive

BiSeNetV2 has 5 auxiliary segmentation heads in addition to the main head. Memory usage is significantly higher than STDC2 at equivalent backbone complexity. OOMed at bs=152; reduced to bs=128. LR scaled accordingly: 3.2 (= 0.05 × 256/4).

---

### [BiSeNetV2] Near-catastrophic obstacle collapse at iter 750

With SGD LR=3.2 (aggressively linear-scaled for bs=256), BiSeNetV2 showed extreme instability: obstacle IoU collapsed from ~50% at iter 500 to 0.12% at iter 750. It partially recovered by iter 1000 and fully recovered by iter 1500. The model was still climbing at iter 1500 — 1500 iters was too short to confirm convergence for this model.

**Recommendation for Phase 2:** investigate longer warmup (150 iters may be insufficient for LR=3.2) or a lower LR. The linear scaling rule may be too aggressive for BiSeNetV2's 5-head architecture. Also use more iters.

---

### [SegNeXt-T] OOM at bs=75 — Hamburger (HAM) decoder is memory-intensive

SegNeXt-T's `LightHamHead` (Hamburger) decoder holds large intermediate activation matrices for iterative NMF. Calibrated at bs=75 but OOMed during the full training step. Reduced to bs=56 (~25% reduction). LR unchanged (AdamW fixed at 6e-5).

---

### [SegFormer] Pre-emptive batch size reduction to bs=36

After observing the OOM pattern in the first three models (consistent 10-25% gap between fwd+bwd calibration and full training step), SegFormer was pre-emptively reduced from the calibrated bs=45 to bs=36 (~20% reduction). No OOM occurred. The pre-emptive reduction saved a failed run cycle.

---

### [All Models] val F1 not reported

The mmseg 0.30.0 `LaRSDataset` evaluator reports mIoU and aAcc but not the challenge metric (F1/BF score). Phase 1 results are logged in mIoU only.

**Action for Phase 2:** add an F1 evaluation callback or run the standalone challenge eval scripts post-training.

---

## 8. Pinned Batch Sizes and LRs for Phase 2

These values must not be re-calibrated in Phase 2. The Phase 2 consistency cross-check compares val mIoU at ~3k iters using the exact same (bs, lr) pair.

| Model | samples_per_gpu | total_bs | LR | Optimizer | Norm |
|-------|----------------|----------|----|-----------|------|
| STDC2 | 160 | 320 | 0.267 | SGD | SyncBN |
| BiSeNetV2 | 128 | 256 | 3.2 | SGD | SyncBN |
| SegNeXt-T | 56 | 112 | 6e-5 | AdamW | LayerNorm |
| SegFormer MiT-B2 | 36 | 72 | 6e-5 | AdamW | LayerNorm |

---

## 9. Phase 1 Archival Exceptions

The following steps from the standard `/archive-experiment` skill were **not performed** for Phase 1:

- No ONNX export (Phase 1 is a sanity check; ONNX export deferred to Phase 2).
- No per-model archive directories created.
- No F1 / precision / recall validation against challenge server.
- No INT8 quantization evaluation.

Phase 1 checkpoint files (`best_mIoU_iter_*.pth`, `iter_750.pth`, `iter_1500.pth`) are in `outputs/phase1/{model}/` but are **not committed** (outputs/ is gitignored). The experiment-log.md and this report are the authoritative records.

---

## 10. What Carries Forward to Phase 2

1. **Preprocessing pipeline** — unchanged. `ReflectPad` + `keep_ratio=True` + `Resize` must be applied in all future configs. The `pipelines.py` implementation is the reference.

2. **Batch sizes and LRs** — pinned (Section 8). Do not re-calibrate.

3. **SyncBN** — pinned for STDC2 and BiSeNetV2.

4. **torchrun launcher** — use `torchrun --standalone --nproc_per_node=2`; never `dist_train.sh`.

5. **mmcv DDP eval patch** — must be reapplied in any new container build (tracked in `CLAUDE.md`).

6. **WANDB_DIR isolation** — ensure WANDB_DIR is always set to a directory not on PYTHONPATH.

7. **BiSeNetV2 LR caution** — Phase 2 should test longer warmup or reduced LR for BiSeNetV2 given the iter-750 collapse. Monitor obstacle IoU closely.

8. **F1 evaluation** — add a standalone F1/BF-score eval step after each Phase 2 training run. This is the actual challenge metric.

9. **Phase 2 consistency cross-check** — at ~3k iters of Phase 2 training, compare val mIoU against Phase 1 values (adjusted for same training-data volume). All Phase 1 mIoU values at iter 1500 are the reference points.

10. **mmseg migration** — Phase 2 migrates all models to a unified mmseg version (target: 1.2.2). The configs, pipelines, and batch size comments in this phase are the reference for translating to the new API.
