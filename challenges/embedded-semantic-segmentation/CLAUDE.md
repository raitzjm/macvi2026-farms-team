# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (obstacle=0, water=1, sky=2) on LaRS dataset.
We train and export an ONNX model. The challenge platform handles INT8
quantization (Qualcomm SNPE) and deployment (Luxonis RVC4) - we do NOT do this.
Our job: train model -> export ONNX -> submit ONNX to challenge server.
Winner: highest F1 detection score (FPS tiebreaker). Must achieve >=30 FPS at 768x384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Dataset
- LaRS: /home/farms/data/lars_dataset/ (train: 2605, val: 198, test: 1205 images)
- Test set has NO segmentation masks
- External data allowed but must be disclosed
- img_dir: {split}/images/, ann_dir: {split}/semantic_masks/
- Symlink tree at /home/farms/lars_linked/ must be rebuilt each container session (not persistent)

## Workspace
- mmsegmentation-macvi/: copied from github.com/lojzezust/mmsegmentation-macvi
- Training resolution: 768x384 (CRITICAL - always train at target resolution)

## Key Dependencies
- Conda env `macvi` at /home/farms/miniforge3/envs/macvi/ (on PATH by default)
- Python 3.11 (conda-forge), CUDA 12.1
- PyTorch 2.1.2+cu121, mmcv-full 1.7.2 (prebuilt wheel — no source build)
- mmsegmentation 0.30.0 installed non-editable at /opt/macvi-env/mmsegmentation-macvi/
- All baked into the Docker image — no post-start setup needed
- Required patches baked in: setuptools<70, numpy<2, mmcv scatter patch (_functions.py)
- **NOT baked in (must reapply after reinstall)**:
  - `mmcv/parallel/distributed.py` line ~160: `self._use_replicated_tensor_module` →
    `getattr(self, '_use_replicated_tensor_module', False)` (DDP eval fix for PyTorch 2.1.2)
  - `wandb/sdk/lib/wbauth/validation.py`: add early-return for `wandb_` prefix before the
    40-char check (required for new wandb API key format starting with `wandb_`)

## WandB Experiment Tracking

- **Package**: `wandb==0.25.0` installed in `macvi` env (`pip install wandb==0.25.0`)
- **API key**: stored at `~/data/wandb_key.txt` (read-only mount — never commit)
  - Key format starts with `wandb_` (new format) — requires the validation.py patch above
  - Load via `source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh`
    or inline: `export WANDB_API_KEY=$(tr -d '[:space:]' < ~/data/wandb_key.txt)`
- **Entity/Project**: `little_farms` / `embedded_segmentation_macvi2026`
- **Run naming**: `{phase}_{model_name}` — e.g. `phase1_stdc2`, `phase2_segformer_mit-b2`
- **Failed runs**: if a training run fails and must be restarted, **delete the failed run
  from WandB first** (`gh` or the WandB UI) before relaunching, so the project stays clean.
  Use `wandb run delete <run-id>` or delete via the WandB web UI under the project page.
- **WANDB_DIR env var**: MUST be set to a directory that is NOT on `PYTHONPATH`
  (e.g. `outputs/phase1`). If `wandb/` ends up on `PYTHONPATH`, Python imports the wandb
  cache directory instead of the wandb package — silently breaking all imports. The run
  scripts set this correctly; check if adding new scripts.
- **Hook config** (mmcv 1.7.2 `WandbLoggerHook` — logs scalars only, no images):
  ```python
  log_config = dict(
      interval=50,
      hooks=[
          dict(type='TextLoggerHook'),
          dict(type='WandbLoggerHook',
               init_kwargs=dict(
                   project='embedded_segmentation_macvi2026',
                   entity='little_farms',
                   name='<run_name>',
               )),
      ])
  ```
- **Image logging**: mmcv 1.7.2 `WandbLoggerHook` does NOT support image logging (scalars
  only). Upgrade to mmseg 1.x / mmengine in Phase 2 for `WandbVisBackend` +
  `SegVisualizationHook` which logs prediction overlays natively.

## Preprocessing Requirements (CRITICAL — must match challenge server exactly)

The challenge server preprocesses images before feeding them to our ONNX:
1. Resize with **bilinear interpolation, aspect ratio preserved** → fits within 384×768
2. **Center** the resized image in the 384×768 canvas
3. Fill remaining space with **mirror padding** (not zero padding)
4. Normalize with ImageNet mean/std (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

All three points must be matched in training pipelines. The fun experiments got this wrong
(used keep_ratio=False + zero padding), causing a train/test mismatch.

**Required mmseg pipeline settings for every Phase 1+ config:**
- Test pipeline: `dict(type='Resize', keep_ratio=True)` inside MultiScaleFlipAug
- Train + test Pad op: `dict(type='Pad', size=crop_size, padding_mode='reflect')`
  (`padding_mode='reflect'` is mirror padding; default `pad_val=0` is wrong)
- Centering is handled automatically by mmseg when keep_ratio=True + Pad are combined

## Config Philosophy (applies to all phases)

Architecture-specific settings — optimizer type, base LR, loss functions, head config,
auxiliary heads — are left **untouched** per model (inherited from the upstream LaRS or
ADE20K reference config). These are tuned for each architecture and must not be
homogenised across models.

Shared settings — resolution, preprocessing pipeline, schedule length, eval frequency,
dataset paths — are **standardised** across all models in each phase.

**Runner type**: all mmseg-macvi LaRS configs use `IterBasedRunner` (iter-based, not
epoch-based). This is the mmseg 0.30.0 default and applies to all Phase 1 models.

**LR scaling when changing batch size** from the upstream reference:
- SGD optimisers (STDC2, BiSeNetV2): apply the linear scaling rule —
  `new_lr = base_lr × (new_bs / reference_bs)`. Skipping this for large batch
  size increases leaves SGD steps too small and weakens results artificially.
- AdamW optimisers (SegNeXt-T, SegFormer): keep LR fixed regardless of batch size.
- Reference values and rationale are documented in issue #9.
- Document both batch_size and lr in a comment in the config so Phase 2 can replicate
  the exact (bs, lr) pair.

**SyncBN for DDP**: models using BatchNorm must switch to `SyncBN` when training with DDP
(multiple GPUs), so BN statistics are synchronized across GPUs rather than computed
independently per device. This applies to STDC2 and BiSeNetV2. Models using LayerNorm
(SegNeXt-T, SegFormer) are unaffected. Set `norm_cfg = dict(type='SyncBN', requires_grad=True)`
in the model config. This setting is pinned — carry it into Phase 2.

**Batch size strategy**: at the start of Phase 1, calibrate the maximum batch size that
fits on the training GPU without OOM. Document this value in the config comment. This
becomes the pinned batch size — reuse the exact same value in Phase 2 for the consistency
check to remain valid. Do not change it between phases.

## Experiment Phases
Phase 1: Short validation runs (~1.5k iters, ~1.25h per model) — validate correct preprocessing
          and confirm architecture rankings. Models: STDC2, BiSeNetV2, SegNeXt-T, SegFormer MiT-B2.
          Purpose: establish a reference baseline before the mmseg migration in Phase 2,
          so Phase 2 translated configs can be cross-checked for consistency at the ~1.5k iter mark.
          1.5k iters chosen for 2-GPU DDP: 1500 × total_bs ≈ 3000 × (total_bs/2) — same training
          data as the original 1-GPU 3k plan, but half the wall-clock time (~1.25h/model, ~5h total).
          Eval every 250 iters → 6 evaluation points. Phase 2 is the real baseline; Phase 1 is a sanity check.
          See issue #9 for full details, batch size reference values, and smoke test workflow.
          Once Phase 1 is complete, methodology and rationale will be captured in
          `phase1/phase1_report.md` — that becomes the permanent team reference.
Phase 2: Full training runs — migrate all models to a single unified mmseg version
          (target: 1.2.2) for consistency. All serious experiments from here onward
          use this unified stack. New models may be introduced here.
          Consistency check: compare val mIoU at the ~3k iter mark against Phase 1 logs.
          Phase 2 eval frequency is independent (e.g. every 5k iters) — only one early
          checkpoint near iter 3k is needed for the cross-check.
          Batch size must match Phase 1 exactly — do not re-calibrate.
          Models: SegFormer MiT-B2, SegNeXt-T, STDC2, PIDNet-S, BiSeNetV2 (original 5).
          Added late to Phase 2: SegNeXt-S (EXP-010), UNet-s5-d16 (EXP-011).
            SegNeXt-S: larger backbone than T (AdamW, LR=6e-5 fixed, calibrate BS).
            UNet-s5-d16: DeepLabv3 head + CE+Dice loss, SGD (linear-scale LR), no pretrain.
            Reference configs: segnext_s/unet_s5d16_768x384_lars_phase2.py in phase2/configs/.
Phase 3: Training tricks — 4-class training, loss tuning, BoundaryLoss, architecture experiments
Phase 4: External data — conditional, only if distribution analysis shows gaps

Phases can be modified, reordered, added, or skipped as the project evolves.

## ONNX Export Requirements
- Output: argmaxed class map, shape 1×1×384×768, dtype int64
- Opset version: 13 (server uses SNPE which requires opset 13)
- See fun_experiments/{model}/export_onnx.py for reference implementations

## Output Directory Convention

Training live outputs (checkpoints, logs) go to:
  `challenges/embedded-semantic-segmentation/outputs/{phase}/{model_name}/`

Pass this as `--work-dir` to the train script. Examples:
  `outputs/phase1/stdc2/`
  `outputs/phase2/stdc2/`

This keeps phases isolated and prevents checkpoint collisions. The `outputs/` tree
is gitignored — only curated artifacts are committed (see Phase 1 Artifact Archival).

## Artifact Archival

After each training run completes, use the `/archive-experiment` skill.
It handles ONNX export, pth vs ONNX precision validation, artifact commit, and
experiment-log.md update. INT8 quantization evaluation is deferred to Phase 2.

### Phase 2 committed checkpoints

Best checkpoints for INT8-stable models are committed to Git LFS under
`phase2/artifacts/{model}/best_mIoU_iter_NNNN.pth`.

| Model | Checkpoint | Val mIoU | Val F1 | Q | INT8-stable? |
|-------|-----------|----------|--------|---|--------------|
| pidnets (EXP-008) | best_mIoU_iter_4750.pth | 93.25% | 76.17% | 71.03% | ✓ |
| pidnetm (EXP-016) | best_mIoU_iter_3250.pth | 92.33% | 72.82% | 67.23% | ✓ |
| stdc2 (EXP-007) | best_mIoU_iter_2750.pth | 90.69% | 68.92% | 62.50% | ✓ |
| bisenetv2 (EXP-009) | best_mIoU_iter_6000.pth | 91.09% | 69.06% | 62.91% | ✓ |
| segnext_t_uperhead (EXP-015) | best_mIoU_iter_4250.pth | 92.48% | 80.21% | 74.18% | ✗ (−78%) |
| segnext_s (EXP-010) | best_mIoU_iter_4750.pth | 92.40% | 72.33% | 66.83% | ✗ (−82%) |
| segnext_t (EXP-006) | best_mIoU_iter_4250.pth | 91.79% | 71.74% | 65.85% | ✗ (−67%) |

SegNeXt checkpoints (EXP-006, EXP-010, EXP-015) are committed to LFS at the owner's
explicit request for future reference, despite INT8 catastrophic failure (MSCAN backbone
large-kernel depthwise attention is the root cause — not fixable by head replacement alone).
Do not commit SegNeXt ONNXes to LFS unless explicitly requested.

EXP-015 (MSCAN-T + UPerHead + BoundaryLoss): FP32 Q=74.18% (best of all models, highest F1 at 80.21%).
Replacing LightHamHead with UPerHead improves FP32 F1 by +8.5% but does NOT fix INT8 sensitivity.

See `eval_metrics.md` for full Phase 2 eval results.

## Current Status
Phase 1 COMPLETE (branch: exp/phase1-baselines). All 4 models done. See experiment-log.md.
Final mIoU: SegFormer 97.00%, STDC2 95.76%, SegNeXt-T 95.61%, BiSeNetV2 92.55%.
Models running sequentially: STDC2 → BiSeNetV2 → SegNeXt-T → SegFormer MiT-B2.
See issue #9 for task tracking and run decisions.

## Published Baselines (from challenge website)
Three variants per model:
- orig: trained AND evaluated at original 2048×1024 resolution
- orig (768×384): same weights, inference-only at 768×384 (no retraining)
- quantized: INT8 quantized via Qualcomm SNPE, runs on device (Luxonis RVC4) at 768×384

The challenge platform does quantization — we submit ONNX, they quantize.
FPS requirement: ≥30 FPS. Models below threshold cannot win (shown for reference).

| Model                   | orig F1 | orig (768×384) F1 | Δ res  | quantized F1 | FPS  | ≥30 FPS? |
|-------------------------|---------|-------------------|--------|--------------|------|----------|
| FCN (ResNet-50)         | 57.9    | 52.8              | -5.2   | 54.0         | 19.7 | ✗        |
| FCN (ResNet-101)        | 63.4    | 52.7              | -10.6  | 53.8         | 16.8 | ✗        |
| DeepLabv3+ (ResNet-101) | 64.0    | 58.0              | -6.0   | 57.4         | 16.6 | ✗        |
| BiSeNetv1 (ResNet-50)   | 42.8    | 45.1              | +2.3   | 45.6         | 28.7 | ✗        |
| BiSeNetv2               | 54.7    | 42.9              | -11.8  | 46.0         | 44.6 | ✓        |
| STDC1                   | 61.8    | 48.5              | -13.3  | 47.9         | 45.8 | ✓        |
| STDC2                   | 64.3    | 50.8              | -13.5  | 49.9         | 38.3 | ✓        |
| SegFormer (MiT-B2)      | 70.0    | 61.8              | -8.2   | 58.0         | 15.0 | ✗        |

Key observation: STDC1 and STDC2 suffer the largest resolution penalty (-13 F1) because they
were not trained at the target resolution. Training natively at 768×384 should recover much
of this gap and is our primary advantage over these published baselines.

## Key References
- Challenge: macvi.org/workshop/cvpr/challenges/usv_seg_emb
- Starter repo: github.com/lojzezust/mmsegmentation-macvi
- Strategy report: docs/MaCVi2026_Preliminary_Report.pdf
