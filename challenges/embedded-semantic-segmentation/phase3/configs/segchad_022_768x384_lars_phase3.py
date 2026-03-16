"""
SegChad @ 768x384 — LaRS — Phase 3 training (EXP-022)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

EXP-022 introduces SegChad — a ConvNeXt-V2 Femto backbone with a 3-stream
D/P/I decoder, PAG fusion, and boundary-guided aggregation.

Architecture:
  - Backbone: ConvNeXt-V2 Femto (5.2M params, channels [48, 96, 192, 384])
    with ImageNet-1k FCMAE pretrained weights via timm.
  - D Branch: boundary refinement stream (1/4 res, 64 ch)
  - P Branch: detail/parsing stream (1/8 res, 128 ch)
  - I Branch: semantic context stream (1/8 res, 256 ch) with PAG fusion
  - Boundary-Guided Fusion at 1/4 res → lightweight decoder → 1/1 res
  - Total params: ~8-10M (backbone 5.2M + decoder ~3-5M)

Optimizer: AdamW (standard for ConvNeXt architectures).
  - Differential LR: backbone 1e-5 (0.1x), decoder/heads 1e-4
  - weight_decay=0.05, cosine annealing with linear warmup
  - AdamW LR does NOT scale with batch size (keep fixed regardless of BS)

Training recipe (EXP-020 augmentation baseline, NOT EXP-021 aggressive):
  - Moderate LSJ: ratio_range=(0.5, 2.0) (same as EXP-020)
  - CopyPasteObstacles for obstacle augmentation
  - EnhancedPhotoMetricDistortion
  - No GenerateEdge (boundary GT generated internally by SegChadHead)
  - Boundary loss weight 10.0 (moderate, appropriate for this architecture)

Hardware: 2x RTX A6000 48 GB (DDP)
  # SERVER: samples_per_gpu=16, total_bs=32
  # LAPTOP SMOKE TEST: temporarily set batch_size=1 via --cfg-options
  # Estimated ~15-20 it/s on 2x A6000 → 12000 iters ≈ 10-13 minutes
  # Well within the 5-hour budget

Schedule:
  24000 iters, eval every 500 iters, save best mIoU.
  Linear warmup (1000 iters) + cosine annealing over 23k iters.
  Extended from 12k: cosine LR reaches ~0 at end so extending post-hoc is
  ineffective; decoder has ~18 randomly init'd layers needing convergence time.

WandB run name: phase3_segchad_022

Baselines to beat:
  EXP-020 (PIDNet-S Phase 3): local val Q=70.51%, server Q=52.1%
  EXP-021 (PIDChad-S Phase 3): TBD

INT8 compatibility:
  ConvNeXt-V2 uses channel-dim LayerNorm (typically INT8-safe on SNPE) and
  GRN (element-wise ops). INT8 smoke test mandatory before full training.
  All decoder ops are INT8-safe (Conv, BN, ReLU, sigmoid, bilinear).

Run (BS/LR probe first):
  # OOM probe (5 iters):
  torchrun --standalone --nproc_per_node=2 <mmseg_train> \\
      phase3/configs/segchad_022_768x384_lars_phase3.py \\
      --work-dir /tmp/segchad_probe --cfg-options train_cfg.max_iters=5

  # INT8 smoke test (500 iters):
  torchrun --standalone --nproc_per_node=2 <mmseg_train> \\
      phase3/configs/segchad_022_768x384_lars_phase3.py \\
      --work-dir /tmp/segchad_smoke --cfg-options train_cfg.max_iters=500

  # Full training:
  source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh segchad_022
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase3.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase3.pipelines',
        'challenges.embedded-semantic-segmentation.phase3.models',
    ],
    allow_failed_imports=False)

# ── WandB run name ─────────────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase3_segchad_022',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Train pipeline (EXP-020 moderate LSJ, no GenerateEdge) ──────────────────
# SegChad generates boundary GT internally from segmentation labels —
# no GenerateEdge pipeline transform needed (unlike PIDNet).
crop_size = (384, 768)
img_scale = (768, 384)
DB_PATH = ('challenges/common/external-dataset-analysis/'
           'shared_data/copypaste_candidates.db')

segchad_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Moderate LSJ (0.5, 2.0) — same as EXP-020 (NOT aggressive 0.1 from EXP-021)
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # CopyPaste before photometric distortion so pasted crops get distorted too
    dict(type='CopyPasteObstacles',
         db_path=DB_PATH,
         n_paste=(1, 3),
         prob_apply=0.7,
         scale_range=(0.5, 1.5),
         prefer_waterline_prob=0.3,
         boundary_blur_radius=2,
         match_brightness=True),
    dict(type='EnhancedPhotoMetricDistortion',
         brightness_delta=77,
         contrast_range=(0.6, 1.5),
         saturation_range=(0.6, 1.5),
         hue_delta=27,
         gamma_range=(0.6, 1.6),
         gamma_prob=0.3),
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # No GenerateEdge — SegChad generates boundary GT internally
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

# SERVER: samples_per_gpu=16, total_bs=32 (2 GPUs)
# ConvNeXt-V2 Femto at 768x384 with 3-stream decoder: ~800MB/image estimated.
# 48GB GPU → conservative BS=16 per GPU to accommodate decoder overhead.
# LAPTOP SMOKE TEST: override to batch_size=1 via --cfg-options train_dataloader.batch_size=1
train_dataloader = dict(
    batch_size=16,
    dataset=dict(pipeline=segchad_train_pipeline))

# ── Model (SegChad: ConvNeXt-V2 Femto + 3-stream decoder) ────────────────────
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(384, 768)),
    backbone=dict(
        type='SegChadBackbone',
        # timm model name: ConvNeXt-V2 Femto, FCMAE pretrained + IN1k finetuned
        # Alternative: 'convnextv2_femto.d1h_in1k' (DINOv2 head-distilled, if available)
        model_name='convnextv2_femto.fcmae_ft_in1k',
        pretrained=True,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type='SegChadHead',
        # ConvNeXt-V2 Femto stage channels
        in_channels=[48, 96, 192, 384],
        channels=256,           # fusion output channels
        num_classes=3,
        d_channels=64,          # D-branch feature channels
        p_channels=128,         # P-branch output channels
        i_channels=256,         # I-branch output channels
        decoder_mid_channels=128,
        decoder_out_channels=64,
        boundary_loss_weight=10.0,  # moderate (NOT 25.0 from EXP-021); class-balanced
                                    # BCE internally reduces magnitude so 10.0 is needed
                                    # (comparable to PIDNet's proven 20.0)
        boundary_width=2,           # 5-pixel kernel (was 4→9px which could swallow
                                    # small distant obstacles — critical for F1)
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            # Main segmentation loss
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Optimizer (AdamW — standard for ConvNeXt) ────────────────────────────────
# Differential LR: backbone gets 0.1x the base LR (1e-5 vs 1e-4).
# This replaces explicit backbone freezing with a gentler approach.
# weight_decay=0.05 is the standard ConvNeXt recipe.
# AdamW LR does NOT scale with batch size — keep fixed.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }))

# ── Schedule (24k iters, cosine annealing) ───────────────────────────────────
# 24k chosen over 12k because: cosine LR reaches ~zero at end of schedule, so
# extending past max_iters is ineffective (LR stays at eta_min=1e-6). The decoder
# has ~18 randomly initialized conv layers — more than PIDNet-S's ~6 — and may
# need more iters to converge. 24k is still ~20-25 min on 2x A6000 (well within
# 5h budget). best_mIoU checkpoint is saved throughout so early convergence is fine.
# Linear warmup (1000 iters) + cosine annealing over remaining 23k iters.
#
# SERVER AGENT — BS calibration:
#   1. Run OOM probe: torchrun --nproc_per_node=2 ... --cfg-options train_cfg.max_iters=5
#   2. Try samples_per_gpu=16 (current), then 24, then 32 — use max that fits OOM-free
#   3. LR: keep lr=1e-4 fixed for total_bs up to ~64 (AdamW is BS-insensitive in this
#      range with 2605 training images + aggressive augmentation). If total_bs >64,
#      consider a mild bump to lr=1.5e-4 — but do not apply linear scaling (that is
#      SGD-only per CLAUDE.md). Update the batch_size comment below with final value.
iters = 24000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1000, by_epoch=False),
    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=1000, end=iters,
         by_epoch=False, T_max=iters - 1000),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,        # save checkpoint every 4000 iters (6 saves over 24k)
        save_best='mIoU',
        rule='greater'),
)
