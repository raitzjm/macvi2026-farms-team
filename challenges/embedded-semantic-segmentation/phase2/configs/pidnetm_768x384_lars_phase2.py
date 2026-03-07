"""
PIDNet-M @ 768×384 — LaRS — Phase 2 (Issue #19)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Motivation:
  PIDNet-S (EXP-008) runs at 67.6 FPS on device — well above the 30 FPS threshold.
  PIDNet-M (28.5M params vs 7.7M for S) is the official larger variant with the same:
  - BN + ReLU throughout → INT8-stable
  - D-branch + BoundaryLoss → boundary supervision
  - PAPPM → global context
  Only architectural change vs S: channels=64 (was 32), making PIDHead in_channels=256 (was 128).

Architecture changes vs EXP-008 (PIDNet-S):
  Backbone : PIDNet, channels=64 (vs 32 for S); ppm_channels=96 (same); stem/branch blocks unchanged
  Head     : PIDHead, in_channels=256 (vs 128 for S); channels=256 (vs 128 for S)
  Pretrained: ImageNet-1K PIDNet-M checkpoint from OpenMMLab

Batch size / LR (SGD — linear scaling rule same as EXP-008):
  Hardware:      2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:     pidnet-m_2xb6 (same as S): 2 GPUs × 6/GPU = total_bs=12, lr=0.01 (SGD)
  Calibration:   2026-03-07 (single-GPU fwd+bwd probe at 768×384, A6000 48 GB):
    [Probe bypasses data_preprocessor; real DDP adds ~15-25% overhead]
    BS=80:  30.70 GB | BS=100: 38.39 GB | BS=120: 46.01 GB | BS=140: OOM
    OOM boundary (probe) ≈ 130-140/GPU. Conservative DDP estimate: ~105/GPU.
    Issue #19 predicted 'roughly half of PIDNet-S (160/GPU)' → 80/GPU.
    Using BS=80 (DDP safety margin ≈ 76% of estimated DDP OOM ~105/GPU).
    Actual DDP training memory (iter 1, 2 GPUs): 36159 MiB ≈ 35.3 GB (73.5% of 48 GB) ✓
  samples_per_gpu = 80, total_bs = 160 (80 × 2 GPUs)
  LR: 0.01 × (80/12) = 0.0667 (same half-of-linear-scale formula as EXP-008).
    EXP-008: 0.01 × (160/12) = 0.133 at bs=160/GPU. At bs=80/GPU: 0.01 × (80/12) = 0.0667.
    If oscillations occur: halve to 0.033.
  Warmup: 1000 iters LinearLR (same as EXP-008, extended for stability).
  Note: SyncBN required for DDP (PIDNet uses BN).
  Note: No Phase 1 cross-check for PIDNet-M (new model, no Phase 1 baseline).

  min_kept for OhemCrossEntropy (same as EXP-008, resolution-scaled):
    131072 × (384×768)/(1024×1024) ≈ 36864. Using 32768 (close, matches EXP-008).

Phase 2 schedule:
  5k iters (same as EXP-008). Eval every 250 iters → 20 checkpoints.
  Scheduler: LinearLR (1000 iters warmup) + PolyLR (power=0.9, same as EXP-008).

WandB run name: phase2_pidnetm
Output dir:     outputs/phase2/pidnetm/

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh pidnetm
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2.pipelines',
    ],
    allow_failed_imports=False)

# ── WandB run name override ────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase2_pidnetm',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── PIDNet train pipeline (same as EXP-008: GenerateEdge for boundary loss) ───
crop_size = (384, 768)
img_scale = (768, 384)

pidnet_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

# ── Dataloader ─────────────────────────────────────────────────────────────────
# CALIBRATED 2026-03-07: bs=80/GPU (conservative, ~76% of estimated DDP OOM ~105/GPU).
# Probe OOM boundary: ~130-140/GPU (bypassed data_preprocessor).
train_dataloader = dict(
    batch_size=80,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDNet-M — only channels differ from PIDNet-S) ─────────────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-m_imagenet1k_20230306-39893c52.pth'  # noqa
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
        type='PIDNet',
        in_channels=3,
        channels=64,            # PIDNet-M (S=32)
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,        # PIDNet-M: 4 × channels = 256 (S=128)
        channels=256,           # PIDNet-M (S=128)
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=32768,     # scaled from 131072 @ 1024×1024 → 384×768
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=32768,
                loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Optimizer (SGD — same formula as EXP-008, halved from full linear scale) ──
# Reference: total_bs=12, lr=0.01.
# EXP-008 (PIDNet-S): bs=160/GPU → lr=0.133 (0.01×160/12; 0.267 caused oscillations).
# PIDNet-M: bs=80/GPU → lr=0.0667 (0.01×80/12, same formula).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.0667,          # 0.01 × (80/12); same half-of-linear-scale as EXP-008
        momentum=0.9,
        weight_decay=0.0005))

# ── Schedule (same as EXP-008: LinearLR 1000 iters warmup + PolyLR power=0.9) ─
iters = 5000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=1000,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=1000,
        end=iters,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=iters,
    val_interval=250)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        save_best='mIoU',
        rule='greater'),
)
