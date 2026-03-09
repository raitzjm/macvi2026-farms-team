"""
PIDNet-S @ 768×384 — LaRS — Phase 2.5 pipeline-fix verification
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Phase 2.5 purpose: re-train with pipeline fixes (keep_ratio=True in training,
centred val padding, img_padding_size de-padding) to measure their effect vs
Phase 2 baseline. All model/optimizer/scheduler settings are IDENTICAL to
Phase 2 EXP-008 (pidnets_768x384_lars_phase2.py).

Batch size / LR (SGD — linear scaling rule: new_lr = base_lr × (new_bs / ref_bs)):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   pidnet-s_2xb6-120k_1024x1024-cityscapes.py:
               2 GPUs × 6/GPU = total_bs=12, lr=0.01 (SGD)
  Calibrated on 2026-03-04 with fwd+bwd pass (single A6000 48 GB):
    bs=80:  17.8 GB | bs=120: 26.7 GB | bs=160: 35.5 GB | bs=200: 44.4 GB | bs=220: 48.9 GB (OOM boundary)
    OOM boundary ≈ 220/GPU. 20% pre-emptive reduction → 176. Rounded to 160 (clean, 74% utilization).
  samples_per_gpu = 160, total_bs = 320 (160 × 2 GPUs)
  LR: lr=0.133 (0.01×160/12, single-GPU linear scale).
    Full 320/12=0.267 caused 79–89% oscillations in Phase 2.
    Best val at 0.267 was at warmup midpoint (lr~0.134) → confirms 0.133 as correct peak.
  NOTE: warmup extended to 1000 iters (was 500) for additional stability.
  Norm: SyncBN required for DDP (PIDNet uses BN)

  min_kept for OhemCrossEntropy scaled from upstream 1024×1024:
    131072 × (384×768)/(1024×1024) ≈ 36864. Using 32768 (close, matches fun exp).

Phase 2.5 schedule (identical to Phase 2 EXP-008):
  5k iters; eval every 250 iters.

  PIDNet requires GenerateEdge in the train pipeline (boundary supervision loss).

Run:
  source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh pidnets
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2_5.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2_5.pipelines',
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
            name='phase2_5_pidnets',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── PIDNet train pipeline (Fix 1: keep_ratio=True + GenerateEdge) ──────────────
# Fix 1 applied here because PIDNet overrides train_dataloader with its own pipeline.
# All other aspects identical to Phase 2 EXP-008 pidnet_train_pipeline.
crop_size = (384, 768)
img_scale = (768, 384)

pidnet_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),          # Fix 1: preserve AR (was False in Phase 2)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # Reflect padding matches challenge server mirror-padding; center=False
    # (default) for training — RandomCrop handles exact sizing.
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # PIDNet-specific: generate edge map for boundary loss
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

# ── Dataloader ─────────────────────────────────────────────────────────────────
# CALIBRATED 2026-03-04: bs=160/GPU (total_bs=320). OOM boundary ~220/GPU.
train_dataloader = dict(
    batch_size=160,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDNet-S — architecture-specific, identical to Phase 2) ────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
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
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
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

# ── Optimizer (SGD — identical to Phase 2 EXP-008, do not change) ─────────────
# lr=0.133 (0.01×160/12, single-GPU linear scale); revised from 0.267 which
# caused oscillations. Warmup extended to 1000 iters for stability.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.133,
        momentum=0.9,
        weight_decay=0.0005))

# ── Schedule (identical to Phase 2 EXP-008) ───────────────────────────────────
# 5k iters; warmup 1000 iters.
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

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,          # save checkpoint every 2500 iters
        save_best='mIoU',
        rule='greater'),
)
