"""
PIDNet-S @ 768×384 — LaRS — Phase 2 full training
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Batch size / LR (SGD — linear scaling rule: new_lr = base_lr × (new_bs / ref_bs)):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   pidnet-s_2xb6-120k_1024x1024-cityscapes.py:
               2 GPUs × 6/GPU = total_bs=12, lr=0.01 (SGD)
  Calibration needed before first run at 768×384 on A6000 48 GB.
    Starting estimate: samples_per_gpu=80 (pre-emptive 20% below expected OOM boundary).
    OOM boundary unknown — calibrate with fwd+bwd pass at 768×384.
    After calibration, pin samples_per_gpu and update this comment.
  LR (estimated): 0.01 × (total_bs / 12)  — e.g. if total_bs=160: lr=0.133
  NOTE: warmup added for stability with large LR (not in upstream default).
  Norm: SyncBN required for DDP (PIDNet uses BN)
  NOTE: No Phase 1 cross-check for PIDNet (new model in Phase 2).

  min_kept for OhemCrossEntropy scaled from upstream 1024×1024:
    131072 × (384×768)/(1024×1024) ≈ 36864. Using 32768 (close, matches fun exp).

Phase 2 schedule:
  5k iters; equivalent to 5k × total_bs samples. Eval every 250 iters.
  The upstream 120k iters at bs=12 = 1.44M samples. At bs=160, 5k iters = 800k samples.
  Extend to 7k if not converged (monitor at 5k).

  PIDNet requires GenerateEdge in the train pipeline (boundary supervision loss).

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh pidnets

  !! Before first run: calibrate batch size by testing fwd+bwd at bs=80, 100, 120 !!
  !! Then update samples_per_gpu and lr in this config accordingly.               !!
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
            name='phase2_pidnets',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── PIDNet train pipeline (adds GenerateEdge for boundary supervision) ─────────
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
    # FIX: reflect padding matches challenge server mirror-padding
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # PIDNet-specific: generate edge map for boundary loss
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

# ── Dataloader override ────────────────────────────────────────────────────────
# CALIBRATE before training: test fwd+bwd at bs=80, 100, 120 on A6000 48GB.
# Reference: upstream total_bs=12 on Cityscapes 1024×1024.
# Starting estimate: bs=80/GPU → total_bs=160. Update after calibration.
train_dataloader = dict(
    batch_size=80,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDNet-S — architecture-specific, do not change) ───────────────────
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

# ── Optimizer (SGD — architecture-specific, do not change) ────────────────────
# Reference: upstream total_bs=12, lr=0.01 (linear scaling rule).
# Estimated: total_bs=160 (80×2 GPUs) → lr = 0.01 × (160/12) = 0.133.
# UPDATE after calibration: adjust lr proportionally if batch size changes.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.133,           # 0.01 × (160/12); update after calibration
        momentum=0.9,
        weight_decay=0.0005))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 5k iters; warmup 500 iters.
# Upstream used no warmup but with total_bs=12; at ~13× larger bs, warmup helps.
iters = 5000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=500,
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
