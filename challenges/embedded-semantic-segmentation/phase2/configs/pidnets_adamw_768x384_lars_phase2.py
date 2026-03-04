"""
PIDNet-S (AdamW variant) @ 768×384 — LaRS — Phase 2 additional experiment
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

AdamW version of the SGD PIDNet-S run (EXP-008).
Do NOT override the SGD config (pidnets_768x384_lars_phase2.py).

Batch size / LR (AdamW — LR fixed, no linear scaling):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Optimizer:   AdamW, lr=6e-5 (same default as all Phase 2 AdamW models)
  Batch size:  Starting estimate: 64/GPU (total_bs=128).
               Calibrate fwd+bwd before first run; SGD version used 160/GPU.
               AdamW adds optimizer state (~20% more memory); conservative start.
  NOTE: After calibration, update samples_per_gpu and total_bs comment.
  NOTE: PIDNet requires GenerateEdge in train pipeline (boundary supervision loss).

Phase 2 schedule:
  5k iters. Eval every 250 iters → 20 checkpoints.
  WandB run: phase2_pidnets_adamw

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh pidnets_adamw

  !! Before first run: calibrate batch size fwd+bwd at bs=64, 96, 128/GPU !!
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
            name='phase2_pidnets_adamw',
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

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# Starting estimate: 64/GPU (total_bs=128). Calibrate before first run.
train_dataloader = dict(
    batch_size=64,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDNet-S — same architecture as SGD version) ───────────────────────
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

# ── Optimizer (AdamW — fixed LR, same as all Phase 2 AdamW models) ────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 5k iters; warmup 500 iters (AdamW standard linear warmup + PolyLR linear decay).
iters = 5000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
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
