"""
STDC2 @ 768×384 — LaRS — Phase 3 augmentation training (EXP-020)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Phase 3 purpose: augmentation-focused training with CopyPasteObstacles +
EnhancedPhotoMetricDistortion. Mosaic is NOT included — DetailAggregateLoss
applies a Laplacian convolution to GT at runtime, making mosaic seams
become spurious edge targets (same risk as PIDNet-S GenerateEdge).

NOTE: STDC2 does NOT use GenerateEdgeMasked (no GenerateEdge in pipeline).
The base lars_dataset.py train_pipeline is used directly.

ReflectPad is NOT used in the STDC2 train pipeline:
  STDC2 does not use BoundaryLoss and does not have spurious-edge risk from
  padded regions in the same way PIDNet-S does. Plain bottom-right padding
  still applies via the base pipeline's ReflectPad (center=False).

Batch size / LR: identical to Phase 2.5.
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  samples_per_gpu=160, total_bs=320, lr=0.267 (SGD linear-scaled from ref).
  Warmup: 300 iters (identical to Phase 2.5).

Schedule:
  5000 iters (extended from Phase 2.5's 3000).
  Eval every 500 iters; save best mIoU.

WandB run name: phase3_stdc2

Baseline to beat:
  EXP-017 (Phase 2.5): local val Q=64.49%, server Q=45.8

IMPORTANT: Run ONLY after EXP-019 (PIDNet-S) fully completes.
Hardware constraint: 2× RTX A6000 — only one DDP run at a time.

Run:
  source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh stdc2
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase3.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase3.pipelines',
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
            name='phase3_stdc2',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (STDC2 — identical to Phase 2.5) ────────────────────────────────────
checkpoint_stdc2 = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
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
        type='STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet2',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_stdc2)),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=3,
        ignore_index=255,
        in_index=3,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            ignore_index=255,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce',
                     use_sigmoid=True, loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            ]),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Dataloader ─────────────────────────────────────────────────────────────────
# bs=160 calibrated on RTX A6000 48 GB at 768×384 (pinned from Phase 1).
train_dataloader = dict(batch_size=160)

# ── Optimizer (SGD — identical to Phase 2.5 EXP-017) ─────────────────────────
# total_bs=320, lr=0.267: linear-scaled from ref (bs=12, lr=0.01).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.267, momentum=0.9, weight_decay=0.0005))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 5k iters (extended from Phase 2.5's 3k); warmup 300 iters.
iters = 5000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=300, by_epoch=False),
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=300, end=iters, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,       # save checkpoint every 2500 iters
        save_best='mIoU',
        rule='greater'),
)
