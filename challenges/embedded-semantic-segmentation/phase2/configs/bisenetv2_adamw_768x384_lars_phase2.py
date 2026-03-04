"""
BiSeNetV2 (AdamW variant) @ 768×384 — LaRS — Phase 2 additional experiment
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

AdamW version of the SGD BiSeNetV2 run (EXP-009).
Do NOT override the SGD config (bisenetv2_768x384_lars_phase2.py).

Batch size / LR (AdamW — LR fixed, no linear scaling):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Optimizer:   AdamW, lr=6e-5 (same default as all Phase 2 AdamW models)
  Batch size:  Starting estimate: 64/GPU (total_bs=128).
               Calibrate fwd+bwd before first run; SGD version used 128/GPU.
               AdamW adds optimizer state (~20% more memory); conservative start.
  NOTE: After calibration, update samples_per_gpu and total_bs comment.

Phase 2 schedule:
  5k iters. Eval every 250 iters → 20 checkpoints.
  WandB run: phase2_bisenetv2_adamw

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh bisenetv2_adamw

  !! Before first run: calibrate batch size fwd+bwd at bs=64, 96, 112/GPU !!
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
            name='phase2_bisenetv2_adamw',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (BiSeNetV2 — same architecture as SGD version) ──────────────────────
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
        type='BiSeNetV2',
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=3,
            ignore_index=255,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=3,
            ignore_index=255,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=3,
            ignore_index=255,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=3,
            ignore_index=255,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# Starting estimate: 64/GPU (total_bs=128). Calibrate before first run.
train_dataloader = dict(batch_size=64)

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
