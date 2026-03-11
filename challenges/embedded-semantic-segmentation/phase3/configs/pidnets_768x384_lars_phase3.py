"""
PIDNet-S @ 768×384 — LaRS — Phase 3 augmentation training (EXP-019)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Phase 3 purpose: augmentation-focused training with CopyPasteObstacles +
EnhancedPhotoMetricDistortion + GenerateEdgeMasked (Option 1 fix, issue #28).
GenerateEdgeMasked makes keep_ratio=True safe for PIDNet-S by zeroing edge
targets in reflect-padded regions.

Batch size / LR: identical to Phase 2 EXP-008 and Phase 2.5.
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  samples_per_gpu=160, total_bs=320, lr=0.133 (SGD linear-scaled from ref).
  Warmup extended to 1500 iters to absorb augmented variance (was 1000).

Schedule:
  8000 iters (extended from 5000 — augmented data needs more steps).
  Eval every 500 iters; save best mIoU.

WandB run name: phase3_pidnets

Baseline to beat:
  EXP-008 (Phase 2): local val Q=71.03%, server Q=49.5

Run:
  source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh pidnets
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
            name='phase3_pidnets',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── PIDNet train pipeline (Phase 2.5 fixes + CopyPaste + GenerateEdgeMasked) ──
# PIDNet overrides train_dataloader with its own pipeline (needs GenerateEdge
# and must come after ReflectPad so img_padding_size is already set).
crop_size = (384, 768)
img_scale = (768, 384)
DB_PATH = ('challenges/common/external-dataset-analysis/'
           'shared_data/copypaste_candidates.db')

pidnet_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
    # ReflectPad writes img_padding_size BEFORE GenerateEdgeMasked reads it
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # GenerateEdgeMasked: zeros edge targets in padded regions (Option 1 fix)
    dict(type='GenerateEdgeMasked', edge_width=4),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

train_dataloader = dict(
    batch_size=160,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDNet-S — identical to Phase 2 EXP-008) ──────────────────────────
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
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Optimizer (SGD — identical to Phase 2 EXP-008) ───────────────────────────
# lr=0.133 (0.01×160/12); full 0.267 caused oscillations in Phase 2.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.133, momentum=0.9, weight_decay=0.0005))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 8k iters; warmup extended to 1500 to absorb augmented variance.
iters = 8000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=iters, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,       # save checkpoint every 4000 iters
        save_best='mIoU',
        rule='greater'),
)
