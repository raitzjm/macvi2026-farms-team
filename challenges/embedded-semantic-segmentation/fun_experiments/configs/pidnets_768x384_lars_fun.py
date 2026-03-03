"""
PIDNet-S @ 768×384 — LaRS — Fun Experiment
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121

Run with:
    python tools/train.py configs/pidnets_768x384_lars_fun.py --work-dir outputs/fun_pidnets_768x384
"""

custom_imports = dict(
    imports=['lars_dataset_v2'],
    allow_failed_imports=False)

DATA_ROOT = '/home/farms/lars_linked/'

# ── Resolution ────────────────────────────────────────────────────────────────
crop_size = (384, 768)   # (H, W) — what the network sees each step

# ── Data pre-processor (normalization + padding, replaces Normalize in pipeline)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# ── Model ─────────────────────────────────────────────────────────────────────
norm_cfg = dict(type='BN', requires_grad=True)   # SyncBN → BN (single GPU)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
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
        num_classes=3,           # obstacle, water, sky
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
                min_kept=32768,  # scaled from 131072 @ 1024×1024 → 384×768
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

# ── Pipelines ─────────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(768, 384),        # (W, H) base scale for random-resize jitter
        ratio_range=(0.5, 2.0),
        keep_ratio=False),       # stretch to target aspect ratio
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(768, 384), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# ── Dataloaders ───────────────────────────────────────────────────────────────
# samples_per_gpu → batch_size; will be updated after calibration
train_dataloader = dict(
    batch_size=15,          # max bs for 4GB GPU at 768×384 (bs=16 OOMs)
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'train',
        data_prefix=dict(img_path='images', seg_map_path='semantic_masks'),
        ann_file='image_list.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'val',
        data_prefix=dict(img_path='images', seg_map_path='semantic_masks'),
        ann_file='image_list.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ── Evaluators ────────────────────────────────────────────────────────────────
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ── Schedule ──────────────────────────────────────────────────────────────────
# Will be updated after timing calibration (target: ~2 hours)
iters = 14000   # ~2 hrs at bs=15, 0.490 s/iter

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=1000,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=iters,
        by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# ── Runtime ───────────────────────────────────────────────────────────────────
default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method='fork'))
randomness = dict(seed=42)
log_level = 'INFO'
load_from = None
resume = False
