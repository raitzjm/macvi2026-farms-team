# Option C (Pixi) dry-run config — STDC1 on LaRS
# Based on stdc1_in1k-pre_512x1024_80k_lars.py from mmsegmentation-macvi,
# reduced to 2605 iterations (≈1 epoch) for dependency validation only.

_base_ = [
    '../mmsegmentation-macvi/configs/_base_/models/stdc.py',
    '../mmsegmentation-macvi/configs/_base_/default_runtime.py',
]

# ---------- Dataset ----------
dataset_type = 'LaRSDataset'
data_root = '/home/farms/lars_linked/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 768)  # target inference resolution (H, W)
ignore_idx = 255

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(768, 384), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=ignore_idx),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    # samples_per_gpu=2: bs=1 causes BN error in STDC AttentionRefinementModule
    samples_per_gpu=2,
    workers_per_gpu=2,
    pin_memory=False,
    train=dict(
        type=dataset_type,
        data_root=data_root + 'train',
        split='image_list.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root + 'val',
        split='image_list.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'val',
        split='image_list.txt',
        pipeline=test_pipeline),
)

# ---------- Model ----------
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(ignore_index=ignore_idx, num_classes=3),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            ignore_index=ignore_idx,
            num_classes=3,
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
            ignore_index=ignore_idx,
            num_classes=3,
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
            ignore_index=ignore_idx,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]),
    ],
)

# ---------- Schedule ----------
# Dry-run: ~1 epoch (2605 train images / 2 samples_per_gpu = 1303 iter/epoch)
# Run 2605 iters to match Options A/B for easy comparison.
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000,
)
runner = dict(type='IterBasedRunner', max_iters=2605)
checkpoint_config = dict(by_epoch=False, interval=2605)  # save once at end
evaluation = dict(interval=2605, metric='mIoU', pre_eval=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
