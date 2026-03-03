MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT = '/home/farms/lars_linked/'

_base_ = [f'{MMSEG_ROOT}/configs/stdc/stdc2_in1k-pre_512x1024_80k_lars.py']

# ── Resolution ────────────────────────────────────────────────────────────────
# Target deployment resolution: 768×384 (W×H).
# crop_size is (H, W) in mmseg; img_scale is (W, H) in mmseg Resize.
crop_size = (384, 768)   # what the network sees each step: 384 tall × 768 wide
img_scale = (768, 384)   # base scale for random-resize jitter (W, H)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ignore_idx = 255

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
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
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),  # stretch to exact 768×384
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=12,      # max bs for 4GB GPU at 768×384 (bs=13 OOMs)
    workers_per_gpu=2,
    pin_memory=False,
    train=dict(
        data_root=DATA_ROOT + 'train',
        pipeline=train_pipeline),
    val=dict(
        data_root=DATA_ROOT + 'val',
        pipeline=test_pipeline),
    test=dict(
        data_root=DATA_ROOT + 'test',
        pipeline=test_pipeline),
)

# ── Schedule ──────────────────────────────────────────────────────────────────
# Calibrated: bs=12 → ~0.58 s/iter → 12000 iters ≈ 2 hrs on RTX 3050 4GB.
# Evaluation every 2000 iters; saves best mIoU checkpoint automatically.
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True, save_best='mIoU')
