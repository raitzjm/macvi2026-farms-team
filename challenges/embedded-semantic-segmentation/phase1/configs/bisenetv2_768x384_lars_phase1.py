"""
BiSeNetV2 @ 768×384 — LaRS — Phase 1 baseline
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  macvi conda env at /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs original bisenetv2 lars config (mandatory, matches challenge server):
  - Train Pad: padding_mode='reflect'  (was pad_val=0 zero-padding)
  - Test Resize: keep_ratio=True       (was keep_ratio=True in lars.py — verify)
  - Test Pad: size=crop_size, padding_mode='reflect'  (was absent)

Batch size / LR calibration (SGD — linear scaling rule):
  Reference: bs=4, lr=0.05  (bisenetv2_fcn_4x4_1024x1024_160k_lars.py)
  Phase 1:   bs=4, lr=0.05  (using reference as starting point; calibrate at 768×384)
  NOTE: At 768×384 the GPU may fit more than bs=4. Increase and rescale lr if OOM-free:
        new_lr = 0.05 × (new_bs / 4). Update comment and rerun if batch size changes.

Run:
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  python mmsegmentation-macvi/tools/train.py \\
      challenges/embedded-semantic-segmentation/phase1/configs/bisenetv2_768x384_lars_phase1.py \\
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/bisenetv2/
"""

MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT  = '/home/farms/lars_linked/'

# ── Resolution ────────────────────────────────────────────────────────────────
crop_size = (384, 768)   # (H, W)
img_scale = (768, 384)   # (W, H)

# ── Normalization ─────────────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ── Model (BiSeNetV2 — architecture-specific, do not change) ──────────────────
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
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

# ── Pipelines (corrected preprocessing) ───────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # FIX: reflect padding matches challenge server mirror-padding
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255, padding_mode='reflect'),
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
            # FIX: keep_ratio=True preserves aspect ratio (challenge server behaviour)
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # FIX: reflect-pad to exact crop_size (centering + mirror padding)
            dict(type='Pad', size=crop_size, padding_mode='reflect'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

# ── Dataset ───────────────────────────────────────────────────────────────────
data = dict(
    # bs=4 is reference from 1024×1024 config; calibrate upward at 768×384 if headroom exists.
    # If increasing bs, scale lr: new_lr = 0.05 × (new_bs / 4). Update comment above.
    samples_per_gpu=4,
    workers_per_gpu=2,
    pin_memory=False,
    train=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'train',
        pipeline=train_pipeline),
    val=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'val',
        pipeline=test_pipeline),
    test=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'test',
        pipeline=test_pipeline))

# ── Optimizer (SGD — architecture-specific, do not change) ────────────────────
# bs=4, lr=0.05: matches reference. Scale lr if bs changes: new_lr = 0.05 × (new_bs / 4).
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000)

# ── Phase 1 Schedule (10k iters) ──────────────────────────────────────────────
runner = dict(type='IterBasedRunner', max_iters=10000)
# interval=5000 ensures a periodic checkpoint before the run ends (inherited
# schedule_160k has interval=16000 which would miss the 10k window)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True, save_best='mIoU')

# ── Logging (WandB + text) ────────────────────────────────────────────────────
# WANDB_API_KEY must be set in the environment before training.
# Load it with:  source phase1/scripts/setup_wandb.sh
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='MacVi2026_Embedded_Semantic_Segmentation_Phase1',
                name='bisenetv2_phase1',
            ),
            interval=50,
            by_epoch=False,
            with_step=True,
            log_artifact=False,
        ),
    ])

# ── Runtime ───────────────────────────────────────────────────────────────────
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
seed = 42
