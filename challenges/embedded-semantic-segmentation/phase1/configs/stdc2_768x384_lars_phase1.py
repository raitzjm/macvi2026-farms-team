"""
STDC2 @ 768×384 — LaRS — Phase 1 baseline
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  macvi conda env at /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs fun_experiments (mandatory, matches challenge server):
  - Train Pad: padding_mode='reflect'  (was pad_val=0 zero-padding)
  - Test Resize: keep_ratio=True       (was keep_ratio=False)
  - Test Pad: size=crop_size, padding_mode='reflect'  (was absent)

Batch size / LR calibration (SGD — linear scaling rule):
  Reference: bs=12, lr=0.01  (stdc1_in1k-pre_512x1024_80k_lars.py)
  Phase 1:   bs=12, lr=0.01  (calibrated on RTX 3050 4GB at 768×384 in fun_experiments)
  LR kept same because batch size matches reference.

Run:
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  python mmsegmentation-macvi/tools/train.py \\
      challenges/embedded-semantic-segmentation/phase1/configs/stdc2_768x384_lars_phase1.py \\
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/stdc2/
"""

MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT  = '/home/farms/lars_linked/'

# ── Resolution ────────────────────────────────────────────────────────────────
crop_size = (384, 768)   # (H, W) — target deployment resolution
img_scale = (768, 384)   # (W, H) — for mmseg Resize

# ── Normalization ─────────────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ── Model (STDC2: STDCContextPathNet with STDCNet2 backbone) ──────────────────
checkpoint_stdc2 = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
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
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            ]),
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
    # bs=12 calibrated at 768×384 on RTX 3050 4GB (fun_experiments stdc2)
    samples_per_gpu=12,
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
# bs=12, lr=0.01: matches reference; no LR scaling needed.
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
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
# schedule_80k has interval=8000 which would miss the 10k window)
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
                name='stdc2_phase1',
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
