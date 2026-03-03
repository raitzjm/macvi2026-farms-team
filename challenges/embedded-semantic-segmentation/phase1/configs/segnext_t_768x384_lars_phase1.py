"""
SegNeXt-T @ 768×384 — LaRS — Phase 1 baseline
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  macvi conda env at /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs fun_experiments segnext_t config (mandatory, matches challenge server):
  - Train Pad: padding_mode='reflect'  (was pad_val=0 zero-padding)
  - Test Resize: keep_ratio=True       (was keep_ratio=False)
  - Test Pad: size=crop_size, padding_mode='reflect'  (was absent)

Batch size / LR calibration (AdamW — keep LR fixed regardless of batch size):
  Reference: AdamW, lr=0.00006  (upstream SegNeXt config)
  Phase 1:   bs=5, lr=0.00006   (calibrated on RTX 3050 4GB at 768×384 in fun_experiments)

Run:
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  python mmsegmentation-macvi/tools/train.py \\
      challenges/embedded-semantic-segmentation/phase1/configs/segnext_t_768x384_lars_phase1.py \\
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/segnext_t/
"""

MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT  = '/home/farms/lars_linked/'

# ── Resolution ────────────────────────────────────────────────────────────────
crop_size = (384, 768)   # (H, W)
img_scale = (768, 384)   # (W, H)

# ── Normalization ─────────────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ── Model (SegNeXt-T — architecture-specific, do not change) ──────────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        ignore_index=255,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
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
    # bs=5 calibrated at 768×384 on RTX 3050 4GB (fun_experiments segnext_t)
    samples_per_gpu=5,
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

# ── Optimizer (AdamW — architecture-specific, do not change) ──────────────────
# AdamW: LR kept fixed at 0.00006 regardless of batch size (no linear scaling).
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }))
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# ── Phase 1 Schedule (10k iters) ──────────────────────────────────────────────
runner = dict(type='IterBasedRunner', max_iters=10000)
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
                name='segnext_t_phase1',
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
