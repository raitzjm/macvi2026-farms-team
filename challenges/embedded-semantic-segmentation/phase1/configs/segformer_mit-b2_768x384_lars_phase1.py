"""
SegFormer MiT-B2 @ 768×384 — LaRS — Phase 1 baseline
(Upper-bound reference model — above 30 FPS threshold but useful for accuracy comparison)
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  macvi conda env at /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs original segformer lars config (mandatory, matches challenge server):
  - Train Pad: padding_mode='reflect'  (was pad_val=0 zero-padding)
  - Test Resize: keep_ratio=True       (was keep_ratio=True in lars.py — now explicit)
  - Test Pad: size=crop_size, padding_mode='reflect'  (was absent)
  - Test mode: 'whole' instead of 'slide' (we train at target res; slide is for multi-scale)

Batch size / LR calibration (AdamW — keep LR fixed regardless of batch size):
  Reference: AdamW, lr=0.00006, bs=4  (segformer_mit-b0_8x1_1024x1024_160k_lars.py)
  Phase 1:   bs=4, lr=0.00006   (conservative start; calibrate upward at 768×384 if headroom)
  NOTE: At 768×384 the GPU may fit more than bs=4. Increase bs if OOM-free (lr stays fixed).

Run:
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  python mmsegmentation-macvi/tools/train.py \\
      challenges/embedded-semantic-segmentation/phase1/configs/segformer_mit-b2_768x384_lars_phase1.py \\
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/segformer_mit-b2/
"""

MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT  = '/home/farms/lars_linked/'

# ── Resolution ────────────────────────────────────────────────────────────────
crop_size = (384, 768)   # (H, W)
img_scale = (768, 384)   # (W, H)

# ── Normalization ─────────────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ── Model (SegFormer MiT-B2 — architecture-specific, do not change) ───────────
checkpoint_mit_b2 = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_mit_b2)),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    # 'whole' mode: we train at target res, no need for sliding window
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
    # AdamW: lr stays fixed at 0.00006 regardless of bs change.
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
                name='segformer_mit-b2_phase1',
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
