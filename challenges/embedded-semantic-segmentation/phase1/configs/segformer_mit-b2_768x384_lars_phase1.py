"""
SegFormer MiT-B2 @ 768×384 — LaRS — Phase 1 baseline
(Upper-bound accuracy reference; above 30 FPS threshold but useful for comparison)
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs original segformer lars config (mandatory — matches challenge server):
  - Train Pad: padding_mode='reflect'        (original had pad_val=0 zero-padding)
  - Test  Resize: keep_ratio=True            (original lars.py had keep_ratio=True — now explicit)
  - Test  Pad: size=crop_size, reflect       (original had NO Pad in test pipeline)
  - Test  mode: 'whole' instead of 'slide'   (training at target res — no sliding window needed)

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   AdamW, lr=0.00006, bs=4  (segformer_mit-b0_8x1_1024x1024_160k_lars.py per-GPU)
  Per-GPU:     samples_per_gpu=45  (OOM boundary at 768×384; OOM at bs=46)
  Effective:   total_bs=90 (45 × 2 GPUs)
  LR:          0.00006 (fixed — AdamW does not use linear scaling)
  Norm: LayerNorm — SyncBN not applicable
  Pinned for Phase 2 consistency: samples_per_gpu=45, num_gpus=2, lr=0.00006

Run:
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  python mmsegmentation-macvi/tools/train.py \\
      challenges/embedded-semantic-segmentation/phase1/configs/segformer_mit-b2_768x384_lars_phase1.py \\
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/segformer_mit-b2/
"""

MMSEG_ROOT = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/mmsegmentation-macvi'
DATA_ROOT  = '/home/farms/lars_linked/'

custom_imports = dict(
    imports=['challenges.embedded-semantic-segmentation.phase1.pipelines'],
    allow_failed_imports=False)

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
    # FIX vs upstream: 'whole' mode — training at 768×384 target res; no sliding window
    test_cfg=dict(mode='whole'))

# ── Pipelines ─────────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # FIX vs original: reflect padding matches challenge server mirror-padding
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
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
            # FIX vs original: explicit keep_ratio=True (challenge server behaviour)
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # FIX vs original: reflect-pad to exact crop_size (was absent)
            dict(type='ReflectPad', size=crop_size),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

# ── Dataset ───────────────────────────────────────────────────────────────────
data = dict(
    # bs=45 calibrated on RTX A6000 48 GB at 768×384 (OOM at bs=46)
    # Pinned for Phase 2 — do not re-calibrate.
    samples_per_gpu=45,
    workers_per_gpu=4,
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
# AdamW: LR fixed at 0.00006 regardless of batch size (no linear scaling).
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
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# ── Phase 1 Schedule (3k iters — short validation run; Phase 2 is the real baseline) ──
runner = dict(type='IterBasedRunner', max_iters=3000)
checkpoint_config = dict(by_epoch=False, interval=1500)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True, save_best='mIoU')

# ── Logging (WandB + text) ────────────────────────────────────────────────────
# WANDB_API_KEY must be set before training:  source phase1/scripts/setup_wandb.sh
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='little_farms',
                project='embedded_segmentation_macvi2026',
                name='phase1_segformer_mit-b2',
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
