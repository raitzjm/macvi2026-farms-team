"""
BiSeNetV2 @ 768×384 — LaRS — Phase 1 baseline
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs original bisenetv2 lars config (mandatory — matches challenge server):
  - Train Pad: padding_mode='reflect'        (original had pad_val=0 zero-padding)
  - Test  Resize: keep_ratio=True            (original lars.py had keep_ratio=True but test Pad absent)
  - Test  Pad: size=crop_size, reflect       (original had NO Pad in test pipeline)

Batch size / LR (SGD — linear scaling rule: new_lr = base_lr × (new_bs / ref_bs)):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   bs=4,   lr=0.05  (bisenetv2_fcn_4x4_1024x1024_160k_lars.py upstream, per-GPU)
  Per-GPU:     samples_per_gpu=128  (OOM at bs=152 during first fwd pass — 5 total prediction heads
               (1 decode_head + 4 auxiliary_heads) need more memory than STDC2's 3 heads;
               reduced from 152 after confirmed OOM at full training step)
  Effective:   total_bs=256 (128 × 2 GPUs), lr=3.2 (= 0.05 × 256/4)
  NOTE: lr=3.2 is large; poly decay + warmup expected to stabilise training.
  Pinned for Phase 2 consistency: samples_per_gpu=128, num_gpus=2, lr=3.2
  Norm: SyncBN already set (required for DDP)

Run:
  # Use run_one_model.sh for the standard launch with WandB setup.
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  torchrun --standalone --nproc_per_node=2 \
      mmsegmentation-macvi/tools/train.py --launcher pytorch \
      challenges/embedded-semantic-segmentation/phase1/configs/bisenetv2_768x384_lars_phase1.py \
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/bisenetv2/
"""

# mmseg at: challenges/embedded-semantic-segmentation/mmsegmentation-macvi/
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
    # bs=128 (reduced from 152 after OOM at full training step — 5 aux heads exceed STDC2 memory budget)
    # Pinned for Phase 2 — do not re-calibrate.
    samples_per_gpu=128,
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

# ── Optimizer (SGD — architecture-specific, do not change type/momentum/wd) ───
# total_bs=256 (128×2 GPUs), lr=3.2: linear-scaled from ref (bs=4, lr=0.05): 0.05 × (256/4)
optimizer = dict(type='SGD', lr=3.2, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False,
    warmup='linear',
    warmup_iters=150)

# ── Phase 1 Schedule (1.5k iters — 2 GPUs; equivalent data to 1-GPU 3k plan) ──
# 1500 iters × 304 total_bs = 456k samples ≈ 3000 iters × 152 (original 1-GPU plan).
# Warmup scaled proportionally: 150 = 300 × (1500/3000).
runner = dict(type='IterBasedRunner', max_iters=1500)
checkpoint_config = dict(by_epoch=False, interval=750)
evaluation = dict(interval=250, metric='mIoU', pre_eval=True, save_best='mIoU')

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
                name='phase1_bisenetv2',
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
