"""
SegNeXt-T @ 768×384 — LaRS — Phase 1 baseline
mmsegmentation 0.30.0 + mmcv-full 1.7.2 + PyTorch 2.1.2+cu121
Conda env: macvi  |  /home/farms/miniforge3/envs/macvi/

Preprocessing fixes vs fun_experiments segnext_t config (mandatory — matches challenge server):
  - Train Pad: padding_mode='reflect'        (fun_experiments had pad_val=0 zero-padding)
  - Test  Resize: keep_ratio=True            (fun_experiments had keep_ratio=False)
  - Test  Pad: size=crop_size, reflect       (fun_experiments had NO Pad in test pipeline)

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   AdamW, lr=0.00006  (upstream SegNeXt config)
  Per-GPU:     samples_per_gpu=56  (OOM at bs=75 during full training step — HAM decoder is
               memory-intensive; reduced ~25% from fwd+bwd-only boundary of bs=75)
  Effective:   total_bs=112 (56 × 2 GPUs)
  LR:          0.00006 (fixed — AdamW does not use linear scaling)
  Norm: LayerNorm — SyncBN not applicable
  Pinned for Phase 2 consistency: samples_per_gpu=56, num_gpus=2, lr=0.00006

Run:
  # Use run_one_model.sh for the standard launch with WandB setup.
  source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
  torchrun --standalone --nproc_per_node=2 \
      mmsegmentation-macvi/tools/train.py --launcher pytorch \
      challenges/embedded-semantic-segmentation/phase1/configs/segnext_t_768x384_lars_phase1.py \
      --work-dir challenges/embedded-semantic-segmentation/outputs/phase1/segnext_t/
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

# ── Pipelines ─────────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # FIX vs fun_experiments: reflect padding matches challenge server mirror-padding
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
            # FIX vs fun_experiments: keep_ratio=True preserves aspect ratio
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # FIX vs fun_experiments: reflect-pad to exact crop_size (was absent)
            dict(type='ReflectPad', size=crop_size),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

# ── Dataset ───────────────────────────────────────────────────────────────────
data = dict(
    # bs=56 (reduced from 75 after OOM at full training step — HAM decoder memory-intensive)
    # Pinned for Phase 2 — do not re-calibrate.
    samples_per_gpu=56,
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
    warmup_iters=250,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# ── Phase 1 Schedule (1.5k iters — 2 GPUs; equivalent data to 1-GPU 3k plan) ──
# 1500 iters × 112 total_bs = 168k samples = 3000 iters × 56 (original 1-GPU plan).
# Warmup scaled proportionally: 250 = 500 × (1500/3000).
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
                name='phase1_segnext_t',
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
