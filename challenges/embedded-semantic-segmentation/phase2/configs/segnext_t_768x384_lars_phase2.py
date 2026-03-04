"""
SegNeXt-T @ 768×384 — LaRS — Phase 2 full training
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   AdamW, lr=0.00006 (upstream SegNeXt)
  Per-GPU:     samples_per_gpu=56  (pinned from Phase 1; OOM at 75 — HAM decoder)
  Effective:   total_bs=112 (56 × 2 GPUs)
  LR:          0.00006 (fixed — AdamW does not use linear scaling)
  Norm: LayerNorm — SyncBN not applicable
  Pinned from Phase 1 — do not re-calibrate.

Phase 2 schedule:
  5k iters; SegNeXt-T was still climbing at iter 1500 in Phase 1.
  5k iters × 112 total_bs = 560k samples (well past convergence estimate).
  Eval every 250 iters → 20 checkpoints.
  Cross-check: record mIoU at iter 1500 vs Phase 1 ref (89.81%).

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh segnext_t
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2.pipelines',
    ],
    allow_failed_imports=False)

# ── WandB run name override ────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase2_segnext_t',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (SegNeXt-T — architecture-specific, do not change) ──────────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(384, 768)),
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

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# bs=56 calibrated on RTX A6000 48 GB at 768×384 (pinned from Phase 1).
train_dataloader = dict(batch_size=56)

# ── Optimizer (AdamW — architecture-specific, do not change) ──────────────────
# AdamW: LR fixed at 0.00006 regardless of batch size (no linear scaling).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 5k iters; warmup 500 iters (scaled proportionally from Phase 1: 250 × 2).
iters = 5000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=iters,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=iters,
    val_interval=250)    # eval every 250 iters; 1500-iter point captured for cross-check
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,          # save checkpoint every 2500 iters
        save_best='mIoU',
        rule='greater'),
)
