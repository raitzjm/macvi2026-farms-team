"""
SegFormer MiT-B2 @ 768×384 — LaRS — Phase 2 full training
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   AdamW, lr=0.00006, bs=4  (segformer_mit-b0 upstream)
  Per-GPU:     samples_per_gpu=36  (pinned from Phase 1)
  Effective:   total_bs=72 (36 × 2 GPUs)
  LR:          0.00006 (fixed — AdamW does not use linear scaling)
  Norm: LayerNorm — SyncBN not applicable
  Pinned from Phase 1 — do not re-calibrate.

Phase 2 schedule:
  3k iters; SegFormer converged at iter 1000 in Phase 1.
  3k iters × 72 total_bs = 216k samples (3× Phase 1 convergence point of 72k).
  Eval every 250 iters → 12 checkpoints.
  Cross-check: record mIoU at iter 1500 vs Phase 1 ref (92.02%).

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh segformer_mit-b2
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
            name='phase2_segformer_mit-b2',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (SegFormer MiT-B2 — architecture-specific, do not change) ───────────
checkpoint_mit_b2 = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

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
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    # FIX: 'whole' mode — training at 768×384 target res; no sliding window
    test_cfg=dict(mode='whole'))

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# bs=36 calibrated on RTX A6000 48 GB at 768×384 (pinned from Phase 1).
train_dataloader = dict(batch_size=36)

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
# 3k iters; warmup 500 iters (scaled proportionally from Phase 1: 250 × 2).
iters = 3000
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
        interval=1500,          # save checkpoint every 1500 iters
        save_best='mIoU',
        rule='greater'),
)
