"""
BiSeNetV2 @ 768×384 — LaRS — Phase 2 full training
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Batch size / LR (SGD — linear scaling rule: new_lr = base_lr × (new_bs / ref_bs)):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   total_bs=16 (4x4 config: 4 GPUs × 4/GPU), lr=0.05
  Per-GPU:     samples_per_gpu=128  (pinned from Phase 1)
  Effective:   total_bs=256 (128 × 2 GPUs)
  LR:          0.80 (CORRECTED: 0.05 × (256/16), total_bs-to-total_bs)
               Phase 1 used lr=3.2 (WRONG: mixed per-GPU/total frames, 4× too high)
               Phase 1 instability (obstacle collapse at iter 750) traced to this error.
  Warmup:      300 iters (increased from Phase 1's 150 for additional stability)
  Norm: SyncBN required for DDP (BN stats synchronized across GPUs)

Phase 2 schedule:
  6k iters; Phase 1 BiSeNetV2 was far from converged at 1500 iters.
  6k iters × 256 total_bs = 1.5M samples.
  Eval every 250 iters → 24 checkpoints.
  Monitor obstacle IoU closely — Phase 1 saw collapse; corrected LR should prevent it.
  Cross-check skipped (Phase 1 was unstable — not a meaningful reference).

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh bisenetv2
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
            name='phase2_bisenetv2',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (BiSeNetV2 — architecture-specific, do not change) ──────────────────
norm_cfg = dict(type='SyncBN', requires_grad=True)

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

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# bs=128 calibrated on RTX A6000 48 GB at 768×384 (pinned from Phase 1).
train_dataloader = dict(batch_size=128)

# ── Optimizer (SGD — architecture-specific, do not change type/momentum/wd) ───
# total_bs=256 (128×2 GPUs), lr=0.80: CORRECTED linear scaling.
# Ref: bisenetv2_fcn_4x4_1024x1024 = 4 GPUs × 4/GPU = total_bs=16, lr=0.05.
# Correct: 0.05 × (256/16) = 0.80 (not 3.2 — Phase 1 mixed per-GPU/total).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.80,
        momentum=0.9,
        weight_decay=0.0005))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 6k iters; warmup 300 iters (increased from Phase 1's 150 for stability).
iters = 6000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=300,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=300,
        end=iters,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=iters,
    val_interval=250)    # eval every 250 iters; monitor obstacle IoU closely
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,          # save checkpoint every 2000 iters
        save_best='mIoU',
        rule='greater'),
)
