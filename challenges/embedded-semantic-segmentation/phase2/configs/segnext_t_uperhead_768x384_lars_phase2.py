"""
MSCAN-T + UPerHead + BoundaryLoss @ 768×384 — LaRS — Phase 2 (Issue #18)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Motivation:
  SegNeXt-T (EXP-006) fails catastrophically under INT8 (mIoU 91.79% → 24.91%)
  due to the LightHamHead's iterative NMF (7 steps) + GroupNorm.  The MSCAN-T
  backbone is INT8-stable (BN + depthwise convs).  This experiment replaces the
  decoder with UPerHead (BN throughout, PPM global context) and adds boundary
  supervision via an auxiliary MSCANBoundaryHead + BoundaryLoss + GenerateEdge.

Architecture changes vs EXP-006:
  Backbone  : MSCAN-T (unchanged)
  Decoder   : UPerHead (BN + PPM) instead of LightHamHead (GN + NMF)
  Auxiliary : MSCANBoundaryHead (stride-4 feature → 1-ch boundary logit + BoundaryLoss)
  Norm      : BN throughout (backbone already BN; UPerHead/BoundaryHead use SyncBN for DDP)

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:      2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:     AdamW, lr=0.00006 (upstream SegNeXt)
  Calibrated:    2026-03-07 (single-GPU fwd+bwd probe at 768×384, A6000 48 GB):
    bs=40: 40.82 GB | bs=48: 43.64 GB | bs=52: 44.26 GB | bs=54: OOM (47 GB needed)
    OOM boundary ≈ 54/GPU. 85% target → 45.9 → rounded UP to 48 (88.9%, multiple of 8).
  samples_per_gpu = 48,  total_bs = 96 (48 × 2 GPUs)
  LR:            0.00006 (fixed — AdamW does not use linear scaling, same as EXP-006)
  Note: UPerHead (4 feature maps + PPM) is heavier than LightHamHead; OOM boundary
        dropped from ~75/GPU (HAM, Phase 1) to ~54/GPU (UPerHead + BoundaryHead).
  Norm: backbone uses BN; UPerHead and BoundaryHead use SyncBN (required for DDP BN layers).

WandB run name: phase2_segnext_t_uperhead  (suffix "uperhead" distinguishes from EXP-006)
Output dir:     outputs/phase2/segnext_t_uperhead/

Phase 2 schedule:
  5k iters (same as EXP-006). Eval every 250 iters → 20 checkpoints.
  Scheduler: LinearLR (500 iters warmup) + PolyLR (same as EXP-006).

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh segnext_t_uperhead
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
# Unique suffix "uperhead" to distinguish from EXP-006 (segnext_t with HAM decoder).
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase2_segnext_t_uperhead',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Norms ─────────────────────────────────────────────────────────────────────
# MSCAN backbone: BN (its design; stays BN even under DDP — follows EXP-006)
# UPerHead + BoundaryHead: SyncBN required for DDP (these are BN-based heads)
backbone_norm_cfg = dict(type='BN', requires_grad=True)
head_norm_cfg = dict(type='SyncBN', requires_grad=True)

# ── PIDNet-style train pipeline (adds GenerateEdge for BoundaryLoss) ──────────
crop_size = (384, 768)
img_scale = (768, 384)

uperhead_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # GenerateEdge: produces gt_edge_map for MSCANBoundaryHead + BoundaryLoss
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

# ── Dataloader ─────────────────────────────────────────────────────────────────
# CALIBRATED 2026-03-07: bs=48/GPU (88.9% of OOM boundary ~54/GPU). total_bs=96.
train_dataloader = dict(
    batch_size=48,
    dataset=dict(pipeline=uperhead_train_pipeline))

# ── Model ─────────────────────────────────────────────────────────────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa

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
        norm_cfg=backbone_norm_cfg),    # BN (backbone design)
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=head_norm_cfg,         # SyncBN (DDP)
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='MSCANBoundaryHead',
        in_channels=32,                 # MSCAN stage-0 output (stride 4)
        in_index=0,
        channels=64,
        num_classes=1,                  # binary boundary logit
        norm_cfg=head_norm_cfg,         # SyncBN (DDP)
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=False,
        loss_decode=dict(type='BoundaryLoss', loss_weight=20.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Optimizer (AdamW — fixed LR, no linear scaling) ───────────────────────────
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

# ── Schedule (same as EXP-006: LinearLR 500iters + PolyLR) ───────────────────
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
    val_interval=250)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        save_best='mIoU',
        rule='greater'),
)
