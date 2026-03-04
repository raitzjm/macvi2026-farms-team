"""
SegNeXt-S @ 768×384 — LaRS — Phase 2 full training  [added late to Phase 2]
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Batch size / LR (AdamW — keep LR fixed regardless of batch size):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   AdamW, lr=0.00006 (upstream SegNeXt-S, same as T)
  Per-GPU:     samples_per_gpu=48 (starting estimate; calibrate before first run)
               SegNeXt-T was limited to 56/GPU by HAM decoder; S is larger so
               start at 48. Calibrate fwd+bwd at 48, 56 to find OOM boundary.
  Effective:   total_bs=96 (48 × 2 GPUs) — update after calibration
  LR:          0.00006 (fixed — AdamW does not use linear scaling)
  Norm: LayerNorm — SyncBN not applicable
  NOTE: After calibration, update samples_per_gpu, total_bs comment, and issue.

Phase 2 schedule:
  5k iters. SegNeXt-S is new in Phase 2 (no Phase 1 cross-check).
  Eval every 250 iters → 20 checkpoints.
  If not converged at 5k, extend to 7k.

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh segnext_s

  !! Before first run: calibrate batch size by testing fwd+bwd at bs=48, 56 !!
  !! Then update samples_per_gpu and total_bs comment accordingly.          !!
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
            name='phase2_segnext_s',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (SegNeXt-S — architecture-specific, do not change) ──────────────────
# SegNeXt-S vs SegNeXt-T: larger backbone embed_dims & depths, wider decode head.
# Pretrained from ImageNet-1K classification checkpoint.
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'  # noqa
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
        embed_dims=[64, 128, 320, 512],     # S: wider than T [32, 64, 160, 256]
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],               # S: shallower per-stage than T [3,3,5,2]
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],        # S: from stages 1,2,3 (vs T: [64,160,256])
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
# Starting estimate: 48/GPU (total_bs=96). Update after calibration.
train_dataloader = dict(batch_size=48)

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
# 5k iters; warmup 500 iters.
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

# ── Hooks override ────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,          # save checkpoint every 2500 iters
        save_best='mIoU',
        rule='greater'),
)
