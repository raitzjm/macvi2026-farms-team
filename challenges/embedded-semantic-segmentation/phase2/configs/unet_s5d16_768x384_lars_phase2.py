"""
UNet-s5-d16 (FCN head, CE+Dice) @ 768×384 — LaRS — Phase 2 full training  [added late to Phase 2]
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Reference: mmseg 1.2.2 upstream unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
  mmsegmentation-macvi (Phase 1 fork) also has: fcn_unet_s5-d16_4x4_512x1024_160k_lars.py

Architecture: UNet 5-stage encoder-decoder + FCNHead (1-conv) main head
  + FCNHead auxiliary head (loss_weight=0.4).
  FCNHead chosen over ASPPHead (DeepLabv3) to stay closest to original U-Net:
    original U-Net uses a 1×1 conv output = FCNHead(num_convs=1).
    ASPPHead adds dilated multi-scale pooling not present in original U-Net.
  Primary loss: CE(1.0) + Dice(3.0) — standard from upstream fcn_ce-dice variant.
    Dice weight=3.0 (not 1.0) matches the upstream mmseg convention for medical/seg tasks.
  Auxiliary loss: CE only (loss_weight=0.4, standard).
  No pretrained backbone (UNet is trained from scratch).

Batch size / LR (AdamW — LR fixed, no linear scaling):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Optimizer:   AdamW (same as SegFormer/SegNeXt); original SGD reference not used.
  LR:          6e-5 (same as all AdamW models in Phase 2; no scaling).
  Input size:  768×384 (LaRS) vs 64×64 (DRIVE) — much larger; OOM boundary unknown.
  Calibration: test fwd+bwd at bs=4, 6, 8 per GPU on A6000 48 GB.
    Starting estimate: 4/GPU (total_bs=8) — conservative due to 768×384 resolution
    and UNet full encoder-decoder (large memory footprint at this resolution).
  NOTE: Calibrate BS before training; update samples_per_gpu and issue.

  SyncBN required for DDP (UNet uses BN in all stages).

Phase 2 schedule:
  5k iters. UNet is new in Phase 2 (no Phase 1 cross-check).
  Do trial runs before committing to full training.
  Eval every 250 iters → 20 checkpoints.
  If not converged at 5k, extend to 7k.

  test_cfg mode='whole': do NOT use slide mode (DRIVE default) at 768×384.

Run:
  source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh unet

  !! Before first run: calibrate batch size fwd+bwd at bs=4, 6, 8/GPU        !!
  !! Then update samples_per_gpu, lr, and issue with pinned values.           !!
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
            name='phase2_unet_adamw',
        )),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (UNet-s5-d16 — architecture-specific, do not change) ────────────────
# SyncBN required for DDP (UNet uses BatchNorm).
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
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=3.0),
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # Use whole-image inference — NOT slide mode (DRIVE's 64×64 default)
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# CALIBRATED 2026-03-04 (single-GPU fwd+bwd, BN, 768×384, A6000 48 GB):
#   bs=4→10.1GB  bs=8→19.9GB  bs=12→29.7GB  bs=14→34.6GB
#   bs=16→39.5GB(82%)  bs=18→44.4GB(93%)  bs=20→OOM
# OOM boundary ≈ 19/GPU. 20% reduction → bs=16/GPU (total_bs=32, 82% utilization).
train_dataloader = dict(batch_size=16)

# ── Optimizer (AdamW — same as SegFormer/SegNeXt in Phase 2) ─────────────────
# LR=6e-5 fixed (AdamW; no linear scaling). Same for all Phase 2 AdamW models.
# UNet trains from scratch — standard AdamW weight decay.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01))

# ── Schedule ──────────────────────────────────────────────────────────────────
# 5k iters; warmup 500 iters (AdamW standard; PolyLR linear decay).
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
