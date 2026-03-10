"""
STDC2 @ 768×384 — LaRS — Phase 2.5 pipeline-fix verification
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

Phase 2.5 purpose: re-train with pipeline fixes (keep_ratio=True in training,
centred val padding, img_padding_size de-padding) to measure their effect vs
Phase 2 baseline. All model/optimizer/scheduler settings are IDENTICAL to
Phase 2 EXP-007 (stdc2_768x384_lars_phase2.py).

Batch size / LR (SGD — linear scaling rule: new_lr = base_lr × (new_bs / ref_bs)):
  Hardware:    2× RTX A6000 48 GB (DDP, 2 GPUs)
  Reference:   bs=12, lr=0.01 (stdc1_in1k-pre_512x1024_80k_lars.py upstream)
  Per-GPU:     samples_per_gpu=160  (pinned from Phase 1)
  Effective:   total_bs=320 (160 × 2 GPUs), lr=0.267 (= 0.01 × 320/12)
  Norm: SyncBN required for DDP (BN stats synchronized across GPUs)
  Pinned from Phase 1 — do not re-calibrate.

Phase 2.5 schedule (identical to Phase 2 EXP-007):
  3k iters; eval every 250 iters → 12 checkpoints.

Run:
  source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh stdc2
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2_5.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2_5.pipelines',
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
            name='phase2_5_stdc2',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Model (STDC2: STDCContextPathNet + STDCNet2 backbone) ─────────────────────
checkpoint_stdc2 = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
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
        type='STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet2',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_stdc2)),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=3,
        ignore_index=255,
        in_index=3,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            ignore_index=255,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            ]),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Dataloader (batch_size overrides _base_) ──────────────────────────────────
# bs=160 calibrated on RTX A6000 48 GB at 768×384 (pinned from Phase 1).
train_dataloader = dict(batch_size=160)

# ── Optimizer (SGD — identical to Phase 2 EXP-007, do not change) ─────────────
# total_bs=320 (160×2 GPUs), lr=0.267: linear-scaled from ref (bs=12, lr=0.01).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.267,
        momentum=0.9,
        weight_decay=0.0005))

# ── Schedule (identical to Phase 2 EXP-007) ───────────────────────────────────
# 3k iters; warmup 300 iters (scaled from Phase 1: 150 × 2).
iters = 3000
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
