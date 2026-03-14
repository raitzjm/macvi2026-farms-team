"""
PIDChad-S @ 768×384 — LaRS — Phase 3 augmentation training (EXP-021)
mmsegmentation 1.2.2 + mmcv 2.1.0 + PyTorch 2.1.2+cu121
Conda env: macvi2  |  /home/farms/miniforge3/envs/macvi2/

EXP-021 introduces PIDChad-S — PIDNet-S with two structural enhancements:
  - AFFM: I-branch context gates P-branch before LightBag (kills reflection FPs)
  - FPPM: parallel pyramid pooling at scales (1,2,3,6,12) replaces PAPPM
    (stronger global anchor for waterline estimation)

Both modules use only INT8-safe ops. See phase3/models/pidchad.py.

Training recipe extends EXP-020:
  - Aggressive LSJ (0.1, 2.0): 48.6% of FNs are <100px — extreme small-scale signal
  - Obstacle class upweighted (1.5×) in all semantic losses
  - BoundaryLoss weight 25.0 (up from 20.0) for waterline quality
  - Extended schedule: 25000 iters, warmup 5000 iters

Hardware: 2× RTX A6000 48 GB (DDP)
  samples_per_gpu=160, total_bs=320, lr=0.133 (SGD linear-scaled, same as EXP-020)

Schedule:
  25000 iters, eval every 500 iters, save best mIoU.

WandB run name: phase3_pidchad_s_021

Baselines to beat:
  EXP-020 (PIDNet-S Phase 3): local val Q=70.51%, server Q=52.1%

Run (BS/LR probe first — see Step 3 in issue #52):
  # OOM probe (5 iters):
  torchrun --standalone --nproc_per_node=2 <mmseg_train> \\
      phase3/configs/pidchad_s_768x384_lars_phase3_exp021.py \\
      --work-dir /tmp/pidchad_probe --cfg-options train_cfg.max_iters=5

  # INT8 smoke test (500 iters — see Step 4 in issue #52):
  torchrun --standalone --nproc_per_node=2 <mmseg_train> \\
      phase3/configs/pidchad_s_768x384_lars_phase3_exp021.py \\
      --work-dir /tmp/pidchad_smoke --cfg-options train_cfg.max_iters=500

  # Full training:
  source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh
  bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh pidchad_s_021
  # (run_phase3.sh maps model→config as {model}_768x384_lars_phase3.py and
  #  work-dir as outputs/phase3/{model}. Model key: pidchad_s_021)
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase3.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase3.pipelines',
        'challenges.embedded-semantic-segmentation.phase3.models',
    ],
    allow_failed_imports=False)

# ── WandB run name ─────────────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase3_pidchad_s_021',
        )),
]
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Train pipeline (same as EXP-020 + aggressive LSJ (0.1, 2.0)) ──────────────
crop_size = (384, 768)
img_scale = (768, 384)
DB_PATH = ('challenges/common/external-dataset-analysis/'
           'shared_data/copypaste_candidates.db')

pidnet_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Aggressive LSJ: (0.1, 2.0) — extreme small-scale signal for tiny obstacle recall
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # CopyPaste before photometric distortion so pasted crops get distorted too
    dict(type='CopyPasteObstacles',
         db_path=DB_PATH,
         n_paste=(1, 3),
         prob_apply=0.7,
         scale_range=(0.5, 1.5),
         prefer_waterline_prob=0.3,
         boundary_blur_radius=2,
         match_brightness=True),
    dict(type='EnhancedPhotoMetricDistortion',
         brightness_delta=77,
         contrast_range=(0.6, 1.5),
         saturation_range=(0.6, 1.5),
         hue_delta=27,
         gamma_range=(0.6, 1.6),
         gamma_prob=0.3),
    # ReflectPad writes img_padding_size BEFORE GenerateEdgeMasked reads it
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # GenerateEdgeMasked: zeros edge targets in padded regions (Option 1 fix)
    dict(type='GenerateEdgeMasked', edge_width=4),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

train_dataloader = dict(
    batch_size=160,
    dataset=dict(pipeline=pidnet_train_pipeline))

# ── Model (PIDChad-S) ──────────────────────────────────────────────────────────
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
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
        type='PIDChadNet',           # PIDNet-S + AFFM + FPPM
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        fppm_scales=[1, 2, 3, 6, 12],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDChadHead',          # PIDHead subclass for config identification
        in_channels=128,
        channels=128,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            # P-branch semantic loss (obstacle upweighted 1.5×)
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                 class_weight=[1.5, 1.0, 1.0]),
            # I-branch OHEM loss (obstacle upweighted 1.5×)
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0,
                 class_weight=[1.5, 1.0, 1.0]),
            # Boundary/waterline loss (increased 20→25 for WE_IoU improvement)
            dict(type='BoundaryLoss', loss_weight=25.0),
            # Boundary-conditioned OHEM on I-branch (obstacle upweighted 1.5×)
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0,
                 class_weight=[1.5, 1.0, 1.0]),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── Optimizer (SGD — same as EXP-020) ────────────────────────────────────────
# lr=0.133 (0.01×160/12 linear scaling from PIDNet-S ref); same backbone capacity.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.133, momentum=0.9, weight_decay=0.0005))

# ── Schedule (25k iters, warmup 5000) ────────────────────────────────────────
# Extended from EXP-020 (15k): aggressive LSJ (0.1, 2.0) needs more iterations
# to recover from augmentation shock and for longer-tail small-scale learning.
# Warmup doubled to 5000 (was 2500): aggressive LSJ causes stronger aug shock.
iters = 25000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=5000, by_epoch=False),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=5000, end=iters, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,        # save checkpoint every 5000 iters
        save_best='mIoU',
        rule='greater'),
)
