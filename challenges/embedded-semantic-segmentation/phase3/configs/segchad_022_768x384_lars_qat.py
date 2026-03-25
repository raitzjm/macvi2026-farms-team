"""
SegChad @ 768×384 — LaRS — Phase 3 QAT (QEXP-009)
Fine-tuning for 2000 iterations to recover INT8 performance.

Checkpoint: outputs/phase3/segchad_022/best_mIoU_iter_17000.pth
Baseline INT8: mIoU=90.27%, F1=74.94%, Q=67.65% (ΔQ=-6.90% vs FP32)

Quantizer: TorchNativeQuantizer (same as SegNeXt QAT configs — AdamW architecture).
norm_cfg: BN (not SyncBN) — single GPU, no DDP for QAT fine-tuning.

Skipped module classes:
  - timm.models.convnext.GlobalResponseNorm: uses torch.norm with dynamic dims,
    not traceable. Element-wise multiply/divide — INT8-safe, no need to quantize.
  - timm.layers.norm.LayerNorm2d: channel-dim LayerNorm — INT8-safe, skip tracing.
  Standard mmseg segmentor methods skipped as in all QAT configs.

If tracing fails with an unrecognised class, add its full dotted path to
skipped_module_classes and retry.
"""

_base_ = [
    './_base_/lars_dataset.py',
    './_base_/runtime.py',
]

# ── Custom Imports ─────────────────────────────────────────────────────────────
custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase3.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase3.pipelines',
        'challenges.embedded-semantic-segmentation.phase3.models',
    ],
    allow_failed_imports=False)

# ── Model ──────────────────────────────────────────────────────────────────────
checkpoint = ('challenges/embedded-semantic-segmentation/outputs/phase3/'
              'segchad_022/best_mIoU_iter_17000.pth')

# Switch SyncBN → BN: QAT runs single-GPU (no DDP), SyncBN unnecessary.
norm_cfg = dict(type='BN', requires_grad=True)

segchad_model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        _scope_='mmseg',
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(384, 768)),
    backbone=dict(
        type='SegChadBackbone',
        model_name='convnextv2_femto.fcmae_ft_in1k',
        pretrained=False,   # weights loaded from float_checkpoint
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type='SegChadHead',
        in_channels=[48, 96, 192, 384],
        channels=256,
        num_classes=3,
        d_channels=64,
        p_channels=128,
        i_channels=256,
        decoder_mid_channels=128,
        decoder_out_channels=64,
        boundary_loss_weight=10.0,
        boundary_width=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ── QAT model wrapper ──────────────────────────────────────────────────────────
# train_qat.py reads model.architecture + model.float_checkpoint, then applies
# prepare_qat_model.wrap_model_qat (manual FakeQuant, pure PyTorch — no mmrazor).
# The quantizer block is not instantiated by train_qat.py and is omitted here.
model = dict(
    architecture=dict(**segchad_model),
    float_checkpoint=checkpoint)

# ── Training ───────────────────────────────────────────────────────────────────
# Use base train pipeline (EXP-022 moderate LSJ, no GenerateEdge).
# No CopyPaste override — base pipeline includes it; fine for QAT fine-tuning.
train_dataloader = dict(
    batch_size=2,
    num_workers=2)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=100, end=2000,
         by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=250)
val_cfg = dict(type='ValLoop')
test_cfg = val_cfg

# ── Runtime ────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        rule='greater'),
    sync=dict(type='SyncBuffersHook'))

visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
