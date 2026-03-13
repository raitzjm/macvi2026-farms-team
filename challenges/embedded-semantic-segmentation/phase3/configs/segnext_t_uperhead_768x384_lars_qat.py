"""
MSCAN-T + UPerHead + BoundaryLoss @ 768×384 — LaRS — Phase 3 QAT
Fine-tuning for 2000 iterations to recover INT8 performance.
"""

_base_ = [
    '../../phase2_5/configs/_base_/lars_dataset.py',
    '../../phase2/configs/_base_/runtime.py'
]

# ── Custom Imports ────────────────────────────────────────────────────────────
custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2_5.pipelines',
    ],
    allow_failed_imports=False)

# ── Model ─────────────────────────────────────────────────────────────────────
checkpoint = 'challenges/embedded-semantic-segmentation/phase2/artifacts/segnext_t_uperhead/best_mIoU_iter_4250.pth'

norm_cfg = dict(type='BN', requires_grad=True)
mscan_t_uperhead_model = dict(
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
        type='MSCAN',
        init_cfg=None,
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=160,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# QAT Configuration
global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

model = dict(
    _scope_='mmrazor',
    type='MMArchitectureQuant',
    architecture=dict(
        _scope_='mmseg',
        **mscan_t_uperhead_model),
    float_checkpoint=checkpoint,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmseg.models.segmentors.encoder_decoder.EncoderDecoder.loss',
                'mmseg.models.segmentors.encoder_decoder.EncoderDecoder.predict',
                'mmseg.models.segmentors.base.BaseSegmentor.forward',
                'mmcv.cnn.bricks.drop.DropPath.forward'
            ])))

# ── Training ──────────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,
    num_workers=2)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=100, end=2000, by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=250)

val_cfg = dict(type='ValLoop')
test_cfg = val_cfg

# ── Runtime ───────────────────────────────────────────────────────────────────
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
