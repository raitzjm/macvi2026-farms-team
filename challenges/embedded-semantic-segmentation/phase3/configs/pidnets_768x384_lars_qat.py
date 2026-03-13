"""
PIDNet-S @ 768×384 — LaRS — Phase 3 QAT
Fine-tuning for 2000 iterations to achieve peak INT8 accuracy.
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
checkpoint = 'challenges/embedded-semantic-segmentation/phase2/artifacts/pidnets/best_mIoU_iter_4750.pth'

norm_cfg = dict(type='BN', requires_grad=True)
pidnet_s_model = dict(
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
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=None),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=32768, loss_weight=1.0),
        ]),
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
        **pidnet_s_model),
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
# PIDNet requires GenerateEdge in the train pipeline
pidnet_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(768, 384), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(384, 768), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ReflectPad', size=(384, 768), seg_pad_val=255),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=pidnet_train_pipeline))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=100, end=2000, by_epoch=False)
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
