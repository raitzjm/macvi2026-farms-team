"""
SegNeXt-T @ 768×384 — LaRS — Phase 3 Quantization-Aware Training (QAT)
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
# We need to import the original SegNeXt config to get the architecture
# and then wrap it in MMRazor's ArchitectureQuant.

# Load original model config (copied from phase2/configs/segnext_t_768x384_lars_phase2.py)
checkpoint = 'challenges/embedded-semantic-segmentation/phase2/artifacts/segnext_t/best_mIoU_iter_4250.pth'

norm_cfg = dict(type='BN', requires_grad=True) # Switch to BN for fine-tuning
mscan_t_model = dict(
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
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_S=1, MD_R=16, train_steps=6, eval_steps=7, rand_init=False),
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
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
        **mscan_t_model),
    data_preprocessor=dict(
        _scope_='mmseg',
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(384, 768)),
    float_checkpoint=checkpoint,
    quantizer=dict(
        type='mmrazor.TorchNativeQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmseg.models.segmentors.encoder_decoder.EncoderDecoder.loss',
                'mmseg.models.segmentors.encoder_decoder.EncoderDecoder.predict',
                'mmseg.models.segmentors.base.BaseSegmentor.forward',
            ],
            skipped_module_classes=[
                'mmcv.cnn.bricks.drop.DropPath',
                'mmseg.models.decode_heads.ham_head.LightHamHead',
                'mmseg.models.decode_heads.ham_head.NMF2D',
                'mmseg.models.backbones.mscan.MSCAAttention'
            ])))




# ── Training ──────────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2, # Small batch size for 4GB VRAM
    num_workers=2)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
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
    type='MacViSegVisualizer', # Use the fixed visualizer from Phase 2.5
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
