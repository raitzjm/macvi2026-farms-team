"""
Base dataset config for Phase 2 LaRS training.
mmseg 1.x API: train/val dataloaders with ReflectPad preprocessing.

Data layout (symlink tree, rebuilt each container session):
  /home/farms/lars_linked/{train,val}/images/*.jpg
  /home/farms/lars_linked/{train,val}/semantic_masks/*.png
  /home/farms/lars_linked/{train,val}/image_list.txt

Custom imports are specified per-model config (not here) because they
must appear at the top level of the final config, not inside _base_.
"""

DATA_ROOT = '/home/farms/lars_linked/'

crop_size = (384, 768)   # (H, W) — target deployment resolution
img_scale = (768, 384)   # (W, H) — for mmseg Resize

# ── data_preprocessor (merged into model dict per config) ─────────────────────
# Normalization is done in the model's data_preprocessor, NOT in the pipeline.
# Override pad_size_divisor if needed (not used here — fixed crop_size).
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# ── Pipelines ─────────────────────────────────────────────────────────────────
# ReflectPad matches challenge server: mirror padding after keep_ratio resize.
# RandomResize with keep_ratio=False: stretch to target AR, then crop.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,           # (W, H) base scale for resize jitter
        ratio_range=(0.5, 2.0),
        keep_ratio=False),         # stretch to target AR (standard mmseg training)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # FIX: reflect padding matches challenge server mirror-padding
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # FIX: keep_ratio=True preserves aspect ratio (challenge server behaviour)
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # FIX: reflect-pad to exact crop_size after aspect-ratio resize
    dict(type='ReflectPad', size=crop_size),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# ── Dataloaders (batch_size overridden per model config) ──────────────────────
train_dataloader = dict(
    batch_size=2,           # placeholder — overridden in model config
    num_workers=4,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'train',
        data_prefix=dict(img_path='images', seg_map_path='semantic_masks'),
        ann_file='image_list.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LaRSDataset',
        data_root=DATA_ROOT + 'val',
        data_prefix=dict(img_path='images', seg_map_path='semantic_masks'),
        ann_file='image_list.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ── Evaluators ────────────────────────────────────────────────────────────────
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
