"""
Base dataset config for Phase 2.5 LaRS training.
mmseg 1.x API: train/val dataloaders with ReflectPad preprocessing.

Phase 2.5 pipeline fixes vs Phase 2:
  Fix 1: RandomResize keep_ratio=True in train pipeline (was False in Phase 2 —
          that was an undocumented regression; keep_ratio=True preserves AR and
          matches the challenge server and test pipeline preprocessing).
  Fix 3: ReflectPad in test/val pipeline uses center=True (centred padding),
          matching the challenge server's centred mirror-pad behaviour.
          Training pipeline keeps center=False (bottom-right only; RandomCrop
          handles sizing so centering there is irrelevant).

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
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# ── Pipelines ─────────────────────────────────────────────────────────────────
# Fix 1: keep_ratio=True in training — preserves image AR during random resize,
#        matching challenge server behaviour. Phase 2 used keep_ratio=False
#        (a regression vs Phase 1). RandomCrop then crops to exact crop_size.
# Fix 3 (train): center=False in ReflectPad — RandomCrop handles exact sizing;
#        centring during training is not meaningful.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,           # (W, H) base scale for resize jitter
        ratio_range=(0.5, 2.0),
        keep_ratio=True),          # Fix 1: preserve AR (was False in Phase 2)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # Reflect padding matches challenge server mirror-padding; center=False
    # (default) — bottom-right only for training.
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    # Fix 2: include img_padding_size in meta_keys so mmengine can de-pad
    # predictions before metric computation (default meta_keys omits it).
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

# Fix 3 (test): center=True — split padding evenly on both sides, matching the
#               challenge server's centred mirror-pad preprocessing.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # Fix 3: centred reflect-pad to exact crop_size after AR-preserving resize.
    dict(type='ReflectPad', size=crop_size, center=True),
    dict(type='LoadAnnotations'),
    # Fix 2: include img_padding_size in meta_keys for correct val de-padding.
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

# ── Dataloaders (batch_size overridden per model config) ──────────────────────
train_dataloader = dict(
    batch_size=2,           # placeholder — overridden in model config
    num_workers=8,
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
