"""
Base dataset config for Phase 3 LaRS training.
mmseg 1.x API: train/val dataloaders with Phase 2.5 pipeline fixes +
Phase 3 augmentations (CopyPasteObstacles, EnhancedPhotoMetricDistortion).

Pipeline fixes carried forward from Phase 2.5:
  Fix 1: RandomResize keep_ratio=True in train pipeline (preserves AR).
  Fix 2: img_padding_size in meta_keys (de-pad before metric computation).
  Fix 3: ReflectPad center=True in test/val (centred padding, matches server).

Phase 3 additions:
  - CopyPasteObstacles: paste external obstacles onto water regions.
  - EnhancedPhotoMetricDistortion: wider bounded augmentation ranges + gamma.

PIDNet-S overrides train_dataloader in its own config to insert
GenerateEdgeMasked; all other models use the train_pipeline defined here.
"""

DB_PATH = ('challenges/common/external-dataset-analysis/'
           'shared_data/copypaste_candidates.db')
DATA_ROOT = '/home/farms/lars_linked/'

crop_size = (384, 768)   # (H, W)
img_scale = (768, 384)   # (W, H)

# ── Pipelines ─────────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # Copy-paste before photometric distortion so pasted crops get distorted too
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
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='ReflectPad', size=crop_size, center=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'img_padding_size')),
]

# ── Dataloaders ────────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,            # placeholder — overridden per model config
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

# ── Evaluators ─────────────────────────────────────────────────────────────────
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
