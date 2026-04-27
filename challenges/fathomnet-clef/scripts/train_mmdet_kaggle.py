#!/usr/bin/env python3
"""
MMDetection training script for FathomNet-CLEF on Kaggle using DeformableDETR
Usage: python train_mmdet_kaggle.py --data-path /kaggle/input/fathomnet-2026 --wandb-key YOUR_KEY
"""
import os
import argparse
import json
from pathlib import Path

def prepare_coco_annotations(data_path, output_path):
    """Convert COCO annotations with proper category ID mapping."""
    with open(f'{data_path}/train_dataset.json') as f:
        data = json.load(f)
    
    # Map 1-41 -> 0-31
    cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {c['id']: i for i, c in enumerate(cats)}
    
    # Update annotations
    for ann in data['annotations']:
        ann['category_id'] = cat_id_to_idx[ann['category_id']]
    
    # Update categories
    data['categories'] = [{'id': i, 'name': c['name']} for i, c in enumerate(cats)]
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return len(data['images'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/kaggle/input/fathomnet-2026')
    parser.add_argument('--wandb-key', default=os.environ.get('WANDB_API_KEY', ''))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    
    os.environ['WANDB_API_KEY'] = args.wandb_key
    
    # Prepare annotations with fixed category IDs
    ann_path = '/tmp/fathomnet_train.json'
    prepare_coco_annotations(args.data_path, ann_path)
    print(f"Prepared annotations: {ann_path}")
    
    # Initialize wandb
    import wandb
    wandb.init(entity='little_farms', project='fathomnet2026', name='mmdet-kaggle')
    
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Import mmdet to register models
    import mmdet.models
    from mmengine.config import Config
    from mmengine.runner import Runner
    
    cfg = Config()
    
    # DeformableDETR config - simplified for smoke test
    cfg.model = dict(
        type='DeformableDETR',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32
        ),
        backbone=dict(
            type='ResNet',
            depth='50',
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'
        ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4
        ),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            dim=256,
            num_heads=8,
            dropout=0.1
        ),
        decoder=dict(
            type='DetrTransformerDecoder',
            num_layers=6,
            dim=256,
            num_heads=8,
            dropout=0.1,
            return_intermediate=True
        ),
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=128),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(type='HungarianAssigner', cls_cost=dict(type='FocalLossCost', weight=2.0), 
                        reg_cost=dict(type='BBoxL1Cost', weight=5.0), iou_cost=dict(type='IoUCost', weight=2.0))
        ),
        test_cfg=dict(max_per_img=100)
    )
    
    metainfo = dict(classes=['amphipod', 'anemone', 'barnacle', 'benthic worm', 'bivalve', 
        'black coral', 'bony fish', 'brittle star', 'calycophoran siphonophore', 'chiton', 
        'crab', 'feather star', 'hydroid', 'isopod', 'jelly', 'larvacean', 'octopus', 
        'physonect siphonophore', 'pyrosome', 'sea cucumber', 'sea fan', 'sea pen', 
        'sea slug', 'sea snail', 'sea squirt', 'sea star', 'shrimp', 'soft coral', 
        'sponge', 'squat lobster', 'stony coral', 'urchin'])
    
    cfg.train_dataloader = dict(
        batch_size=args.batch,
        num_workers=2,
        dataset=dict(
            type='CocoDataset',
            data_root=args.data_path,
            metainfo=metainfo,
            ann_file=ann_path,
            data_prefix=dict(img='train_set'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', scale=(480, 480)),
                dict(type='PackDetInputs')
            ]
        )
    )
    
    cfg.val_dataloader = dict(
        batch_size=args.batch,
        num_workers=2,
        dataset=dict(
            type='CocoDataset',
            data_root=args.data_path,
            metainfo=metainfo,
            ann_file=ann_path,  # Use same for val since test has no labels
            data_prefix=dict(img='train_set'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(480, 480)),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='PackDetInputs')
            ]
        )
    )
    
    cfg.test_dataloader = cfg.val_dataloader
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=args.epochs, val_interval=1)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.val_evaluator = dict(type='CocoMetric')
    cfg.test_cfg = dict(type='TestLoop')
    cfg.test_evaluator = dict(type='CocoMetric')
    
    cfg.optim_wrapper = dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
    )
    
    cfg.default_hooks = dict(
        logger=dict(type='LoggerHook', interval=10),
        checkpoint=dict(type='CheckpointHook', interval=1)
    )
    
    cfg.launcher = 'none'
    cfg.work_dir = '/tmp/mmdet_output'
    
    print(f"Starting DeformableDETR training for {args.epochs} epochs...")
    print(f"WandB: https://wandb.ai/little_farms/fathomnet2026/runs/{wandb.run.id}")
    
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    print("Training complete!")
    wandb.finish()

if __name__ == '__main__':
    main()