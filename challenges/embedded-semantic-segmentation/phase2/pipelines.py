"""
Custom mmseg pipeline transforms and model heads for Phase 2+ configs
(mmseg 1.x API).

ReflectPad: drop-in replacement for Pad that uses mirror (reflect) padding
instead of zero-padding.  mmcv.impad supports padding_mode natively; this
class simply exposes that option through the mmseg 1.x TRANSFORMS registry.

MSCANBoundaryHead: lightweight auxiliary boundary supervision head for
MSCAN+UPerHead experiments.  Takes the stride-4 MSCAN feature (in_index=0,
32 channels) and predicts a binary boundary map supervised by BoundaryLoss
against the edge ground truth produced by GenerateEdge in the train pipeline.

Usage in config:
    custom_imports = dict(
        imports=[
            'challenges.embedded-semantic-segmentation.phase2.datasets.lars_dataset',
            'challenges.embedded-semantic-segmentation.phase2.pipelines',
        ],
        allow_failed_imports=False)

    # In train pipeline (replace the regular Pad):
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255)

    # In test pipeline:
    dict(type='ReflectPad', size=crop_size)

    # As auxiliary boundary head (requires GenerateEdge in train pipeline):
    auxiliary_head=dict(
        type='MSCANBoundaryHead',
        in_channels=32, in_index=0, channels=64, num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=False,
        loss_decode=dict(type='BoundaryLoss', loss_weight=20.0))
"""

import torch
import torch.nn as nn
import mmcv
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS, TRANSFORMS
from mmseg.utils import SampleList


@TRANSFORMS.register_module()
class ReflectPad:
    """Pad image and mask with mirror (reflect) padding.

    Wraps mmcv.impad with padding_mode='reflect'.  The padding area is filled
    by mirroring the image content, matching the challenge server behaviour.

    Args:
        size (tuple): Target (H, W) after padding.
        seg_pad_val (int): Value used to pad segmentation masks.
            For training, use 255 (ignore index).
            For inference, masks are not present so this has no effect.
    """

    def __init__(self, size, seg_pad_val=255):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = tuple(size)          # (H, W)
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results):
        img = results['img']
        h, w = img.shape[:2]
        target_h, target_w = self.size

        # If image already fits (or is larger), fall back to constant padding
        # to avoid reflect-pad errors on tiny images (reflect requires padding
        # < image dimension).
        if h >= target_h and w >= target_w:
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['pad_shape'] = img.shape
            return

        need_reflect = (target_h - h) < h and (target_w - w) < w
        mode = 'reflect' if need_reflect else 'constant'

        padded = mmcv.impad(img, shape=self.size, padding_mode=mode)
        results['img'] = padded
        results['img_shape'] = padded.shape[:2]
        results['pad_shape'] = padded.shape

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            seg = results[key]
            h, w = seg.shape[:2]
            target_h, target_w = self.size
            if h >= target_h and w >= target_w:
                return
            need_reflect = (target_h - h) < h and (target_w - w) < w
            mode = 'reflect' if need_reflect else 'constant'
            # mmcv.impad handles 2-D (H, W) and 3-D (H, W, C) masks natively.
            # Do NOT add a channel dim: impad squeezes C=1 back out, breaking
            # the shape (observed: (H,W,1) → impad → (H,W) → [..0] → (H,)).
            padded = mmcv.impad(
                seg,
                shape=self.size,
                pad_val=self.seg_pad_val,
                padding_mode=mode)
            results[key] = padded

    def __call__(self, results):
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'size={self.size}, seg_pad_val={self.seg_pad_val})')


@MODELS.register_module()
class MSCANBoundaryHead(BaseDecodeHead):
    """Auxiliary boundary supervision head for MSCAN + UPerHead experiments.

    Takes one MSCAN feature map (typically in_index=0, stride 4, 32 channels)
    and predicts a 1-channel binary boundary map.  Supervised by BoundaryLoss
    against the ground-truth edge map produced by GenerateEdge in the training
    pipeline (stored in data_sample.gt_edge_map).

    The main UPerHead handles semantic segmentation with CrossEntropyLoss;
    this auxiliary head adds boundary supervision without modifying the
    UPerHead architecture.

    Args:
        All args inherited from BaseDecodeHead.  Typical values:
          in_channels=32, in_index=0, channels=64, num_classes=1,
          norm_cfg=dict(type='BN', requires_grad=True),
          act_cfg=dict(type='ReLU', inplace=True),
          loss_decode=dict(type='BoundaryLoss', loss_weight=20.0)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Single 3×3 conv: in_channels → channels, then cls_seg: channels → 1
        self.conv_bn_relu = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.conv_bn_relu(x)
        return self.cls_seg(x)   # shape: (B, 1, H', W')

    def _stack_batch_gt(self, batch_data_samples: SampleList):
        """Return stacked gt_edge_map tensors (not gt_sem_seg)."""
        gt_edge_segs = [
            data_sample.gt_edge_map.data
            for data_sample in batch_data_samples
        ]
        return torch.stack(gt_edge_segs, dim=0)

    def loss_by_feat(self, seg_logits, batch_data_samples: SampleList):
        """Compute BoundaryLoss against gt_edge_map.

        Overrides BaseDecodeHead.loss_by_feat to avoid passing ignore_index
        to BoundaryLoss (which does not accept that argument).
        """
        bd_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        bd_label = bd_label.squeeze(1)   # (B, 1, H, W) → (B, H, W)
        loss = dict()
        losses_decode = (
            self.loss_decode
            if isinstance(self.loss_decode, nn.ModuleList)
            else [self.loss_decode])
        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(seg_logits, bd_label)
        return loss
