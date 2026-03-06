"""
Custom mmseg pipeline transforms for Phase 2+ configs (mmseg 1.x API).

ReflectPad: drop-in replacement for Pad that uses mirror (reflect) padding
instead of zero-padding.  mmcv.impad supports padding_mode natively; this
class simply exposes that option through the mmseg 1.x TRANSFORMS registry.

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
"""

import mmcv
from mmseg.registry import TRANSFORMS


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
