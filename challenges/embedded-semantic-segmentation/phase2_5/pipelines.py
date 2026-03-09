"""
Custom mmseg pipeline transforms and visualizer for Phase 2.5 configs
(mmseg 1.x API).

Changes vs Phase 2 pipelines.py:
  Fix 2: ReflectPad now writes img_padding_size to results, so the mmseg
          runner can de-pad predictions before metric computation — fixing
          the training val mIoU underestimation from Phase 2.
  Fix 3: ReflectPad gains a center= parameter (default False for train,
          True for val/test via config). When center=True the padded area is
          split evenly on both sides, matching the challenge server's centred
          mirror-pad preprocessing. The img_padding_size is updated accordingly.
  Fix 4: MacViSegVisualizer subclasses SegLocalVisualizer and overlays ignore
          areas (GT==255) on the predicted segmentation image in WandB, making
          it easier to compare GT and prediction overlays visually.

ReflectPad usage in config:
    # Train pipeline (no centering, padding_size written for de-pad):
    dict(type='ReflectPad', size=crop_size, seg_pad_val=255)

    # Test/val pipeline (centred padding):
    dict(type='ReflectPad', size=crop_size, center=True)

MacViSegVisualizer usage in runtime config:
    visualizer = dict(
        type='MacViSegVisualizer',
        vis_backends=vis_backends,
        alpha=0.8)
"""

import cv2
import numpy as np
import torch
import mmcv
from mmseg.registry import MODELS, TRANSFORMS, VISUALIZERS
from mmseg.visualization import SegLocalVisualizer


@TRANSFORMS.register_module()
class ReflectPad:
    """Pad image and mask with mirror (reflect) padding.

    Phase 2.5 version adds:
    - img_padding_size written to results dict (Fix 2), enabling accurate
      de-padding during metric computation.
    - center parameter (Fix 3): when True, the pad area is split evenly on
      both sides (left/right, top/bottom), matching the challenge server's
      centred mirror-pad. When False (default, used in training) padding is
      applied to the bottom and right edges only.

    Args:
        size (tuple): Target (H, W) after padding.
        seg_pad_val (int): Value used to pad segmentation masks (255 = ignore).
        center (bool): If True, split padding evenly on both sides. Use True
            in the test/val pipeline to match the challenge server preprocessing.
            Default: False (bottom-right padding, used during training).
    """

    def __init__(self, size, seg_pad_val=255, center=False):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = tuple(size)   # (H, W)
        self.seg_pad_val = seg_pad_val
        self.center = center

    def _compute_padding(self, h, w):
        """Return (pad_top, pad_bottom, pad_left, pad_right) integers."""
        target_h, target_w = self.size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if self.center:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_top = 0
            pad_bottom = pad_h
            pad_left = 0
            pad_right = pad_w
        return pad_top, pad_bottom, pad_left, pad_right

    def _border_type(self, h, w):
        """Choose reflect vs constant border based on image size vs pad size."""
        target_h, target_w = self.size
        # reflect requires padding strictly less than image dimension
        if (target_h - h) < h and (target_w - w) < w:
            return cv2.BORDER_REFLECT_101
        return cv2.BORDER_CONSTANT

    def _pad_img(self, results):
        img = results['img']
        h, w = img.shape[:2]
        target_h, target_w = self.size

        pad_top, pad_bottom, pad_left, pad_right = self._compute_padding(h, w)

        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            # Already fits — no padding needed.
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['pad_shape'] = img.shape
            results['img_padding_size'] = [0, 0, 0, 0]  # [left, right, top, bottom]
            return

        border = self._border_type(h, w)
        padded = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            border,
            value=0)
        results['img'] = padded
        results['img_shape'] = padded.shape[:2]
        results['pad_shape'] = padded.shape
        # img_padding_size format: [left, right, top, bottom]
        results['img_padding_size'] = [pad_left, pad_right, pad_top, pad_bottom]

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            seg = results[key]
            h, w = seg.shape[:2]
            target_h, target_w = self.size

            pad_top, pad_bottom, pad_left, pad_right = self._compute_padding(h, w)
            if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
                continue

            border = self._border_type(h, w)
            # mmcv.impad handles 2-D (H, W) and 3-D (H, W, C) masks natively.
            # Do NOT add a channel dim: impad squeezes C=1 back out, breaking
            # the shape (observed: (H,W,1) → impad → (H,W) → [..0] → (H,)).
            # Use cv2.copyMakeBorder directly to control top/bottom/left/right.
            padded = cv2.copyMakeBorder(
                seg,
                pad_top, pad_bottom, pad_left, pad_right,
                border,
                value=self.seg_pad_val)
            results[key] = padded

    def __call__(self, results):
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'size={self.size}, seg_pad_val={self.seg_pad_val}, '
                f'center={self.center})')


@VISUALIZERS.register_module()
class MacViSegVisualizer(SegLocalVisualizer):
    """SegLocalVisualizer that overlays ignore areas (GT==255) on predicted images.

    Fix 4: In Phase 2, the GT overlay already shows ignore areas distinctively
    (class 255 is handled by the base class), but the predicted overlay does not
    — making it hard to compare GT and prediction side-by-side in WandB.

    This subclass temporarily adds a 4th display class (dark-grey, index 3) for
    pixels where GT==255, marks them in the prediction tensor, calls the parent
    visualizer, then restores everything.  The result is that predicted images in
    WandB show grey pixels wherever the GT is ignored, matching the GT overlay
    appearance.

    Args:
        ignore_color (list): RGB colour used for ignore areas in prediction overlay.
            Default: [100, 100, 100] (dark grey).
    """

    IGNORE_DISPLAY_IDX = 3  # temporary extra class index used only for rendering
    DEFAULT_IGNORE_COLOR = [100, 100, 100]

    def __init__(self, name='visualizer', ignore_color=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_color = ignore_color or self.DEFAULT_IGNORE_COLOR

    def add_datasample(self, name, image, data_sample=None,
                       draw_gt=True, draw_pred=True, show=False,
                       wait_time=0, out_file=None, step=0):
        """Visualize with ignore-area overlay on the predicted segmentation."""
        if (data_sample is not None and draw_pred
                and hasattr(data_sample, 'gt_sem_seg')
                and hasattr(data_sample, 'pred_sem_seg')):

            gt = data_sample.gt_sem_seg.data          # (1, H, W) or (H, W)
            ignore_mask = (gt == 255).squeeze()        # (H, W) bool

            if ignore_mask.any():
                # Save originals
                orig_pred_data = data_sample.pred_sem_seg.data.clone()
                orig_palette = list(self.dataset_meta.get('palette', []))
                orig_classes = list(self.dataset_meta.get('classes', []))

                # Temporarily extend palette with ignore colour
                self.dataset_meta['palette'] = orig_palette + [self.ignore_color]
                self.dataset_meta['classes'] = orig_classes + ['ignore']

                # Mark ignore pixels in prediction
                new_pred = orig_pred_data.squeeze().clone()
                new_pred[ignore_mask] = self.IGNORE_DISPLAY_IDX
                data_sample.pred_sem_seg.data = new_pred.unsqueeze(0)

                try:
                    super().add_datasample(
                        name, image, data_sample,
                        draw_gt=draw_gt, draw_pred=draw_pred,
                        show=show, wait_time=wait_time,
                        out_file=out_file, step=step)
                finally:
                    # Restore originals unconditionally
                    self.dataset_meta['palette'] = orig_palette
                    self.dataset_meta['classes'] = orig_classes
                    data_sample.pred_sem_seg.data = orig_pred_data
                return

        super().add_datasample(
            name, image, data_sample,
            draw_gt=draw_gt, draw_pred=draw_pred,
            show=show, wait_time=wait_time,
            out_file=out_file, step=step)
