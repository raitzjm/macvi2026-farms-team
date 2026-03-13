"""
Custom mmseg pipeline transforms and visualizer for Phase 3 configs
(mmseg 1.x API).

Inherits all Phase 2.5 transforms (ReflectPad, MacViSegVisualizer) and adds:

  CopyPasteObstacles:
      Pastes random obstacle crops from the external dataset DB onto LaRS
      training images, with boundary blurring to prevent spurious edge
      targets in GenerateEdgeMasked / DetailAggregateLoss.

  GenerateEdgeMasked:
      Drop-in replacement for GenerateEdge that zeros out edge targets in
      reflect-padded regions. This is Option 1 from issue #28 — the fix
      that makes keep_ratio=True safe for PIDNet-S.

  EnhancedPhotoMetricDistortion:
      Extends PhotoMetricDistortion with wider (but bounded) contrast/
      saturation ranges and an optional RandomGamma step.

DB path: challenges/common/external-dataset-analysis/shared_data/copypaste_candidates.db
DB schema (table: candidates):
    dataset TEXT, filename TEXT (abs mask path), x1 INT, y1 INT, x2 INT, y2 INT,
    area_native REAL, area_normalized REAL, area_pct REAL, size_tier TEXT
Image paths are resolved from images.txt in the harmonized dataset dir
(parent of masks/).
"""

import os
import random
import time
import multiprocessing
# sqlite3 is imported lazily inside CopyPasteObstacles._load_candidates()
# to avoid a libstdc++ version conflict when torchrun spawns worker processes
# before the conda LD_LIBRARY_PATH is fully resolved.

import cv2
import numpy as np
import torch
import mmcv
from mmengine.dist import master_only
from mmengine.logging import MMLogger
from mmseg.datasets.transforms import GenerateEdge, PhotoMetricDistortion
from mmseg.registry import MODELS, TRANSFORMS, VISUALIZERS
from mmseg.visualization import SegLocalVisualizer

# ── Global shared manager for CopyPaste cache ────────────────────────────────
# Initialized on first import in the main process.
_manager = None
_shared_cache = None

def _get_shared_cache():
    global _manager, _shared_cache
    if _shared_cache is None:
        _manager = multiprocessing.Manager()
        _shared_cache = _manager.dict()
    return _shared_cache


# ── ReflectPad (Phase 2.5, unchanged) ─────────────────────────────────────────

@TRANSFORMS.register_module()
class ReflectPad:
    """Pad image and mask with mirror (reflect) padding.

    Phase 2.5 version:
    - img_padding_size written to results dict (Fix 2).
    - center parameter (Fix 3): when True, split padding evenly on both sides.

    Args:
        size (tuple): Target (H, W) after padding.
        seg_pad_val (int): Value used to pad segmentation masks (255 = ignore).
        center (bool): If True, split padding evenly on both sides. Default: False.
    """

    def __init__(self, size, seg_pad_val=255, center=False):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = tuple(size)   # (H, W)
        self.seg_pad_val = seg_pad_val
        self.center = center

    def _compute_padding(self, h, w):
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
        target_h, target_w = self.size
        if (target_h - h) < h and (target_w - w) < w:
            return cv2.BORDER_REFLECT_101
        return cv2.BORDER_CONSTANT

    def _pad_img(self, results):
        img = results['img']
        h, w = img.shape[:2]
        pad_top, pad_bottom, pad_left, pad_right = self._compute_padding(h, w)

        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['pad_shape'] = img.shape
            results['img_padding_size'] = [0, 0, 0, 0]
            return

        border = self._border_type(h, w)
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, border, value=0)
        results['img'] = padded
        results['img_shape'] = padded.shape[:2]
        results['pad_shape'] = padded.shape
        # [left, right, top, bottom]
        results['img_padding_size'] = [pad_left, pad_right, pad_top, pad_bottom]

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            seg = results[key]
            h, w = seg.shape[:2]
            pad_top, pad_bottom, pad_left, pad_right = self._compute_padding(h, w)
            if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
                continue
            border = self._border_type(h, w)
            padded = cv2.copyMakeBorder(
                seg, pad_top, pad_bottom, pad_left, pad_right,
                border, value=self.seg_pad_val)
            results[key] = padded

    def __call__(self, results):
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'size={self.size}, seg_pad_val={self.seg_pad_val}, '
                f'center={self.center})')


# ── CopyPasteObstacles ─────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class CopyPasteObstacles:
    """Pastes random obstacle crops from external dataset DB onto LaRS training images.

    DB path: challenges/common/external-dataset-analysis/shared_data/copypaste_candidates.db
    DB schema (table: candidates):
        dataset TEXT, filename TEXT (abs mask path),
        x1 INT, y1 INT, x2 INT, y2 INT,
        area_native REAL, area_normalized REAL, area_pct REAL, size_tier TEXT

    Image paths are resolved from images.txt in the harmonized dataset dir
    (parent dir of masks/). The stem of the mask filename is matched 1:1 with
    the corresponding line in images.txt (same ordering by stem).

    CRITICAL — boundary_blur_radius:
        Blurs the paste mask boundary before updating the GT segmentation map.
        Without this, GenerateEdgeMasked / DetailAggregateLoss picks up
        artificial paste seams as edge targets → model learns paste artifacts
        → FP explosion on the server.

    Tier sampling weights (oversample tiny/small to address 47% tiny-FN weakness):
        tiny=4, small=2, medium=1, large=0.5

    Args:
        db_path (str): Absolute or repo-relative path to the SQLite DB.
        n_paste (tuple): (min, max) number of obstacles to paste per image.
        prob_apply (float): Probability of applying the transform at all.
        scale_range (tuple): (min, max) scale factor for resizing crops.
        prefer_waterline_prob (float): Probability of placing near the waterline.
        boundary_blur_radius (int): Gaussian blur radius for paste mask boundary.
        match_brightness (bool): Normalise crop L* to scene water L* (handles
            WaterScenes L*≈151 vs LaRS ≈113 mismatch).
    """

    TIER_WEIGHTS = {'tiny': 4.0, 'small': 2.0, 'medium': 1.0, 'large': 0.5}

    def __init__(self, db_path, n_paste=(1, 3), prob_apply=0.7,
                 scale_range=(0.5, 1.5), prefer_waterline_prob=0.3,
                 boundary_blur_radius=2, match_brightness=True,
                 cache_candidates=True):
        self.db_path = db_path
        self.n_paste = n_paste
        self.prob_apply = prob_apply
        self.scale_range = scale_range
        self.prefer_waterline_prob = prefer_waterline_prob
        self.boundary_blur_radius = boundary_blur_radius
        self.match_brightness = match_brightness
        self.cache_candidates = cache_candidates
        self._candidates = None   # list of (img_path, mask_path, x1, y1, x2, y2, tier)
        self._weights = None      # np.ndarray, normalised sampling weights

    def _build_image_lookup(self, harmonized_dir):
        """Build stem → image_path mapping from images.txt in harmonized_dir."""
        images_txt = os.path.join(harmonized_dir, 'images.txt')
        masks_dir = os.path.join(harmonized_dir, 'masks')
        if not os.path.isfile(images_txt):
            return {}
        # images.txt lines are in the same order as sorted mask filenames
        with open(images_txt) as f:
            img_paths = [l.strip() for l in f if l.strip()]
        mask_names = sorted(os.listdir(masks_dir))
        lookup = {}
        for img_path, mask_name in zip(img_paths, mask_names):
            stem = os.path.splitext(mask_name)[0]
            lookup[stem] = img_path
        return lookup

    def _load_candidates(self):
        """Load all candidates from DB into memory on first call."""
        import sqlite3  # lazy import — avoids libstdc++ conflict at module load time
        logger = MMLogger.get_current_instance()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            'SELECT dataset, filename, x1, y1, x2, y2, area_native, size_tier '
            'FROM candidates'
        ).fetchall()
        conn.close()

        # Build per-dataset image lookup (stem → image_path)
        # harmonized_dir = parent of masks/ directory (= parent of dirname(filename))
        dataset_lookup = {}
        for dataset, mask_path, *_ in rows:
            if dataset in dataset_lookup:
                continue
            harmonized_dir = os.path.dirname(os.path.dirname(mask_path))
            dataset_lookup[dataset] = self._build_image_lookup(harmonized_dir)

        # Build candidate list
        candidates = []
        weights = []
        for dataset, mask_path, x1, y1, x2, y2, area_native, size_tier in rows:
            stem = os.path.splitext(os.path.basename(mask_path))[0]
            lookup = dataset_lookup.get(dataset, {})
            img_path = lookup.get(stem)
            if img_path is None:
                continue   # skip if image not found
            candidates.append((img_path, mask_path, x1, y1, x2, y2, size_tier))
            weights.append(self.TIER_WEIGHTS.get(size_tier, 1.0))

        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        self._candidates = candidates
        self._weights = weights_arr

        # Pre-load candidates into shared RAM cache if requested
        if self.cache_candidates:
            cache = _get_shared_cache()
            # We use a lock-like check: only first worker to see empty cache populates it
            if len(cache) == 0:
                logger.info(f'Populating CopyPaste shared RAM cache ({len(candidates)} candidates)...')
                start_time = time.time()
                # To avoid OOM during loading, we populate carefully
                for i, (img_path, mask_path, x1, y1, x2, y2, _) in enumerate(candidates):
                    if i % 5000 == 0 and i > 0:
                        logger.info(f'Cached {i}/{len(candidates)} candidates...')
                    
                    # Store as numpy arrays to avoid re-decoding
                    src_img = cv2.imread(img_path)
                    src_mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if src_img is not None and src_mask_img is None:
                        # Corner case: image exists but mask doesn't?
                        continue
                    
                    # Store only the crop to save RAM
                    crop_img = src_img[y1:y2, x1:x2].copy()
                    crop_mask = (src_mask_img[y1:y2, x1:x2] == 0).astype(np.uint8).copy()
                    cache[i] = (crop_img, crop_mask)
                
                logger.info(f'CopyPaste cache populated in {time.time() - start_time:.1f}s.')

    def _estimate_waterline(self, seg_mask):
        """Return approximate waterline row (top of water region)."""
        water_rows = np.where((seg_mask == 1).mean(axis=1) > 0.1)[0]
        return int(water_rows.min()) if len(water_rows) > 0 else seg_mask.shape[0] // 2

    def __call__(self, results):
        if random.random() > self.prob_apply:
            return results
        if self._candidates is None:
            self._load_candidates()
        if not self._candidates:
            return results

        img = results['img']           # HWC uint8 BGR
        seg = results['gt_seg_map']    # HW uint8, mmseg 1.x key
        H, W = img.shape[:2]
        water_mask = (seg == 1)
        if not water_mask.any():
            return results

        n = random.randint(*self.n_paste)
        waterline_row = self._estimate_waterline(seg)
        idxs = np.random.choice(len(self._candidates), size=n, replace=True, p=self._weights)

        cache = _get_shared_cache() if self.cache_candidates else {}

        for idx in idxs:
            if idx in cache:
                crop_img, crop_mask = cache[idx]
                # Deep copy to avoid modifying shared memory if we do in-place ops later
                crop_img = crop_img.copy()
                crop_mask = crop_mask.copy()
            else:
                img_path, mask_path, x1, y1, x2, y2, size_tier = self._candidates[idx]
                src_img = cv2.imread(img_path)
                src_mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if src_img is None or src_mask_img is None:
                    continue
                crop_img = src_img[y1:y2, x1:x2]
                crop_mask = (src_mask_img[y1:y2, x1:x2] == 0).astype(np.uint8)

            if crop_img.size == 0 or crop_mask.size == 0:
                continue

            # Resize crop
            scale = random.uniform(*self.scale_range)
            ch = max(1, int(crop_img.shape[0] * scale))
            cw = max(1, int(crop_img.shape[1] * scale))
            crop_img = cv2.resize(crop_img, (cw, ch))
            crop_mask = cv2.resize(crop_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

            if ch >= H or cw >= W:
                continue

            # Choose paste location over water
            use_waterline = random.random() < self.prefer_waterline_prob
            water_candidates = np.argwhere(water_mask)
            if len(water_candidates) == 0:
                continue

            top, left = None, None
            for _ in range(20):
                if use_waterline:
                    wy = max(0, min(waterline_row + random.randint(-20, 20), H - ch - 1))
                    wx = random.randint(0, max(0, W - cw - 1))
                else:
                    wc = water_candidates[random.randint(0, len(water_candidates) - 1)]
                    wy, wx = int(wc[0]), int(wc[1])
                    wy = max(0, min(wy, H - ch - 1))
                    wx = max(0, min(wx, W - cw - 1))
                if wy + ch <= H and wx + cw <= W:
                    top, left = wy, wx
                    break
            if top is None:
                continue

            # Brightness normalisation: match crop L* to destination water region
            if self.match_brightness:
                dest_water_pixels = img[water_mask]
                if len(dest_water_pixels) > 0:
                    dest_lab = cv2.cvtColor(
                        dest_water_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
                    dest_l = dest_lab[..., 0].mean()
                    crop_lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB).astype(float)
                    crop_l_mean = crop_lab[..., 0].mean()
                    if crop_l_mean > 1.0:
                        crop_lab[..., 0] = np.clip(
                            crop_lab[..., 0] * (dest_l / crop_l_mean), 0, 255)
                    crop_img = cv2.cvtColor(crop_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # Boundary blending: blur mask to prevent sharp seam edge targets
            blur_k = 2 * self.boundary_blur_radius + 1
            blurred_mask = cv2.GaussianBlur(crop_mask.astype(float), (blur_k, blur_k), 0)
            hard_mask = blurred_mask > 0.5     # pixels to update GT label
            alpha = blurred_mask[..., None]    # soft alpha for image blending

            # Blend image
            roi = img[top:top+ch, left:left+cw]
            img[top:top+ch, left:left+cw] = (
                alpha * crop_img + (1 - alpha) * roi).astype(np.uint8)
            # Update GT (only hard interior, not blurred boundary)
            seg[top:top+ch, left:left+cw][hard_mask] = 0  # obstacle class

        results['img'] = img
        results['gt_seg_map'] = seg
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'db_path={self.db_path}, n_paste={self.n_paste}, '
                f'prob_apply={self.prob_apply}, scale_range={self.scale_range}, '
                f'boundary_blur_radius={self.boundary_blur_radius})')


# ── GenerateEdgeMasked ─────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class GenerateEdgeMasked(GenerateEdge):
    """GenerateEdge extended to zero out edge targets in reflect-padded regions.

    With keep_ratio=True + ReflectPad in train, images have padded borders whose
    reflected mask content creates spurious edges. GenerateEdge would compute
    loss on these, causing the model to hallucinate obstacle boundaries at image
    borders (root cause of EXP-018 FP explosion, −10.6 Q on server).

    Reads img_padding_size = [left, right, top, bottom] written by ReflectPad.
    Sets edge map pixels in padded border to 0 (no edge target).
    If img_padding_size is absent (no padding occurred), behaves identically to
    the parent GenerateEdge.

    This is Option 1 from issue #28 — the fix that makes keep_ratio=True safe
    for PIDNet-S.
    """

    def transform(self, results):
        results = super().transform(results)
        padding = results.get('img_padding_size', None)
        if padding is not None:
            left, right, top, bottom = padding
            edge = results['gt_edge_map']   # HW uint8
            H, W = edge.shape
            if top > 0:
                edge[:top, :] = 0
            if bottom > 0:
                edge[H - bottom:, :] = 0
            if left > 0:
                edge[:, :left] = 0
            if right > 0:
                edge[:, W - right:] = 0
            results['gt_edge_map'] = edge
        return results


# ── EnhancedPhotoMetricDistortion ─────────────────────────────────────────────

@TRANSFORMS.register_module()
class EnhancedPhotoMetricDistortion(PhotoMetricDistortion):
    """PhotoMetricDistortion with wider (but bounded) ranges + optional RandomGamma.

    Overrides defaults to use bounded wider ranges recommended for Phase 3:
        brightness_delta=77   (vs default 32)
        contrast_range=(0.6, 1.5)   (vs default (0.5, 1.5))
        saturation_range=(0.6, 1.5) (vs default (0.5, 1.5))
        hue_delta=27          (vs default 18)

    Adds RandomGamma step applied after the base PhotoMetricDistortion:
        gamma_range=(0.6, 1.6), gamma_prob=0.3

    Do not exceed these ranges — extreme distortions wash out tiny obstacle
    textures and worsen ORCA open-water hallucinations.

    Args:
        gamma_range (tuple): (min, max) gamma value for power-law correction.
        gamma_prob (float): Probability of applying gamma correction.
        All other args: passed to PhotoMetricDistortion.
    """

    def __init__(self,
                 brightness_delta=77,
                 contrast_range=(0.6, 1.5),
                 saturation_range=(0.6, 1.5),
                 hue_delta=27,
                 gamma_range=(0.6, 1.6),
                 gamma_prob=0.3):
        super().__init__(
            brightness_delta=brightness_delta,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue_delta=hue_delta)
        self.gamma_range = gamma_range
        self.gamma_prob = gamma_prob

    def _apply_gamma(self, img):
        """Apply random gamma correction with probability gamma_prob."""
        if random.random() > self.gamma_prob:
            return img
        gamma = random.uniform(*self.gamma_range)
        # Build lookup table for uint8 gamma correction (fast)
        inv_gamma = 1.0 / gamma
        lut = np.array([
            np.clip((i / 255.0) ** inv_gamma * 255, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(img, lut)

    def transform(self, results):
        results = super().transform(results)
        results['img'] = self._apply_gamma(results['img'])
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'brightness_delta={self.brightness_delta}, '
                f'contrast_range=({self.contrast_lower}, {self.contrast_upper}), '
                f'saturation_range=({self.saturation_lower}, {self.saturation_upper}), '
                f'hue_delta={self.hue_delta}, '
                f'gamma_range={self.gamma_range}, gamma_prob={self.gamma_prob})')


# ── MacViSegVisualizer (Phase 2.5, unchanged) ──────────────────────────────────

@VISUALIZERS.register_module()
class MacViSegVisualizer(SegLocalVisualizer):
    """SegLocalVisualizer that overlays ignore areas (GT==255) on predicted images.

    Fix 4 (Phase 2.5): In the predicted overlay, pixels where GT==255 are shown
    in dark grey, matching the GT overlay appearance in WandB.

    Args:
        ignore_color (list): RGB colour used for ignore areas. Default: [100, 100, 100].
    """

    IGNORE_DISPLAY_IDX = 3
    DEFAULT_IGNORE_COLOR = [100, 100, 100]

    def __init__(self, name='visualizer', ignore_color=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_color = ignore_color or self.DEFAULT_IGNORE_COLOR

    @master_only
    def add_datasample(self, name, image, data_sample=None,
                       draw_gt=True, draw_pred=True, show=False,
                       wait_time=0, out_file=None, step=0,
                       with_labels=True):
        if (data_sample is not None and draw_pred
                and hasattr(data_sample, 'gt_sem_seg')
                and hasattr(data_sample, 'pred_sem_seg')):

            gt = data_sample.gt_sem_seg.data
            ignore_mask = (gt == 255).squeeze()

            if ignore_mask.any():
                orig_pred_data = data_sample.pred_sem_seg.data.clone()
                orig_palette = list(self.dataset_meta.get('palette', []))
                orig_classes = list(self.dataset_meta.get('classes', []))

                self.dataset_meta['palette'] = orig_palette + [self.ignore_color]
                self.dataset_meta['classes'] = orig_classes + ['ignore']

                new_pred = orig_pred_data.squeeze().clone()
                new_pred[ignore_mask] = self.IGNORE_DISPLAY_IDX
                data_sample.pred_sem_seg.data = new_pred.unsqueeze(0)

                try:
                    SegLocalVisualizer.add_datasample.__wrapped__(
                        self, name, image, data_sample,
                        draw_gt=draw_gt, draw_pred=draw_pred,
                        show=show, wait_time=wait_time,
                        out_file=out_file, step=step,
                        with_labels=with_labels)
                finally:
                    self.dataset_meta['palette'] = orig_palette
                    self.dataset_meta['classes'] = orig_classes
                    data_sample.pred_sem_seg.data = orig_pred_data
                return

        SegLocalVisualizer.add_datasample.__wrapped__(
            self, name, image, data_sample,
            draw_gt=draw_gt, draw_pred=draw_pred,
            show=show, wait_time=wait_time,
            out_file=out_file, step=step,
            with_labels=with_labels)
