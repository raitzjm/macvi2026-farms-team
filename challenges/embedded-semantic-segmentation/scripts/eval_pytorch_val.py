"""
Evaluate PyTorch model weights directly on the LaRS validation set.

Same metric as eval_onnx_val.py (mIoU, obstacle F1, Q = mIoU × F1) but using
PyTorch inference instead of ONNX Runtime. Run this to verify ONNX conversion
has no measurable effect on evaluation results.

Environment notes:
  - STDC2 and SegNeXt-T: run in `macvi` env (mmseg 0.30.0 / mmcv 1.7.2)
      /home/farms/miniforge3/envs/macvi/bin/python scripts/eval_pytorch_val.py \\
          --models stdc2 segnext_t
  - PIDNet-S: run in `pidnet` env (mmseg 1.2.2 / mmcv 2.1.0)
      /home/farms/miniforge3/envs/pidnet/bin/python scripts/eval_pytorch_val.py \\
          --models pidnets

Usage (from challenge dir):
    python scripts/eval_pytorch_val.py [--models stdc2 pidnets segnext_t]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = '/home/farms/lars_linked/val'
IMG_DIR      = os.path.join(DATA_ROOT, 'images')
SEG_MASK_DIR = os.path.join(DATA_ROOT, 'semantic_masks')
PAN_MASK_DIR = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_masks'
PAN_JSON     = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_annotations.json'

MODELS = {
    'stdc2': dict(
        config=os.path.join(CHALLENGE_DIR, 'configs/stdc2_768x384_lars_fun.py'),
        ckpt  =os.path.join(CHALLENGE_DIR, 'outputs/fun_stdc2_768x384/best_mIoU_iter_10000.pth'),
        api   ='v030',
        align_corners=False,
    ),
    'pidnets': dict(
        config=os.path.join(CHALLENGE_DIR, 'configs/pidnets_768x384_lars_fun.py'),
        ckpt  =os.path.join(CHALLENGE_DIR, 'outputs/fun_pidnets_768x384/best_mIoU_iter_10000.pth'),
        api   ='v122',
        align_corners=True,
    ),
    'segnext_t': dict(
        config=os.path.join(CHALLENGE_DIR, 'configs/segnext_t_768x384_lars_fun.py'),
        ckpt  =os.path.join(CHALLENGE_DIR, 'outputs/fun_segnext_t_768x384/best_mIoU_iter_14000.pth'),
        api   ='v030',
        align_corners=False,
    ),
}

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_H, MODEL_W = 384, 768
NUM_CLASSES = 3
IGNORE_LABEL = 255
COVERAGE_THRESH = 0.70
MIN_PRED_AREA = 100
OBSTACLE_CATS = set([1] + list(range(11, 20)))

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

MEAN_T = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD_T  = torch.tensor([58.395,  57.12,  57.375]).view(1, 3, 1, 1)


# ── Model wrapper ──────────────────────────────────────────────────────────────

class SegWrapper(nn.Module):
    """Backbone → decode_head → interpolate, bypassing img_metas."""
    def __init__(self, model, out_h, out_w, align_corners=False):
        super().__init__()
        self.model = model
        self.out_h = out_h
        self.out_w = out_w
        self.align_corners = align_corners

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        logits = F.interpolate(
            logits, size=(self.out_h, self.out_w),
            mode='bilinear', align_corners=self.align_corners)
        return logits


def load_model(cfg):
    api = cfg['api']
    if api == 'v030':
        try:
            from mmseg.apis import init_segmentor
        except ImportError:
            raise RuntimeError(
                'mmseg 0.30.0 not found. Run in the `macvi` env:\n'
                '  /home/farms/miniforge3/envs/macvi/bin/python ...')
        model = init_segmentor(cfg['config'], cfg['ckpt'], device='cpu')
    elif api == 'v122':
        try:
            from mmseg.apis import init_model
        except ImportError:
            raise RuntimeError(
                'mmseg 1.2.2 not found. Run in the `pidnet` env:\n'
                '  /home/farms/miniforge3/envs/pidnet/bin/python ...')
        model = init_model(cfg['config'], cfg['ckpt'], device='cpu')
    else:
        raise ValueError(f'Unknown api: {api}')

    model.eval()

    # SegNeXt-T: fix rand_init so NMF is deterministic (same fix as ONNX export)
    try:
        ham = model.decode_head.hamburger.ham
        ham.rand_init = False
        with torch.no_grad():
            _ = model.decode_head(model.backbone(torch.zeros(1, 3, MODEL_H, MODEL_W)))
    except AttributeError:
        pass  # not a Hamburger model

    return SegWrapper(model, MODEL_H, MODEL_W, cfg['align_corners']).eval()


# ── Preprocessing / inference ──────────────────────────────────────────────────

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    x = (x - MEAN_T) / STD_T
    return x


def run_pytorch(wrapper, img_path, ori_h, ori_w):
    """Run model; bilinear-upsample logits to ori_shape then argmax."""
    x = preprocess(img_path)
    with torch.no_grad():
        logits = wrapper(x)                                    # (1, 3, H, W)
    logits_up = F.interpolate(logits, size=(ori_h, ori_w),
                              mode='bilinear', align_corners=False)
    pred = logits_up[0].argmax(0).numpy().astype(np.uint8)    # (ori_h, ori_w)
    return pred


# ── Metrics (identical to eval_onnx_val.py) ────────────────────────────────────

def decode_panoptic(pan_path):
    pan = np.array(Image.open(pan_path).convert('RGB'), dtype=np.int32)
    return pan[:, :, 0] + pan[:, :, 1] * 256 + pan[:, :, 2] * 65536


def compute_iou(pred, gt_mask):
    ious = []
    valid = gt_mask != IGNORE_LABEL
    for cls in range(NUM_CLASSES):
        p = (pred == cls) & valid
        g = (gt_mask == cls) & valid
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious


def obstacle_f1(pred_obstacle, pan_id_map, segments_info):
    gt_obs_union = np.zeros(pred_obstacle.shape, dtype=bool)
    gt_instances = []
    for seg in segments_info:
        if seg['category_id'] not in OBSTACLE_CATS:
            continue
        mask = (pan_id_map == seg['id'])
        if mask.sum() == 0:
            continue
        gt_obs_union |= mask
        gt_instances.append(mask)

    tp = fn = 0
    for gt_mask in gt_instances:
        gt_area = gt_mask.sum()
        covered = (pred_obstacle & gt_mask).sum()
        if gt_area > 0 and covered / gt_area >= COVERAGE_THRESH:
            tp += 1
        else:
            fn += 1

    labeled, num_blobs = ndimage.label(pred_obstacle)
    fp = 0
    for blob_id in range(1, num_blobs + 1):
        blob_mask = labeled == blob_id
        if blob_mask.sum() < MIN_PRED_AREA:
            continue
        if (blob_mask & gt_obs_union).sum() == 0:
            fp += 1

    return tp, fp, fn


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(model_name, cfg, pan_data):
    print(f'\n{"="*60}')
    print(f'  Model:      {model_name}')
    print(f'  Checkpoint: {cfg["ckpt"]}')
    print(f'{"="*60}')

    wrapper = load_model(cfg)

    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)
    total_tp = total_fp = total_fn = 0
    n_images = 0

    for img_name, pan_ann in pan_data.items():
        stem     = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        seg_path = os.path.join(SEG_MASK_DIR, stem + '.png')
        pan_path = os.path.join(PAN_MASK_DIR, pan_ann['file_name'])

        if not os.path.exists(img_path):
            continue

        gt_seg  = np.array(Image.open(seg_path), dtype=np.uint8)
        H, W    = gt_seg.shape
        pan_ids = decode_panoptic(pan_path)

        pred = run_pytorch(wrapper, img_path, H, W)

        ious = compute_iou(pred, gt_seg)
        for c, iou in enumerate(ious):
            if not np.isnan(iou):
                iou_sum[c] += iou
                iou_cnt[c] += 1

        pred_obstacle = (pred == 0)
        tp, fp, fn = obstacle_f1(pred_obstacle, pan_ids, pan_ann['segments_info'])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_images += 1

    class_ious = iou_sum / np.maximum(iou_cnt, 1)
    miou       = float(np.nanmean(class_ious))
    precision  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall     = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1         = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    q          = miou * f1

    print(f'  Images evaluated : {n_images}')
    print(f'  IoU obstacle     : {class_ious[0]*100:.2f}%')
    print(f'  IoU water        : {class_ious[1]*100:.2f}%')
    print(f'  IoU sky          : {class_ious[2]*100:.2f}%')
    print(f'  mIoU             : {miou*100:.2f}%')
    print(f'  TP / FP / FN     : {total_tp} / {total_fp} / {total_fn}')
    print(f'  Precision        : {precision*100:.2f}%')
    print(f'  Recall           : {recall*100:.2f}%')
    print(f'  F1               : {f1*100:.2f}%')
    print(f'  Q = mIoU × F1    : {q*100:.2f}%')

    return dict(model=model_name, miou=miou, iou_obs=class_ious[0],
                iou_wat=class_ious[1], iou_sky=class_ious[2],
                f1=f1, precision=precision, recall=recall, q=q,
                tp=total_tp, fp=total_fp, fn=total_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    args = parser.parse_args()

    with open(PAN_JSON) as f:
        pan_json = json.load(f)
    id_to_fname = {img['id']: img['file_name'] for img in pan_json['images']}
    pan_data = {}
    for ann in pan_json['annotations']:
        img_fname = id_to_fname[ann['image_id']]
        pan_data[img_fname] = {
            'file_name':     ann['file_name'],
            'segments_info': ann['segments_info'],
        }

    results = []
    skipped = []
    for name in args.models:
        cfg = MODELS[name]
        if not os.path.exists(cfg['ckpt']):
            print(f'SKIP {name}: checkpoint not found at {cfg["ckpt"]}')
            skipped.append(name)
            continue
        try:
            r = evaluate(name, cfg, pan_data)
            results.append(r)
        except RuntimeError as e:
            print(f'SKIP {name}: {e}')
            skipped.append(name)

    if skipped:
        print(f'\nSkipped (wrong env): {skipped}')
        if 'pidnets' in skipped:
            print('  → PIDNet-S: run with /home/farms/miniforge3/envs/pidnet/bin/python')
        if any(m in skipped for m in ['stdc2', 'segnext_t']):
            print('  → STDC2/SegNeXt-T: run with /home/farms/miniforge3/envs/macvi/bin/python')

    if len(results) > 1:
        print(f'\n{"="*70}')
        print('  SUMMARY (PyTorch weights)')
        print(f'{"="*70}')
        hdr = f"{'Model':<12} {'mIoU':>7} {'IoU obs':>8} {'IoU wat':>8} {'IoU sky':>8} {'F1':>7} {'Q':>7}"
        print(hdr)
        print('-' * 70)
        for r in results:
            print(f"{r['model']:<12} {r['miou']*100:>6.2f}% {r['iou_obs']*100:>7.2f}% "
                  f"{r['iou_wat']*100:>7.2f}% {r['iou_sky']*100:>7.2f}% "
                  f"{r['f1']*100:>6.2f}% {r['q']*100:>6.2f}%")
        print(f'{"="*70}')
        print('  Compare with ONNX results (eval_onnx_val.py) to check conversion effect.')

    print(f'\n  ONNX baseline for reference:')
    print(f'  {"Model":<12} {"mIoU":>7} {"F1":>7} {"Q":>7}')
    print(f'  {"-"*38}')
    onnx_ref = [
        ('stdc2',     90.73, 66.23, 60.09),
        ('pidnets',   92.03, 72.10, 66.35),
        ('segnext_t', 92.29, 70.56, 65.12),
    ]
    for name, miou, f1, q in onnx_ref:
        if name in [r['model'] for r in results]:
            pt_r = next(r for r in results if r['model'] == name)
            dm = pt_r['miou']*100 - miou
            df = pt_r['f1']*100 - f1
            dq = pt_r['q']*100 - q
            print(f'  {name:<12} {miou:>6.2f}%  {f1:>6.2f}%  {q:>6.2f}%  '
                  f'(PyTorch Δ: mIoU{dm:+.2f}% F1{df:+.2f}% Q{dq:+.2f}%)')


if __name__ == '__main__':
    main()
