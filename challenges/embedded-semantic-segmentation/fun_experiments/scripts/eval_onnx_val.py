"""
Evaluate ONNX segmentation models on the LaRS validation set.

Computes:
  - mIoU over 3 classes (obstacle=0, water=1, sky=2)
  - Obstacle detection F1 (challenge metric: 70% pixel coverage threshold)
  - Q = mIoU × F1 (challenge ranking metric)

Metric specification (macvi.org):
  TP: GT obstacle instance where ≥70% of its pixels are predicted as obstacle
  FP: predicted obstacle connected component with no overlap with any GT obstacle
  FN: GT obstacle instance not covered at ≥70%
  F1 = 2*TP / (2*TP + FP + FN)
  Q  = mIoU * F1

Assumptions / caveats (challenge server may differ slightly):
  - GT obstacles: all panoptic segments with category_id in {1, 11-19}
    (Static Obstacle stuff class + all thing obstacle classes)
  - ONNX logits (384×768) are bilinearly upsampled to ori_shape (720×1280), then
    argmax is taken — matching exactly how mmseg's whole_inference(rescale=True)
    works during training evaluation. This is why training reports ~96% mIoU.
  - Predicted blobs: connected components of predicted obstacle class (label=0)
    filtered to area >= MIN_PRED_AREA pixels (at 720×1280 resolution)
  - FP criterion: predicted blob has zero pixel overlap with any GT obstacle segment

Usage:
    python scripts/eval_onnx_val.py
    python scripts/eval_onnx_val.py --models stdc2 pidnets segnext_t

Run from the challenge directory:
    cd /home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation
    python scripts/eval_onnx_val.py
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as TF
from PIL import Image
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = '/home/farms/lars_linked/val'

IMG_DIR       = os.path.join(DATA_ROOT, 'images')
SEG_MASK_DIR  = os.path.join(DATA_ROOT, 'semantic_masks')
PAN_MASK_DIR  = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_masks'
PAN_JSON      = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_annotations.json'

MODELS = {
    'stdc2':    os.path.join(CHALLENGE_DIR, 'outputs/fun_stdc2_768x384/stdc2_768x384.onnx'),
    'pidnets':  os.path.join(CHALLENGE_DIR, 'outputs/fun_pidnets_768x384/pidnets_768x384.onnx'),
    'segnext_t': os.path.join(CHALLENGE_DIR, 'outputs/fun_segnext_t_768x384/segnext_t_768x384.onnx'),
}

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_H, MODEL_W = 384, 768               # ONNX input resolution (H, W)
NUM_CLASSES = 3
IGNORE_LABEL = 255
COVERAGE_THRESH = 0.70                   # challenge TP threshold
MIN_PRED_AREA = 100                      # min pixels for a predicted blob to count
                                         # (at 720×1280 resolution)
OBSTACLE_CATS = set([1] + list(range(11, 20)))  # category_ids that are obstacles

# ImageNet normalisation (RGB order, as used in training)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def preprocess(img_path):
    """Load image, resize to model input, normalise → float32 NCHW tensor."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW
    return x


def run_onnx(session, img_path, ori_h, ori_w):
    """Run model; bilinear-upsample logits to ori_shape then argmax.

    Matches mmseg's whole_inference(rescale=True): logits are upsampled in
    float space before the class decision, giving correct boundary behaviour.
    """
    x = preprocess(img_path)
    logits = session.run(None, {'input': x})[0]              # (1, 3, H, W)
    logits_t = torch.from_numpy(logits)
    logits_up = TF.interpolate(logits_t, size=(ori_h, ori_w),
                               mode='bilinear', align_corners=False)
    pred = logits_up[0].argmax(0).numpy().astype(np.uint8)   # (ori_h, ori_w)
    return pred


def decode_panoptic(pan_path):
    """Return segment-id map (int32) from panoptic PNG.
    Encoding: id = R + G*256 + B*65536
    """
    pan = np.array(Image.open(pan_path).convert('RGB'), dtype=np.int32)
    return pan[:, :, 0] + pan[:, :, 1] * 256 + pan[:, :, 2] * 65536


def compute_iou(pred, gt_mask, num_classes=NUM_CLASSES, ignore=IGNORE_LABEL):
    """Per-class IoU from flat prediction and GT mask."""
    ious = []
    valid = gt_mask != ignore
    for cls in range(num_classes):
        p = (pred == cls) & valid
        g = (gt_mask == cls) & valid
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious


def obstacle_f1(pred_obstacle, pan_id_map, segments_info):
    """
    Compute obstacle detection TP, FP, FN for one image.

    pred_obstacle: bool array (H, W), True where model predicts obstacle
    pan_id_map:    int32 array (H, W), panoptic segment IDs
    segments_info: list of dicts from panoptic_annotations.json
    """
    # Build union of all GT obstacle pixels
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

    # FP: predicted blobs with no overlap with any GT obstacle pixel
    labeled, num_blobs = ndimage.label(pred_obstacle)
    fp = 0
    for blob_id in range(1, num_blobs + 1):
        blob_mask = labeled == blob_id
        if blob_mask.sum() < MIN_PRED_AREA:
            continue
        overlap = (blob_mask & gt_obs_union).sum()
        if overlap == 0:
            fp += 1

    return tp, fp, fn


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(model_name, onnx_path, pan_data):
    print(f'\n{"="*60}')
    print(f'  Model: {model_name}')
    print(f'  ONNX:  {onnx_path}')
    print(f'{"="*60}')

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Accumulators
    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)
    total_tp = total_fp = total_fn = 0
    n_images = 0

    for img_name, pan_ann in pan_data.items():
        stem = os.path.splitext(img_name)[0]
        img_path  = os.path.join(IMG_DIR, img_name)
        seg_path  = os.path.join(SEG_MASK_DIR, stem + '.png')
        pan_path  = os.path.join(PAN_MASK_DIR,  pan_ann['file_name'])

        if not os.path.exists(img_path):
            continue

        # Load GT at original resolution
        gt_seg = np.array(Image.open(seg_path), dtype=np.uint8)
        H, W   = gt_seg.shape
        pan_ids = decode_panoptic(pan_path)

        # Predict: bilinear-upsample logits to ori_shape then argmax
        pred = run_onnx(sess, img_path, H, W)

        # mIoU contribution
        ious = compute_iou(pred, gt_seg)
        for c, iou in enumerate(ious):
            if not np.isnan(iou):
                iou_sum[c] += iou
                iou_cnt[c] += 1

        # Obstacle F1 contribution
        pred_obstacle = (pred == 0)
        tp, fp, fn = obstacle_f1(pred_obstacle, pan_ids, pan_ann['segments_info'])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_images += 1

    # ── Per-class IoU
    class_ious = iou_sum / np.maximum(iou_cnt, 1)
    miou = float(np.nanmean(class_ious))

    # ── F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    q = miou * f1

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
                        choices=list(MODELS.keys()),
                        help='Which models to evaluate (default: all)')
    args = parser.parse_args()

    # Load panoptic annotations indexed by image filename
    with open(PAN_JSON) as f:
        pan_json = json.load(f)

    # Build image_id → file_name map from 'images' list
    id_to_fname = {img['id']: img['file_name'] for img in pan_json['images']}

    # Build img_filename → {file_name (pan png), segments_info}
    pan_data = {}
    for ann in pan_json['annotations']:
        img_fname = id_to_fname[ann['image_id']]
        pan_data[img_fname] = {
            'file_name':    ann['file_name'],
            'segments_info': ann['segments_info'],
        }

    results = []
    for name in args.models:
        path = MODELS[name]
        if not os.path.exists(path):
            print(f'SKIP {name}: ONNX not found at {path}')
            continue
        r = evaluate(name, path, pan_data)
        results.append(r)

    # ── Summary table
    if len(results) > 1:
        print(f'\n{"="*70}')
        print('  SUMMARY')
        print(f'{"="*70}')
        hdr = f"{'Model':<12} {'mIoU':>7} {'IoU obs':>8} {'IoU wat':>8} {'IoU sky':>8} {'F1':>7} {'Q':>7}"
        print(hdr)
        print('-' * 70)
        for r in results:
            print(f"{r['model']:<12} {r['miou']*100:>6.2f}% {r['iou_obs']*100:>7.2f}% "
                  f"{r['iou_wat']*100:>7.2f}% {r['iou_sky']*100:>7.2f}% "
                  f"{r['f1']*100:>6.2f}% {r['q']*100:>6.2f}%")
        print(f'{"="*70}')
        print('  Note: logits bilinearly upsampled to 720×1280 before argmax — matches training eval.')
        print('  F1 threshold: 70% GT coverage. Min pred blob size: 100px.')
        print('  This is a proxy for the challenge server metric (INT8 on RVC4 may differ).')


if __name__ == '__main__':
    main()
