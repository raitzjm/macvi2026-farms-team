"""
Phase 1 ONNX export and validation script.

For each Phase 1 model:
  1. Loads best checkpoint and runs PyTorch inference on val set → mIoU + F1
  2. Exports logits ONNX to outputs/phase1/{model}/{model}_logits.onnx
  3. Exports argmax ONNX to outputs/phase1/{model}/{model}_argmax.onnx
  4. Runs ONNX inference on val set → mIoU + F1
  5. Computes delta (ONNX − PyTorch) for both metrics

Artifacts (ONNX, .pth) are NOT committed — outputs/ is gitignored.
Results are printed to stdout for capture into phase1_report.md.

Usage (from repo root):
    python challenges/embedded-semantic-segmentation/phase1/scripts/export_and_eval.py
    python challenges/embedded-semantic-segmentation/phase1/scripts/export_and_eval.py --models stdc2
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
CHALLENGE_DIR = os.path.join(REPO_ROOT, 'challenges', 'embedded-semantic-segmentation')
PHASE1_DIR = os.path.join(CHALLENGE_DIR, 'phase1')
CONFIGS_DIR = os.path.join(PHASE1_DIR, 'configs')
OUTPUTS_DIR = os.path.join(CHALLENGE_DIR, 'outputs', 'phase1')

VAL_IMG_DIR  = '/home/farms/lars_linked/val/images'
VAL_SEG_DIR  = '/home/farms/lars_linked/val/semantic_masks'
PAN_MASK_DIR = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_masks'
PAN_JSON     = '/home/farms/data/lars_dataset/ann_dir/val/panoptic_annotations.json'

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_H, MODEL_W = 384, 768

MODELS = {
    'stdc2': dict(
        config=os.path.join(CONFIGS_DIR, 'stdc2_768x384_lars_phase1.py'),
        ckpt=os.path.join(OUTPUTS_DIR, 'stdc2', 'best_mIoU_iter_1250.pth'),
        syncbn=True,
        ham_fix=False,
        align_corners=False,
    ),
    'bisenetv2': dict(
        config=os.path.join(CONFIGS_DIR, 'bisenetv2_768x384_lars_phase1.py'),
        ckpt=os.path.join(OUTPUTS_DIR, 'bisenetv2', 'best_mIoU_iter_1500.pth'),
        syncbn=True,
        ham_fix=False,
        align_corners=False,
    ),
    'segnext_t': dict(
        config=os.path.join(CONFIGS_DIR, 'segnext_t_768x384_lars_phase1.py'),
        ckpt=os.path.join(OUTPUTS_DIR, 'segnext_t', 'best_mIoU_iter_1500.pth'),
        syncbn=False,
        ham_fix=True,
        align_corners=False,
    ),
    'segformer_mit-b2': dict(
        config=os.path.join(CONFIGS_DIR, 'segformer_mit-b2_768x384_lars_phase1.py'),
        ckpt=os.path.join(OUTPUTS_DIR, 'segformer_mit-b2', 'best_mIoU_iter_1000.pth'),
        syncbn=False,
        ham_fix=False,
        align_corners=False,
    ),
}

NUM_CLASSES = 3
IGNORE_LABEL = 255
COVERAGE_THRESH = 0.70
MIN_PRED_AREA = 100
OBSTACLE_CATS = set([1] + list(range(11, 20)))

MEAN_T = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD_T  = torch.tensor([58.395,  57.12,  57.375]).view(1, 3, 1, 1)
MEAN_NP = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_NP  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ── SyncBN → BN conversion ───────────────────────────────────────────────────

def convert_syncbn(module):
    """Recursively replace SyncBatchNorm with BatchNorm2d (for ONNX export)."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features, child.eps, child.momentum,
                                child.affine, child.track_running_stats)
            if child.affine:
                bn.weight.data = child.weight.data.clone()
                bn.bias.data   = child.bias.data.clone()
            bn.running_mean = child.running_mean.clone()
            bn.running_var  = child.running_var.clone()
            setattr(module, name, bn)
        else:
            convert_syncbn(child)
    return module


# ── Model wrappers ────────────────────────────────────────────────────────────

class SegWrapperLogits(nn.Module):
    """backbone(img) → decode_head → bilinear upsample → logits (1, C, H, W)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        # NOTE: align_corners=False used for all models. STDC2's decode_head has
        # align_corners=True in its config — this is a known inconsistency that had
        # no measurable effect (<0.01% delta) in Phase 1. Phase 2 export should
        # match align_corners per model (add to MODELS registry).
        logits = F.interpolate(logits, size=(MODEL_H, MODEL_W),
                               mode='bilinear', align_corners=False)
        return logits


class SegWrapperArgmax(nn.Module):
    """Same as SegWrapperLogits but appends argmax → (1, 1, H, W) int64.

    Uses ArgMax(keepdims=0) + Unsqueeze — required by SNPE (keepdims=1 not supported).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        logits = F.interpolate(logits, size=(MODEL_H, MODEL_W),
                               mode='bilinear', align_corners=False)
        return torch.argmax(logits, dim=1).unsqueeze(1)


# ── Model loading ─────────────────────────────────────────────────────────────

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model(cfg_path, ckpt_path, syncbn, ham_fix):
    """Load mmseg model (v0.30.0 API), apply patches, return eval model."""
    sys.path.insert(0, REPO_ROOT)
    from mmseg.apis import init_segmentor

    model = init_segmentor(cfg_path, ckpt_path, device=DEVICE)
    model.eval()

    if syncbn:
        # SyncBN → BN required for ONNX export (SyncBN not ONNX-exportable)
        convert_syncbn(model)

    if ham_fix:
        # SegNeXt-T: rand_init=True → non-deterministic NMF bases → large
        # PyTorch vs ONNX diff. Switch to deterministic (bases computed once).
        ham = model.decode_head.hamburger.ham
        ham.rand_init = False
        with torch.no_grad():
            dev = next(model.parameters()).device
            _ = model.decode_head(model.backbone(torch.zeros(1, 3, MODEL_H, MODEL_W, device=dev)))

    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_pt(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    return (x - MEAN_T) / STD_T


def preprocess_np(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    x = (np.array(img, dtype=np.float32) - MEAN_NP) / STD_NP
    return x.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW


# ── Metrics ───────────────────────────────────────────────────────────────────

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


def eval_predictions(preds_by_name, pan_data):
    """Compute mIoU and F1 from {img_name: pred_array} dict."""
    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)
    total_tp = total_fp = total_fn = 0
    n_images = 0

    for img_name, pan_ann in pan_data.items():
        if img_name not in preds_by_name:
            continue
        stem    = os.path.splitext(img_name)[0]
        seg_path = os.path.join(VAL_SEG_DIR, stem + '.png')
        pan_path = os.path.join(PAN_MASK_DIR, pan_ann['file_name'])

        gt_seg  = np.array(Image.open(seg_path), dtype=np.uint8)
        H, W    = gt_seg.shape
        pan_ids = decode_panoptic(pan_path)
        pred    = preds_by_name[img_name]

        ious = compute_iou(pred, gt_seg)
        for c, iou in enumerate(ious):
            if not np.isnan(iou):
                iou_sum[c] += iou
                iou_cnt[c] += 1

        tp, fp, fn = obstacle_f1((pred == 0), pan_ids, pan_ann['segments_info'])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_images += 1

    class_ious = iou_sum / np.maximum(iou_cnt, 1)
    miou = float(np.nanmean(class_ious))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    q  = miou * f1

    return dict(miou=miou, f1=f1, q=q, precision=precision, recall=recall,
                iou_obs=class_ious[0], iou_wat=class_ious[1], iou_sky=class_ious[2],
                tp=total_tp, fp=total_fp, fn=total_fn, n=n_images)


# ── Per-model pipeline ────────────────────────────────────────────────────────

def run_model(model_name, cfg, pan_data):
    print(f'\n{"="*70}')
    print(f'  MODEL: {model_name}')
    print(f'  Checkpoint: {cfg["ckpt"]}')
    print(f'{"="*70}')

    out_dir = os.path.join(OUTPUTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    onnx_logits = os.path.join(out_dir, f'{model_name}_logits.onnx')
    onnx_argmax = os.path.join(out_dir, f'{model_name}_argmax.onnx')

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print(f'\n[1/4] Loading model...')
    model = load_model(cfg['config'], cfg['ckpt'], cfg['syncbn'], cfg['ham_fix'])
    wrapper_logits = SegWrapperLogits(model).eval()
    wrapper_argmax = SegWrapperArgmax(model).eval()

    # ── Step 2: PyTorch eval (on DEVICE — GPU if available) ──────────────────
    print(f'\n[2/4] PyTorch eval on val set ({len(pan_data)} images) [device={DEVICE}]...')
    img_names = sorted(os.listdir(VAL_IMG_DIR))
    dev = next(model.parameters()).device
    pt_preds = {}
    for img_name in img_names:
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        gt_path  = os.path.join(VAL_SEG_DIR, os.path.splitext(img_name)[0] + '.png')
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        ori_h, ori_w = gt.shape
        x = preprocess_pt(img_path).to(dev)
        with torch.no_grad():
            logits = wrapper_logits(x)
        logits_up = F.interpolate(logits.cpu(), size=(ori_h, ori_w),
                                  mode='bilinear', align_corners=False)
        pt_preds[img_name] = logits_up[0].argmax(0).numpy().astype(np.uint8)

    pt_res = eval_predictions(pt_preds, pan_data)
    print(f'  PyTorch — mIoU: {pt_res["miou"]*100:.2f}%  F1: {pt_res["f1"]*100:.2f}%  '
          f'Q: {pt_res["q"]*100:.2f}%')
    print(f'  IoU obs/wat/sky: {pt_res["iou_obs"]*100:.2f}% / {pt_res["iou_wat"]*100:.2f}% / {pt_res["iou_sky"]*100:.2f}%')

    # ── Step 3: ONNX export (model moved to CPU) ──────────────────────────────
    print(f'\n[3/4] Exporting ONNX...')
    model.cpu()
    wrapper_logits = wrapper_logits.cpu()
    wrapper_argmax = wrapper_argmax.cpu()
    dummy = torch.randn(1, 3, MODEL_H, MODEL_W)

    # Logits export (for local eval — opset 13 for SegFormer unflatten compat)
    torch.onnx.export(
        wrapper_logits, dummy, onnx_logits,
        opset_version=13,
        input_names=['input'], output_names=['output'],
    )

    # Argmax export (for challenge submission, opset 13 required by SNPE)
    torch.onnx.export(
        wrapper_argmax, dummy, onnx_argmax,
        opset_version=13,
        input_names=['input'], output_names=['output'],
    )

    # Sanity check: PyTorch vs ONNX logit diff on dummy input
    sess_l = ort.InferenceSession(onnx_logits, providers=['CPUExecutionProvider'])
    onnx_dummy_out = sess_l.run(None, {'input': dummy.numpy()})[0]
    with torch.no_grad():
        pt_dummy_out = wrapper_logits(dummy).numpy()
    max_diff = float(np.abs(pt_dummy_out - onnx_dummy_out).max())
    status = 'PASSED ✓' if max_diff < 1e-3 else f'WARNING: large diff'
    print(f'  Logits export {status} (max abs diff on dummy: {max_diff:.2e})')
    print(f'  Logits ONNX: {onnx_logits}')
    print(f'  Argmax ONNX: {onnx_argmax}')

    # ── Step 4: ONNX eval ─────────────────────────────────────────────────────
    print(f'\n[4/4] ONNX eval on val set...')
    onnx_preds = {}
    for img_name in img_names:
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        gt_path  = os.path.join(VAL_SEG_DIR, os.path.splitext(img_name)[0] + '.png')
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        ori_h, ori_w = gt.shape
        x_np = preprocess_np(img_path)
        logits_np = sess_l.run(None, {'input': x_np})[0]
        logits_t  = torch.from_numpy(logits_np)
        logits_up = F.interpolate(logits_t, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        onnx_preds[img_name] = logits_up[0].argmax(0).numpy().astype(np.uint8)

    onnx_res = eval_predictions(onnx_preds, pan_data)
    print(f'  ONNX    — mIoU: {onnx_res["miou"]*100:.2f}%  F1: {onnx_res["f1"]*100:.2f}%  '
          f'Q: {onnx_res["q"]*100:.2f}%')
    print(f'  IoU obs/wat/sky: {onnx_res["iou_obs"]*100:.2f}% / {onnx_res["iou_wat"]*100:.2f}% / {onnx_res["iou_sky"]*100:.2f}%')

    d_miou = (onnx_res['miou'] - pt_res['miou']) * 100
    d_f1   = (onnx_res['f1']   - pt_res['f1'])   * 100
    d_q    = (onnx_res['q']    - pt_res['q'])     * 100
    print(f'\n  Delta (ONNX−PyTorch): mIoU{d_miou:+.2f}%  F1{d_f1:+.2f}%  Q{d_q:+.2f}%')
    if abs(d_miou) > 0.5:
        print(f'  WARNING: mIoU delta exceeds 0.5% threshold — investigate before archiving.')
    else:
        print(f'  Delta within acceptable range (<0.5%).')

    return dict(model=model_name, pt=pt_res, onnx=onnx_res,
                d_miou=d_miou, d_f1=d_f1, d_q=d_q, max_diff=max_diff)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help='Which models to process (default: all)')
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

    all_results = []
    for name in args.models:
        if name not in MODELS:
            print(f'Unknown model: {name}')
            continue
        r = run_model(name, MODELS[name], pan_data)
        all_results.append(r)

    if len(all_results) > 1:
        print(f'\n{"="*90}')
        print('  PHASE 1 ONNX VALIDATION SUMMARY')
        print(f'{"="*90}')
        hdr = (f"{'Model':<20} {'PyTorch mIoU':>13} {'PyTorch F1':>11} {'ONNX mIoU':>10} "
               f"{'ONNX F1':>8} {'Δ mIoU':>8} {'Δ F1':>7}")
        print(hdr)
        print('-' * 90)
        for r in all_results:
            pt, onnx = r['pt'], r['onnx']
            print(f"{r['model']:<20} {pt['miou']*100:>12.2f}% {pt['f1']*100:>10.2f}% "
                  f"{onnx['miou']*100:>9.2f}% {onnx['f1']*100:>7.2f}% "
                  f"{r['d_miou']:>+7.2f}% {r['d_f1']:>+6.2f}%")
        print(f'{"="*90}')
        print('  F1 threshold: 70% GT coverage. Min pred blob: 100px at 720×1280.')
        print('  This is a proxy metric — challenge server uses INT8 on RVC4 (may differ).')


if __name__ == '__main__':
    main()
