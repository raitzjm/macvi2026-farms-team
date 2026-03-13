"""
Phase 3 ONNX export and evaluation script (mmseg 1.2.2 API).

Preprocessing matches challenge server (minus quantization on RVC4):
  keep-ratio bilinear resize → center reflect-pad to 384×768 → ImageNet normalize
  After inference, padding is removed before nearest-neighbour resize to original resolution.

Pipeline per model (default):
  1. Load PyTorch checkpoint, run inference at training dtype (logits+bilinear) → mIoU + F1
     (upper bound reference; not exported — supports fp32/fp16/bf16/mixed precision)
  2. Export argmax ONNX at training dtype (fp32/fp16/bf16) → outputs/phase3/{model}/{model}_argmax.onnx
  3. Sanity-check: pixel agreement between PyTorch argmax and ONNX argmax on dummy input
  4. Run argmax ONNX on val set → mIoU + F1 + FPS  (server-matching)

With --quant (appended after step 4):
  5. Static INT8 quantization of argmax ONNX (calibrated on 50 val images)
     → {model}_argmax_int8.onnx
  6. Run INT8 argmax ONNX on val set → mIoU + F1  (approximate INT8 impact)

With --eval-only: skip steps 1-3, run step 4 from existing argmax ONNX.
With --eval-only --quant: run step 4 + step 6 (INT8 ONNX must already exist).

Usage (from repo root, macvi2 env active):
    python challenges/embedded-semantic-segmentation/phase3/scripts/export_and_eval.py
    python challenges/embedded-semantic-segmentation/phase3/scripts/export_and_eval.py --models pidnets
    python challenges/embedded-semantic-segmentation/phase3/scripts/export_and_eval.py --quant
    python challenges/embedded-semantic-segmentation/phase3/scripts/export_and_eval.py --eval-only
    python challenges/embedded-semantic-segmentation/phase3/scripts/export_and_eval.py --eval-only --quant --models pidnets
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
CHALLENGE_DIR = os.path.join(REPO_ROOT, 'challenges', 'embedded-semantic-segmentation')
PHASE3_DIR = os.path.join(CHALLENGE_DIR, 'phase3')
CONFIGS_DIR = os.path.join(PHASE3_DIR, 'configs')
ARTIFACTS_DIR = os.path.join(PHASE3_DIR, 'artifacts')
OUTPUTS_DIR = os.path.join(CHALLENGE_DIR, 'outputs', 'phase3')

VAL_IMG_DIR  = '/home/farms/lars_linked/val/images'
VAL_SEG_DIR  = '/home/farms/lars_linked/val/semantic_masks'
PAN_MASK_DIR = '/home/farms/data/datasets/lars_dataset/ann_dir/val/panoptic_masks'
PAN_JSON     = '/home/farms/data/datasets/lars_dataset/ann_dir/val/panoptic_annotations.json'

MODEL_H, MODEL_W = 384, 768
CALIB_SAMPLES = 50


def _best_ckpt(model_name):
    out_dir = os.path.join(OUTPUTS_DIR, model_name)
    if not os.path.isdir(out_dir):
        return None
    candidates = [f for f in os.listdir(out_dir)
                  if f.startswith('best_mIoU') and f.endswith('.pth')]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(out_dir, candidates[-1])


MODELS = {
    'stdc2': dict(
        config=os.path.join(CONFIGS_DIR, 'stdc2_768x384_lars_phase3.py'),
        ckpt=os.path.join(ARTIFACTS_DIR, 'stdc2', 'best_mIoU_iter_5000.pth'),
        syncbn=True, ham_fix=False, align_corners=True,
    ),
    'pidnets': dict(
        config=os.path.join(CONFIGS_DIR, 'pidnets_768x384_lars_phase3.py'),
        ckpt=os.path.join(ARTIFACTS_DIR, 'pidnets', 'best_mIoU_iter_8000.pth'),
        syncbn=True, ham_fix=False, align_corners=True,
    ),
}

NUM_CLASSES = 3
IGNORE_LABEL = 255
COVERAGE_THRESH = 0.70
MIN_PRED_AREA = 100
OBSTACLE_CATS = set([1] + list(range(11, 20)))

MEAN_NP = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_NP  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ── SyncBN → BN conversion ───────────────────────────────────────────────────

def convert_syncbn(module):
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
    """PyTorch-only: backbone → decode_head → bilinear upsample → logits (1, C, H, W).

    Used for step 2 float32 reference eval. Not exported to ONNX.
    """
    def __init__(self, model, align_corners):
        super().__init__()
        self.model = model
        self.align_corners = align_corners

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return F.interpolate(logits, size=(MODEL_H, MODEL_W),
                             mode='bilinear', align_corners=self.align_corners)


class SegWrapperArgmax(nn.Module):
    """Exported to ONNX: backbone → decode_head → argmax → (1, 1, H, W) int64."""
    def __init__(self, model, align_corners):
        super().__init__()
        self.model = model
        self.align_corners = align_corners

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = F.interpolate(logits, size=(MODEL_H, MODEL_W),
                               mode='bilinear', align_corners=self.align_corners)
        return torch.argmax(logits, dim=1).unsqueeze(1)


# ── Model loading ─────────────────────────────────────────────────────────────

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model(cfg_path, ckpt_path, syncbn, ham_fix):
    sys.path.insert(0, REPO_ROOT)
    from mmseg.apis import init_model
    model = init_model(cfg_path, ckpt_path, device=DEVICE)
    model.eval()
    if syncbn:
        convert_syncbn(model)
    if ham_fix:
        ham = model.decode_head.hamburger.ham
        ham.rand_init = False
        with torch.no_grad():
            dev = next(model.parameters()).device
            _ = model.decode_head(model.backbone(
                torch.zeros(1, 3, MODEL_H, MODEL_W, device=dev)))
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def server_preprocess(img_path):
    """Keep-ratio resize + center reflect-pad + normalize (matches challenge server).

    Returns:
        x_pt:  (1, 3, H, W) float32 torch tensor
        x_np:  (1, 3, H, W) float32 numpy array (for ONNX sessions)
        info:  dict(pad_top, pad_left, new_h, new_w, ori_h, ori_w)
    """
    img = Image.open(img_path).convert('RGB')
    ori_w, ori_h = img.size

    scale = min(MODEL_H / ori_h, MODEL_W / ori_w)
    new_h = round(ori_h * scale)
    new_w = round(ori_w * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    pad_top    = (MODEL_H - new_h) // 2
    pad_bottom = MODEL_H - new_h - pad_top
    pad_left   = (MODEL_W - new_w) // 2
    pad_right  = MODEL_W - new_w - pad_left

    mode = 'reflect'
    if pad_top >= new_h or pad_bottom >= new_h or pad_left >= new_w or pad_right >= new_w:
        mode = 'symmetric'
    arr = np.pad(arr, [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode=mode)
    arr = (arr - MEAN_NP) / STD_NP

    x_pt = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    x_np = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    info = dict(pad_top=pad_top, pad_left=pad_left,
                new_h=new_h, new_w=new_w, ori_h=ori_h, ori_w=ori_w)
    return x_pt, x_np, info


def unpad_and_upsample(logits, info, align_corners):
    """PyTorch float32 path: crop pad, bilinear upsample, argmax → uint8."""
    t, l = info['pad_top'], info['pad_left']
    h, w = info['new_h'], info['new_w']
    crop = logits[:, :, t:t + h, l:l + w]
    up   = F.interpolate(crop, size=(info['ori_h'], info['ori_w']),
                         mode='bilinear', align_corners=align_corners)
    return up[0].argmax(0).numpy().astype(np.uint8)


def unpad_argmax(argmax_out, info):
    """Argmax ONNX path: crop pad from (1,1,H,W) int64, nearest-neighbour resize → uint8."""
    t, l = info['pad_top'], info['pad_left']
    h, w = info['new_h'], info['new_w']
    pred = argmax_out[0, 0, t:t + h, l:l + w]
    pred_pil = Image.fromarray(pred.astype(np.uint8))
    return np.array(pred_pil.resize((info['ori_w'], info['ori_h']), Image.NEAREST),
                    dtype=np.uint8)


# ── INT8 quantization ─────────────────────────────────────────────────────────

class _ValCalibReader(CalibrationDataReader):
    """Feeds preprocessed val images to onnxruntime static quantization calibration."""

    def __init__(self, img_paths, n_samples):
        paths = img_paths[:n_samples]
        print(f'    Calibrating on {len(paths)} images...')
        self._data = [{'input': server_preprocess(p)[1]} for p in paths]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        d = self._data[self._idx]
        self._idx += 1
        return d

    def rewind(self):
        self._idx = 0


def quantize_argmax_onnx(onnx_argmax, onnx_int8, img_paths):
    """Static INT8 quantization of the argmax ONNX.

    Quantizes conv/BN/matmul ops (QUInt8 activations + weights); the ArgMax op
    itself passes through unchanged with int64 output.
    Closest generic approximation to Qualcomm SNPE INT8 quantization.
    """
    reader = _ValCalibReader(img_paths, CALIB_SAMPLES)
    quantize_static(
        onnx_argmax,
        onnx_int8,
        calibration_data_reader=reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )


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
        covered = (pred_obstacle & gt_mask).sum()
        if gt_mask.sum() > 0 and covered / gt_mask.sum() >= COVERAGE_THRESH:
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
    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)
    total_tp = total_fp = total_fn = 0
    n_images = 0

    for img_name, pan_ann in pan_data.items():
        if img_name not in preds_by_name:
            continue
        stem     = os.path.splitext(img_name)[0]
        seg_path = os.path.join(VAL_SEG_DIR, stem + '.png')
        pan_path = os.path.join(PAN_MASK_DIR, pan_ann['file_name'])

        gt_seg  = np.array(Image.open(seg_path), dtype=np.uint8)
        pan_ids = decode_panoptic(pan_path)
        pred    = preds_by_name[img_name]

        for c, iou in enumerate(compute_iou(pred, gt_seg)):
            if not np.isnan(iou):
                iou_sum[c] += iou
                iou_cnt[c] += 1

        tp, fp, fn = obstacle_f1((pred == 0), pan_ids, pan_ann['segments_info'])
        total_tp += tp; total_fp += fp; total_fn += fn
        n_images += 1

    class_ious = iou_sum / np.maximum(iou_cnt, 1)
    miou = float(np.nanmean(class_ious))
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(miou=miou, f1=f1, q=miou * f1, precision=prec, recall=rec,
                iou_obs=class_ious[0], iou_wat=class_ious[1], iou_sky=class_ious[2],
                tp=total_tp, fp=total_fp, fn=total_fn, n=n_images)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _platform_info():
    """Return a string describing the current evaluation platform."""
    ort_ver = ort.__version__
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        cuda = torch.version.cuda
        return f'{gpu}, CUDA {cuda}, ORT {ort_ver}'
    return f'CPU, ORT {ort_ver}'


def _run_argmax_eval(sess, img_names, pan_data):
    """Run an argmax ONNX session (any precision or INT8) on the val set.

    Introspects the session input dtype (float32 / float16) and casts accordingly.
    Returns (metrics_dict, fps) where fps is images/second on this platform.
    """
    inp_type = sess.get_inputs()[0].type   # e.g. 'tensor(float)' or 'tensor(float16)'
    np_dtype = np.float16 if 'float16' in inp_type else np.float32

    preds = {}
    t_start    = time.perf_counter()
    n_inferred = 0
    for img_name in img_names:
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        gt_path  = os.path.join(VAL_SEG_DIR, os.path.splitext(img_name)[0] + '.png')
        if not os.path.exists(gt_path):
            continue
        _, x_np, info = server_preprocess(img_path)
        if np_dtype != np.float32:
            x_np = x_np.astype(np_dtype)
        argmax_np = sess.run(None, {'input': x_np})[0]  # (1, 1, H, W) int64
        preds[img_name] = unpad_argmax(argmax_np, info)
        n_inferred += 1
    t_elapsed = time.perf_counter() - t_start
    fps = n_inferred / t_elapsed if t_elapsed > 0 else float('nan')

    return eval_predictions(preds, pan_data), fps


def _print_res(label, res):
    print(f'  {label:<14} mIoU: {res["miou"]*100:.2f}%  '
          f'F1: {res["f1"]*100:.2f}%  Q: {res["q"]*100:.2f}%  '
          f'Pr: {res["precision"]*100:.2f}%  Re: {res["recall"]*100:.2f}%')
    print(f'  {" "*14} IoU obs/wat/sky: {res["iou_obs"]*100:.2f}% / '
          f'{res["iou_wat"]*100:.2f}% / {res["iou_sky"]*100:.2f}%')


def _print_delta(label_a, label_b, res_a, res_b):
    dm = (res_b['miou'] - res_a['miou']) * 100
    df = (res_b['f1']   - res_a['f1'])   * 100
    dq = (res_b['q']    - res_a['q'])     * 100
    print(f'\n  {label_a} → {label_b}: ΔmIoU{dm:+.2f}%  ΔF1{df:+.2f}%  ΔQ{dq:+.2f}%')


# ── Per-model pipeline ────────────────────────────────────────────────────────

def run_model(model_name, cfg, pan_data, eval_only=False, quant=False):
    print(f'\n{"="*70}')
    print(f'  MODEL: {model_name}')

    out_dir      = os.path.join(OUTPUTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    onnx_argmax  = os.path.join(out_dir, f'{model_name}_argmax.onnx')
    onnx_int8    = os.path.join(out_dir, f'{model_name}_argmax_int8.onnx')
    img_names    = sorted(os.listdir(VAL_IMG_DIR))
    img_paths    = [os.path.join(VAL_IMG_DIR, n) for n in img_names]

    result = dict(model=model_name)

    # ── eval-only: skip model load + export ───────────────────────────────────
    if eval_only:
        if not os.path.exists(onnx_argmax):
            print(f'  ERROR: {onnx_argmax} not found. Run without --eval-only first.')
            print(f'{"="*70}')
            return None
        print(f'  Mode: eval-only  |  ONNX: {onnx_argmax}')
        print(f'{"="*70}')

        print(f'\n[fp32] Argmax ONNX eval...')
        sess_onnx          = ort.InferenceSession(onnx_argmax, providers=['CPUExecutionProvider'])
        fp32_res, fp32_fps = _run_argmax_eval(sess_onnx, img_names, pan_data)
        _print_res('ONNX', fp32_res)
        print(f'  {"FPS":<14} {fp32_fps:.1f} img/s  [{_platform_info()}]')
        result['fp32']     = fp32_res
        result['fp32_fps'] = fp32_fps

        if quant:
            if not os.path.exists(onnx_int8):
                print(f'  WARNING: INT8 ONNX not found at {onnx_int8}. '
                      f'Run without --eval-only to generate it.')
            else:
                print(f'\n[int8] INT8 argmax ONNX eval...')
                sess_int8          = ort.InferenceSession(onnx_int8, providers=['CPUExecutionProvider'])
                int8_res, int8_fps = _run_argmax_eval(sess_int8, img_names, pan_data)
                _print_res('int8', int8_res)
                print(f'  {"FPS":<14} {int8_fps:.1f} img/s  [{_platform_info()}]')
                _print_delta('ONNX', 'int8', fp32_res, int8_res)
                result['int8']     = int8_res
                result['int8_fps'] = int8_fps

        return result

    # ── full pipeline ─────────────────────────────────────────────────────────
    ckpt_path = cfg.get('ckpt') or _best_ckpt(model_name)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f'  ERROR: No checkpoint found for {model_name}. Skipping.')
        return None
    print(f'  Checkpoint: {ckpt_path}')
    print(f'{"="*70}')

    align_corners = cfg.get('align_corners', False)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print(f'\n[1/4] Loading model...')
    model = load_model(cfg['config'], ckpt_path, cfg['syncbn'], cfg['ham_fix'])
    wrapper_logits = SegWrapperLogits(model, align_corners).eval()
    wrapper_argmax = SegWrapperArgmax(model, align_corners).eval()

    # ── Step 2: PyTorch ref eval (logits + bilinear — not exported) ─────────
    dev        = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype   # fp32, fp16, bf16 — whatever was trained
    print(f'\n[2/4] PyTorch eval ({len(pan_data)} images) '
          f'[device={DEVICE}, dtype={model_dtype}]...')
    pt_preds = {}
    for img_name in img_names:
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        gt_path  = os.path.join(VAL_SEG_DIR, os.path.splitext(img_name)[0] + '.png')
        if not os.path.exists(gt_path):
            continue
        x, _, info = server_preprocess(img_path)
        with torch.no_grad():
            logits = wrapper_logits(x.to(dev, dtype=model_dtype))
        pt_preds[img_name] = unpad_and_upsample(logits.float().cpu(), info, align_corners)

    pt_res = eval_predictions(pt_preds, pan_data)
    _print_res('PyTorch', pt_res)
    result['pt'] = pt_res

    # ── Step 3: Export argmax ONNX + pixel-agreement sanity check ────────────
    dtype_str = str(model_dtype).replace('torch.', '')
    print(f'\n[3/4] Exporting argmax ONNX ({dtype_str})...')
    model.cpu()
    wrapper_argmax = wrapper_argmax.cpu()
    dummy = torch.randn(1, 3, MODEL_H, MODEL_W, dtype=model_dtype)

    torch.onnx.export(wrapper_argmax, dummy, onnx_argmax, opset_version=13,
                      input_names=['input'], output_names=['output'])

    # Sanity check: pixel agreement between PyTorch argmax and ONNX argmax on a real image.
    # Use a real image (not random noise) because near-zero logits for noise amplify
    # tiny ORT vs PyTorch numerical differences, causing spurious low agreement.
    sess_argmax = ort.InferenceSession(onnx_argmax, providers=['CPUExecutionProvider'])
    inp_type    = sess_argmax.get_inputs()[0].type
    np_dtype    = np.float16 if 'float16' in inp_type else np.float32
    first_img   = sorted(os.listdir(VAL_IMG_DIR))[0]
    chk_pt, chk_np, _ = server_preprocess(os.path.join(VAL_IMG_DIR, first_img))
    chk_np = chk_np.astype(np_dtype)
    chk_pt = chk_pt.to(dtype=dummy.dtype)                                # match model dtype
    onnx_dummy  = sess_argmax.run(None, {'input': chk_np})[0]            # (1,1,H,W) int64
    with torch.no_grad():
        pt_dummy = wrapper_argmax(chk_pt).numpy()                        # (1,1,H,W) int64
    agreement = float((pt_dummy == onnx_dummy).mean())
    status = 'PASSED ✓' if agreement > 0.999 else 'WARNING: low agreement'
    print(f'  Argmax export {status} (pixel agreement on {first_img[:30]}: {agreement*100:.3f}%)')
    print(f'  Argmax ONNX: {onnx_argmax}  [{inp_type}]')
    result['agreement'] = agreement

    # ── Step 4: argmax ONNX eval ──────────────────────────────────────────────
    print(f'\n[4/4] Argmax ONNX eval (nearest-neighbour resize)...')
    fp32_res, fp32_fps = _run_argmax_eval(sess_argmax, img_names, pan_data)
    _print_res('ONNX', fp32_res)
    print(f'  {"FPS":<14} {fp32_fps:.1f} img/s  [{_platform_info()}]')
    _print_delta('PyTorch', 'ONNX', pt_res, fp32_res)
    result['fp32'] = fp32_res
    result['fp32_fps'] = fp32_fps

    # ── Steps 5+6: INT8 quantization + eval (optional) ────────────────────────
    if quant:
        print(f'\n[5/6] INT8 static quantization of argmax ONNX...')
        quantize_argmax_onnx(onnx_argmax, onnx_int8, img_paths)
        print(f'  INT8 ONNX: {onnx_int8}')

        print(f'\n[6/6] Argmax INT8 eval...')
        sess_int8          = ort.InferenceSession(onnx_int8, providers=['CPUExecutionProvider'])
        int8_res, int8_fps = _run_argmax_eval(sess_int8, img_names, pan_data)
        _print_res('int8', int8_res)
        print(f'  {"FPS":<14} {int8_fps:.1f} img/s  [{_platform_info()}]')
        _print_delta('ONNX', 'int8', fp32_res, int8_res)
        result['int8']     = int8_res
        result['int8_fps'] = int8_fps

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help='Which models to process (default: all)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip model load and ONNX export; use existing argmax ONNX')
    parser.add_argument('--quant', action='store_true',
                        help='Also run static INT8 quantization + eval (approx. server quant)')
    args = parser.parse_args()

    print(f'Platform: {_platform_info()}')

    with open(PAN_JSON) as f:
        pan_json = json.load(f)
    id_to_fname = {img['id']: img['file_name'] for img in pan_json['images']}
    pan_data = {}
    for ann in pan_json['annotations']:
        img_fname = id_to_fname[ann['image_id']]
        pan_data[img_fname] = {'file_name': ann['file_name'],
                               'segments_info': ann['segments_info']}

    all_results = []
    for name in args.models:
        if name not in MODELS:
            print(f'Unknown model: {name}')
            continue
        r = run_model(name, MODELS[name], pan_data,
                      eval_only=args.eval_only, quant=args.quant)
        if r is not None:
            all_results.append(r)

    if len(all_results) < 2:
        return

    has_pt   = all(r.get('pt')   is not None for r in all_results)
    has_int8 = any(r.get('int8') is not None for r in all_results)

    print(f'\n{"="*110}')
    if has_int8:
        print('  PHASE 3 SUMMARY  —  fp32 argmax | INT8 argmax (approx. quantization)')
        print(f'{"="*110}')
        hdr = (f"{'Model':<22} {'FP32 mIoU':>10} {'FP32 F1':>8} {'FP32 Q':>7}"
               f" {'INT8 mIoU':>10} {'INT8 F1':>8} {'INT8 Q':>7} {'ΔmIoU':>7} {'ΔF1':>6}")
        print(hdr)
        print('-' * 110)
        for r in all_results:
            fp = r.get('fp32', {})
            i8 = r.get('int8', {})
            fp_m = fp.get('miou', float('nan')) * 100
            fp_f = fp.get('f1',   float('nan')) * 100
            fp_q = fp.get('q',    float('nan')) * 100
            i8_m = i8.get('miou', float('nan')) * 100 if i8 else float('nan')
            i8_f = i8.get('f1',   float('nan')) * 100 if i8 else float('nan')
            i8_q = i8.get('q',    float('nan')) * 100 if i8 else float('nan')
            print(f"{r['model']:<22}"
                  f" {fp_m:>9.2f}% {fp_f:>7.2f}% {fp_q:>6.2f}%"
                  f" {i8_m:>9.2f}% {i8_f:>7.2f}% {i8_q:>6.2f}%"
                  f" {i8_m-fp_m:>+6.2f}% {i8_f-fp_f:>+5.2f}%")
    else:
        title = ('  PHASE 3 SUMMARY  —  PT float32 | fp32 argmax ONNX'
                 if has_pt else
                 '  PHASE 3 SUMMARY  —  fp32 argmax ONNX (server-matching)')
        print(title)
        print(f'{"="*110}')
        if has_pt:
            hdr = (f"{'Model':<22} {'PT mIoU':>8} {'PT F1':>6} {'PT Q':>6}"
                   f" {'FP32 mIoU':>10} {'FP32 F1':>8} {'FP32 Q':>7}"
                   f" {'Pr':>6} {'Re':>6} {'ΔmIoU':>7} {'ΔF1':>6}")
            print(hdr)
            print('-' * 110)
            for r in all_results:
                pt, fp = r['pt'], r['fp32']
                dm = (fp['miou'] - pt['miou']) * 100
                df = (fp['f1']   - pt['f1'])   * 100
                print(f"{r['model']:<22}"
                      f" {pt['miou']*100:>7.2f}% {pt['f1']*100:>5.2f}% {pt['q']*100:>5.2f}%"
                      f" {fp['miou']*100:>9.2f}% {fp['f1']*100:>7.2f}% {fp['q']*100:>6.2f}%"
                      f" {fp['precision']*100:>5.2f}% {fp['recall']*100:>5.2f}%"
                      f" {dm:>+6.2f}% {df:>+5.2f}%")
        else:
            hdr = f"{'Model':<22} {'mIoU':>8} {'F1':>7} {'Q':>7} {'Pr':>7} {'Re':>7}"
            print(hdr)
            print('-' * 70)
            for r in all_results:
                fp = r['fp32']
                print(f"{r['model']:<22}"
                      f" {fp['miou']*100:>7.2f}% {fp['f1']*100:>6.2f}% {fp['q']*100:>6.2f}%"
                      f" {fp['precision']*100:>6.2f}% {fp['recall']*100:>6.2f}%")

    print(f'{"="*110}')
    print('  F1: 70% GT coverage threshold, 100px min pred blob.')
    if has_int8:
        print('  INT8: onnxruntime static quantization (QUInt8, 50-image calibration) '
              '— approximate only, not Qualcomm SNPE.')
    print('  Server: Qualcomm SNPE INT8 on RVC4.')


if __name__ == '__main__':
    main()
