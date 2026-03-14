"""
Phase 2/2.5 failure analysis — inference and per-image metric computation.

Runs 9 Phase 2/2.5 argmax ONNX models on the 198 LaRS val images.
Preprocessing matches the challenge server (keep-ratio bilinear resize →
center reflect-pad to 384×768 → ImageNet normalize). Argmax ONNX output
is unpadded, then nearest-neighbour resized to original GT resolution.

Before running, ensure segnext_t_uperhead ONNX is exported:
    conda run --no-capture-output -n macvi2 \\
        python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/export_onnx.py

Outputs
-------
per_image_metrics.csv    per-image IoU / Dice / pixel-accuracy for all models
logs/inference.log       progress + per-image timing

Usage (from repo root, macvi conda env active):
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/infer_and_eval.py

Optional flags:
    --val-img-dir   /home/farms/data/datasets/lars_dataset/img_dir/val/images
    --val-seg-dir   /home/farms/data/datasets/lars_dataset/ann_dir/val/semantic_masks
    --out-csv       <this_dir>/per_image_metrics.csv
    --models        segformer_p2 pidnets_p2 ...  (default: all 9)
    --no-cuda       force CPU inference even if CUDA is available
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = Path(__file__).resolve().parent

# All pre-exported ONNX from /home/farms/data/onnx_exports/
# segnext_t_uperhead is exported by export_onnx.py
ONNX_EXPORTS = Path("/home/farms/data/onnx_exports")

ONNX_PATHS = {
    "segformer_p2":          ONNX_EXPORTS / "phase2"   / "segformer_mit-b2" / "segformer_mit-b2_argmax.onnx",
    "pidnets_p2":            ONNX_EXPORTS / "phase2"   / "pidnets"          / "pidnets_argmax.onnx",
    "pidnets_p25":           ONNX_EXPORTS / "phase2_5" / "phase2_5" / "pidnets" / "pidnets_argmax.onnx",
    "segnext_t_p2":          ONNX_EXPORTS / "phase2"   / "segnext_t"        / "segnext_t_argmax.onnx",
    "segnext_t_uperhead_p2": ANALYSIS_DIR / "onnx_exports" / "segnext_t_uperhead" / "segnext_t_uperhead_argmax.onnx",
    "segnext_s_p2":          ONNX_EXPORTS / "phase2"   / "segnext_s"        / "segnext_s_argmax.onnx",
    "stdc2_p2":              ONNX_EXPORTS / "phase2"   / "stdc2"            / "stdc2_argmax.onnx",
    "stdc2_p25":             ONNX_EXPORTS / "phase2_5" / "phase2_5" / "stdc2"  / "stdc2_argmax.onnx",
    "bisenetv2_p2":          ONNX_EXPORTS / "phase2"   / "bisenetv2"        / "bisenetv2_argmax.onnx",
}

ALL_MODELS = list(ONNX_PATHS.keys())

DEFAULT_IMG_DIR = Path("/home/farms/data/datasets/lars_dataset/img_dir/val/images")
DEFAULT_SEG_DIR = Path("/home/farms/data/datasets/lars_dataset/ann_dir/val/semantic_masks")
DEFAULT_CSV     = ANALYSIS_DIR / "per_image_metrics.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_H, MODEL_W = 384, 768
CLASSES          = ["obstacle", "water", "sky"]
N_CLASSES        = 3
IGNORE_INDEX     = 255

# ImageNet mean/std in [0, 255] scale
MEAN_NP = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD_NP  = np.array([58.395,   57.12,   57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing — Phase 2 server-matching
# ---------------------------------------------------------------------------

def server_preprocess(img_path: Path):
    """
    Keep-ratio bilinear resize → center reflect-pad to 384×768 → ImageNet normalize.
    Matches the challenge server preprocessing exactly.

    Returns
    -------
    x_np : np.ndarray  shape (1, 3, 384, 768), float32 — ONNX session input
    info : dict with pad_top, pad_left, new_h, new_w, ori_h, ori_w
    """
    img = Image.open(img_path).convert("RGB")
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

    mode = "reflect"
    if pad_top >= new_h or pad_bottom >= new_h or pad_left >= new_w or pad_right >= new_w:
        mode = "symmetric"
    arr = np.pad(arr, [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode=mode)
    arr = (arr - MEAN_NP) / STD_NP

    x_np = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    info = dict(pad_top=pad_top, pad_left=pad_left,
                new_h=new_h, new_w=new_w, ori_h=ori_h, ori_w=ori_w)
    return x_np, info


def unpad_argmax(argmax_out: np.ndarray, info: dict) -> np.ndarray:
    """
    Crop reflect-pad from (1, 1, 384, 768) int64 argmax output, then
    nearest-neighbour resize to original GT resolution.

    Returns uint8 (ori_H, ori_W) prediction.
    """
    t, l = info["pad_top"], info["pad_left"]
    h, w = info["new_h"], info["new_w"]
    pred = argmax_out[0, 0, t:t + h, l:l + w].astype(np.uint8)
    pred_pil = Image.fromarray(pred)
    return np.array(pred_pil.resize((info["ori_w"], info["ori_h"]), Image.NEAREST),
                    dtype=np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Per-class IoU, Dice, and pixel accuracy at original GT resolution.

    Parameters
    ----------
    pred : (H, W) uint8 — predicted class indices at original resolution
    gt   : (H, W) uint8 — ground-truth semantic mask (255 = ignore)

    Returns
    -------
    dict with keys:
        {class}_iou, {class}_dice, {class}_acc  for each class
        miou  — mean IoU over classes present in GT
    """
    valid  = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v   = gt[valid]

    results      = {}
    present_ious = []

    for c, name in enumerate(CLASSES):
        tp = int(np.sum((pred_v == c) & (gt_v == c)))
        fp = int(np.sum((pred_v == c) & (gt_v != c)))
        fn = int(np.sum((pred_v != c) & (gt_v == c)))

        denom_iou  = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        denom_acc  = tp + fn

        iou  = tp / denom_iou  if denom_iou  > 0 else float("nan")
        dice = tp * 2 / denom_dice if denom_dice > 0 else float("nan")
        acc  = tp / denom_acc  if denom_acc  > 0 else float("nan")

        results[f"{name}_iou"]  = iou
        results[f"{name}_dice"] = dice
        results[f"{name}_acc"]  = acc

        if denom_acc > 0 and not np.isnan(iou):
            present_ious.append(iou)

    results["miou"] = float(np.mean(present_ious)) if present_ious else float("nan")
    return results


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    onnx_path: Path,
    img_dir: Path,
    seg_dir: Path,
    log: logging.Logger,
    providers: list,
) -> list[dict]:
    """Run a single argmax ONNX model on all val images."""
    log.info(f"Loading {model_name} from {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active_ep = sess.get_providers()[0]
    log.info(f"  Execution provider: {active_ep}")

    img_names = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    log.info(f"  {len(img_names)} images found")

    rows    = []
    t_start = time.time()

    for i, img_name in enumerate(img_names):
        img_path = img_dir / img_name
        seg_name = img_name.replace(".jpg", ".png")
        seg_path = seg_dir / seg_name

        if not seg_path.exists():
            log.warning(f"  Missing GT mask for {img_name}, skipping")
            continue

        t0 = time.time()

        x_np, info = server_preprocess(img_path)
        inp_dtype  = sess.get_inputs()[0].type
        if "float16" in inp_dtype:
            x_np = x_np.astype(np.float16)
        argmax_out = sess.run(None, {"input": x_np})[0]     # (1,1,384,768) int64

        gt   = np.array(Image.open(seg_path), dtype=np.uint8)
        pred = unpad_argmax(argmax_out, info)                # (ori_H, ori_W) uint8

        m = compute_metrics(pred, gt)

        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0:
            log.info(
                f"  [{i+1:3d}/{len(img_names)}] {img_name}  "
                f"mIoU={m['miou']:.3f}  t={elapsed*1000:.0f}ms"
            )

        row = {f"{model_name}_{k}": v for k, v in m.items()}
        rows.append({"image_name": img_name, **row})

    total = time.time() - t_start
    log.info(f"  {model_name} done — {len(rows)} images in {total:.1f}s "
             f"({total/len(rows)*1000:.0f}ms/img)")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2/2.5 failure analysis inference")
    parser.add_argument("--val-img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--val-seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--out-csv",     type=Path, default=DEFAULT_CSV)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force CPU inference even if CUDA is available")
    args = parser.parse_args()

    # Logging
    log_path = ANALYSIS_DIR / "logs" / "inference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info("=== Phase 2/2.5 failure analysis — inference start ===")
    log.info(f"Models: {args.models}")
    log.info(f"Val images: {args.val_img_dir}")
    log.info(f"Val masks:  {args.val_seg_dir}")
    log.info(f"Output CSV: {args.out_csv}")

    # ORT execution providers
    if args.no_cuda:
        providers = ["CPUExecutionProvider"]
    else:
        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available
                     else ["CPUExecutionProvider"])
    log.info(f"ORT providers: {providers}")

    # Validate ONNX paths
    missing = []
    for name in args.models:
        if not ONNX_PATHS[name].exists():
            missing.append((name, ONNX_PATHS[name]))
    if missing:
        for name, path in missing:
            log.error(f"ONNX not found for {name}: {path}")
            if name == "segnext_t_uperhead_p2":
                log.error("  -> Run export_onnx.py first (requires macvi2 env)")
        sys.exit(1)

    if not args.val_img_dir.exists():
        log.error(f"Val image dir not found: {args.val_img_dir}")
        sys.exit(1)

    # Run each model
    all_rows_by_name: dict[str, dict] = {}
    for name in args.models:
        rows = run_model(name, ONNX_PATHS[name], args.val_img_dir,
                         args.val_seg_dir, log, providers)
        for r in rows:
            img_name = r["image_name"]
            if img_name not in all_rows_by_name:
                all_rows_by_name[img_name] = {"image_name": img_name}
            all_rows_by_name[img_name].update(r)

    # Assemble DataFrame
    df = pd.DataFrame(list(all_rows_by_name.values()))
    df = df.sort_values("image_name").reset_index(drop=True)
    df.insert(1, "sequence_id",
              df["image_name"].apply(
                  lambda n: "_".join(n.replace(".jpg", "").split("_")[:-1])))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"Saved {len(df)} rows to {args.out_csv}")

    # Aggregate sanity check
    log.info("\n=== Aggregate mIoU check ===")
    for name in args.models:
        col = f"{name}_miou"
        if col in df.columns:
            agg = df[col].mean()
            log.info(f"  {name:<30s}  mean mIoU = {agg*100:.2f}%")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
