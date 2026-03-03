"""
Phase 1 failure analysis — inference and per-image metric computation.

Runs the three Phase 1 ONNX logits models on the 198 LaRS val images.
Preprocessing matches export_and_eval.py (simple resize, no keep_ratio) so
that aggregate metrics are consistent with the reported Phase 1 results.

Outputs
-------
per_image_metrics.csv    per-image IoU / Dice / pixel-accuracy for all models
logs/inference.log       progress + per-image timing

Usage (from repo root, macvi conda env active):
    python challenges/common/lars-analysis/failure_analysis/infer_and_eval.py

Optional flags:
    --val-img-dir   /home/farms/lars_linked/val/images
    --val-seg-dir   /home/farms/lars_linked/val/semantic_masks
    --out-csv       challenges/common/lars-analysis/failure_analysis/per_image_metrics.csv
    --models        stdc2 segnext_t segformer  (default: all three)
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
REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE1_DIR = REPO_ROOT / "challenges" / "embedded-semantic-segmentation" / "phase1"
OUT_DIR = Path(__file__).resolve().parent

ONNX_PATHS = {
    "stdc2":       PHASE1_DIR / "stdc2"              / "stdc2_logits.onnx",
    "segnext_t":   PHASE1_DIR / "segnext_t"          / "segnext_t_logits.onnx",
    "segformer":   PHASE1_DIR / "segformer_mit-b2"   / "segformer_mit-b2_logits.onnx",
}

DEFAULT_IMG_DIR = Path("/home/farms/lars_linked/val/images")
DEFAULT_SEG_DIR = Path("/home/farms/lars_linked/val/semantic_masks")
DEFAULT_CSV     = OUT_DIR / "per_image_metrics.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_H, MODEL_W = 384, 768
CLASSES = ["obstacle", "water", "sky"]
N_CLASSES = 3
IGNORE_INDEX = 255

# ImageNet mean/std in [0, 255] scale (matches export_and_eval.py exactly)
MEAN_NP = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_NP  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_path: Path) -> np.ndarray:
    """
    Simple resize to (MODEL_W, MODEL_H) with bilinear interpolation.
    Matches the preprocessing used in Phase 1 export_and_eval.py so that
    per-image metrics are consistent with the reported aggregate mIoU.

    Returns
    -------
    x : np.ndarray  shape (1, 3, 384, 768), float32
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) - MEAN_NP) / STD_NP   # HWC
    return arr.transpose(2, 0, 1)[np.newaxis]                      # NCHW


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute per-class IoU, Dice, and pixel accuracy at original GT resolution.

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
    valid = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v   = gt[valid]

    results = {}
    present_ious = []

    for c, name in enumerate(CLASSES):
        tp = int(np.sum((pred_v == c) & (gt_v == c)))
        fp = int(np.sum((pred_v == c) & (gt_v != c)))
        fn = int(np.sum((pred_v != c) & (gt_v == c)))

        denom_iou  = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        denom_acc  = tp + fn                             # GT pixels of class c

        iou  = tp / denom_iou  if denom_iou  > 0 else float("nan")
        dice = tp * 2 / denom_dice if denom_dice > 0 else float("nan")
        acc  = tp / denom_acc  if denom_acc  > 0 else float("nan")

        results[f"{name}_iou"]  = iou
        results[f"{name}_dice"] = dice
        results[f"{name}_acc"]  = acc

        if denom_acc > 0 and not np.isnan(iou):   # class present in GT
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
) -> list[dict]:
    """
    Run a single ONNX model on all val images.

    Returns a list of row-dicts (one per image), with keys prefixed by model_name.
    """
    log.info(f"Loading {model_name} from {onnx_path}")
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    img_names = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    log.info(f"  {len(img_names)} images found")

    rows = []
    t_start = time.time()

    for i, img_name in enumerate(img_names):
        img_path  = img_dir  / img_name
        seg_name  = img_name.replace(".jpg", ".png")
        seg_path  = seg_dir / seg_name

        if not seg_path.exists():
            log.warning(f"  Missing GT mask for {img_name}, skipping")
            continue

        t0 = time.time()

        # Pre-process
        x = preprocess(img_path)                               # (1,3,384,768)

        # ONNX inference
        logits = sess.run(None, {"input": x})[0]               # (1,3,384,768)

        # Load GT at original resolution
        gt = np.array(Image.open(seg_path), dtype=np.uint8)    # (ori_H, ori_W)
        ori_h, ori_w = gt.shape

        # Upsample logits to original resolution, then argmax
        logits_hw3 = logits[0].transpose(1, 2, 0)             # (384, 768, 3)
        logits_up  = cv2.resize(
            logits_hw3, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR
        )                                                       # (ori_H, ori_W, 3)
        pred = np.argmax(logits_up, axis=2).astype(np.uint8)   # (ori_H, ori_W)

        # Metrics
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
    parser = argparse.ArgumentParser(description="Phase 1 failure analysis inference")
    parser.add_argument("--val-img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--val-seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--out-csv",     type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--models", nargs="+",
        default=["stdc2", "segnext_t", "segformer"],
        choices=["stdc2", "segnext_t", "segformer"],
    )
    args = parser.parse_args()

    # Logging
    log_path = OUT_DIR / "logs" / "inference.log"
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
    log.info("=== Phase 1 failure analysis — inference start ===")
    log.info(f"Models: {args.models}")
    log.info(f"Val images: {args.val_img_dir}")
    log.info(f"Val masks:  {args.val_seg_dir}")
    log.info(f"Output CSV: {args.out_csv}")

    # Validate paths
    for name in args.models:
        if not ONNX_PATHS[name].exists():
            log.error(f"ONNX not found: {ONNX_PATHS[name]}")
            log.error("Run 'git lfs pull' to fetch the ONNX models.")
            sys.exit(1)
    if not args.val_img_dir.exists():
        log.error(f"Val image dir not found: {args.val_img_dir}")
        log.error("Rebuild the symlink tree (see challenge CLAUDE.md).")
        sys.exit(1)

    # Run each model
    all_rows_by_name: dict[str, dict] = {}
    for name in args.models:
        rows = run_model(name, ONNX_PATHS[name], args.val_img_dir, args.val_seg_dir, log)
        for r in rows:
            img_name = r["image_name"]
            if img_name not in all_rows_by_name:
                all_rows_by_name[img_name] = {"image_name": img_name}
            all_rows_by_name[img_name].update(r)

    # Assemble DataFrame
    df = pd.DataFrame(list(all_rows_by_name.values()))
    df = df.sort_values("image_name").reset_index(drop=True)

    # Add sequence_id column
    df.insert(1, "sequence_id",
              df["image_name"].apply(lambda n: "_".join(n.replace(".jpg", "").split("_")[:-1])))

    # Save
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"Saved {len(df)} rows to {args.out_csv}")

    # Quick aggregate sanity check
    log.info("\n=== Aggregate mIoU check (should match Phase 1 report) ===")
    for name in args.models:
        col = f"{name}_miou"
        if col in df.columns:
            agg = df[col].mean()
            log.info(f"  {name:>15s}  mean mIoU = {agg*100:.2f}%")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
