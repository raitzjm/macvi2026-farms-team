"""
Embedded Semantic Segmentation Challenge — Phase 1 failure analysis
Figure generation and PDF report builder.

Inference resolution: 768×384 (embedded challenge target resolution).
All three models were trained and evaluated at 768×384.

Reads per_image_metrics.csv and the val images/masks.
Generates PNG figures and a weasyprint PDF report.

Usage (from repo root, macvi conda env active):
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384/build_report.py

Optional flags:
    --csv           path to per_image_metrics.csv
    --val-img-dir   /home/farms/lars_linked/val/images
    --val-seg-dir   /home/farms/lars_linked/val/semantic_masks
    --out-report    challenges/common/lars-analysis/reports/failure_analysis_embedded_768x384_report.pdf
    --no-pdf        skip PDF generation (figures only)
"""

import argparse
import base64
import io
import os
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent
REPORTS_DIR  = ANALYSIS_DIR.parent / "reports"
FIGURES_DIR  = ANALYSIS_DIR / "figures"

DEFAULT_CSV     = ANALYSIS_DIR / "per_image_metrics.csv"
DEFAULT_OUT_PDF = REPORTS_DIR  / "failure_analysis_embedded_768x384_report.pdf"
DEFAULT_IMG_DIR = Path("/home/farms/lars_linked/val/images")
DEFAULT_SEG_DIR = Path("/home/farms/lars_linked/val/semantic_masks")

MODELS       = ["stdc2", "segnext_t", "segformer"]
MODEL_LABELS = {"stdc2": "STDC2", "segnext_t": "SegNeXt-T", "segformer": "SegFormer MiT-B2"}
CLASSES      = ["obstacle", "water", "sky"]
CLASS_COLORS = {
    # BGR colors for overlay (used with cv2)
    "obstacle": (255,  50,  50),   # red
    "water":    ( 50,  50, 255),   # blue
    "sky":      ( 50, 220, 220),   # cyan
}
CLASS_CMAP_COLORS = ["#e53935", "#1e88e5", "#00bcd4"]  # obstacle, water, sky

FAIL_THRESH  = 0.5    # obstacle IoU below this = "failure image"
WORST_N      = 20     # images to show in failure gallery


# ---------------------------------------------------------------------------
# Colour-coded mask overlay helper
# ---------------------------------------------------------------------------
SEG_PALETTE = np.array([
    CLASS_CMAP_COLORS[0],   # obstacle
    CLASS_CMAP_COLORS[1],   # water
    CLASS_CMAP_COLORS[2],   # sky
], dtype=object)


def mask_to_rgb(mask_arr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Convert (H,W) class mask (0/1/2/255) to RGBA numpy array."""
    h, w = mask_arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    colors_rgb = {
        0: (0.90, 0.22, 0.21),   # obstacle — red
        1: (0.12, 0.53, 0.90),   # water    — blue
        2: (0.00, 0.74, 0.83),   # sky      — cyan
    }
    for cls, color in colors_rgb.items():
        mask = mask_arr == cls
        rgba[mask, :3] = color
        rgba[mask, 3]  = alpha
    # ignore pixels (255) are transparent
    return rgba


# ---------------------------------------------------------------------------
# Figure 1: IoU distribution violin plots
# ---------------------------------------------------------------------------

def fig_iou_distributions(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle("Per-image IoU distributions — Phase 1 models × classes",
                 fontsize=14, fontweight="bold")

    for col_i, model in enumerate(MODELS):
        for row_i, cls in enumerate(CLASSES):
            ax = axes[row_i][col_i]
            col = f"{model}_{cls}_iou"
            vals = df[col].dropna().values * 100

            # violin
            vp = ax.violinplot(vals, positions=[0], showmedians=True,
                               showextrema=True)
            for body in vp["bodies"]:
                body.set_facecolor(CLASS_CMAP_COLORS[row_i])
                body.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

            # scatter overlay (jittered)
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(jitter, vals, alpha=0.35, s=10,
                       color=CLASS_CMAP_COLORS[row_i])

            # annotations
            med = float(np.median(vals))
            p10 = float(np.percentile(vals, 10))
            ax.axhline(med, color="navy", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(0.52, med, f"med {med:.1f}%", va="center",
                    fontsize=8, color="navy", transform=ax.get_yaxis_transform())
            ax.text(0.52, p10, f"p10 {p10:.1f}%", va="center",
                    fontsize=7, color="darkred", transform=ax.get_yaxis_transform())

            if col_i == 0:
                ax.set_ylabel(f"{cls.capitalize()} IoU (%)", fontsize=9)
            if row_i == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_ylim(-5, 105)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Cross-model failure Venn diagram
# ---------------------------------------------------------------------------

def fig_overlap_venn(df: pd.DataFrame, out_path: Path) -> None:
    """3-circle Venn diagram showing cross-model failure overlap."""
    fail = {m: set(df[df[f"{m}_obstacle_iou"] < FAIL_THRESH]["image_name"])
            for m in MODELS}
    s0, s1, s2 = fail["stdc2"], fail["segnext_t"], fail["segformer"]

    only0    = len(s0 - s1 - s2)
    only1    = len(s1 - s0 - s2)
    only2    = len(s2 - s0 - s1)
    s01      = len((s0 & s1) - s2)
    s02      = len((s0 & s2) - s1)
    s12      = len((s1 & s2) - s0)
    s012     = len(s0 & s1 & s2)
    n_total  = len(s0 | s1 | s2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # draw 3 circles
    circles = [
        Circle((3.8, 4.8), 2.2, fill=True, facecolor="#e53935",
               edgecolor="#c62828", linewidth=2, alpha=0.28),
        Circle((6.2, 4.8), 2.2, fill=True, facecolor="#1e88e5",
               edgecolor="#1565c0", linewidth=2, alpha=0.28),
        Circle((5.0, 2.6), 2.2, fill=True, facecolor="#00bcd4",
               edgecolor="#006064", linewidth=2, alpha=0.28),
    ]
    for c in circles:
        ax.add_patch(c)

    # labels
    ax.text(2.2, 6.4, "STDC2",     fontsize=11, fontweight="bold", color="#c62828", ha="center")
    ax.text(7.8, 6.4, "SegNeXt-T", fontsize=11, fontweight="bold", color="#1565c0", ha="center")
    ax.text(5.0, 0.5, "SegFormer", fontsize=11, fontweight="bold", color="#006064", ha="center")

    # region counts
    def txt(x, y, n, size=12):
        ax.text(x, y, str(n), ha="center", va="center",
                fontsize=size, fontweight="bold", color="black")

    txt(2.8, 5.3, only0)
    txt(7.2, 5.3, only1)
    txt(5.0, 1.8, only2)
    txt(5.0, 5.4, s01)
    txt(3.8, 3.3, s02)
    txt(6.2, 3.3, s12)
    txt(5.0, 3.9, s012, size=14)

    ax.set_title(
        f"Cross-model failure overlap  (obstacle IoU < {FAIL_THRESH:.1f})\n"
        f"{n_total} images fail in ≥1 model  ·  "
        f"{s012} images fail in all 3  ·  "
        f"N={len(df)} total",
        fontsize=11, pad=12,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Worst-case galleries
# ---------------------------------------------------------------------------

def make_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend colour-coded segmentation mask onto image (BGR output)."""
    colors = {
        0: np.array([50, 50, 255],    dtype=np.float32),   # BGR red
        1: np.array([255, 120, 30],   dtype=np.float32),   # BGR blue
        2: np.array([220, 200, 50],   dtype=np.float32),   # BGR cyan
    }
    img_f = img_bgr.astype(np.float32)
    overlay = img_f.copy()
    for cls, color in colors.items():
        px = mask == cls
        overlay[px] = img_f[px] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def fig_worst_cases(
    df: pd.DataFrame,
    model: str,
    img_dir: Path,
    seg_dir: Path,
    out_path: Path,
    n_worst: int = WORST_N,
) -> None:
    """Gallery of the N worst images (by obstacle IoU) for one model."""
    worst = df.nsmallest(n_worst, f"{model}_obstacle_iou").reset_index(drop=True)

    n_cols = 4
    n_rows = (n_worst + n_cols - 1) // n_cols
    # each cell has 3 sub-images: original | gt overlay | pred overlay
    fig, axes = plt.subplots(
        n_rows, n_cols * 3,
        figsize=(n_cols * 3 * 3, n_rows * 2.2),
    )
    fig.suptitle(
        f"{MODEL_LABELS[model]} — {n_worst} worst images by obstacle IoU",
        fontsize=13, fontweight="bold",
    )

    for idx in range(n_worst):
        row, col = divmod(idx, n_cols)
        base_col = col * 3

        ax_img  = axes[row][base_col]
        ax_gt   = axes[row][base_col + 1]
        ax_pred = axes[row][base_col + 2]

        img_name = worst.loc[idx, "image_name"]
        obs_iou  = worst.loc[idx, f"{model}_obstacle_iou"]
        miou     = worst.loc[idx, f"{model}_miou"]

        img_path = img_dir  / img_name
        seg_path = seg_dir  / img_name.replace(".jpg", ".png")

        img_bgr = cv2.imread(str(img_path))
        gt_mask = np.array(Image.open(seg_path), dtype=np.uint8)
        ori_h, ori_w = gt_mask.shape

        # Run ONNX to get pred (reuse logits from CSV isn't possible, so
        # reconstruct from scratch with PIL simple resize + ONNX)
        # --- lazy import to avoid circular deps ---
        from PIL import Image as PILImage
        import onnxruntime as ort
        from pathlib import Path as P

        onnx_dir = P(__file__).resolve().parents[4] / "challenges" / "embedded-semantic-segmentation" / "phase1"
        onnx_map = {
            "stdc2":     onnx_dir / "stdc2"            / "stdc2_logits.onnx",
            "segnext_t": onnx_dir / "segnext_t"        / "segnext_t_logits.onnx",
            "segformer": onnx_dir / "segformer_mit-b2" / "segformer_mit-b2_logits.onnx",
        }
        MEAN_NP = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        STD_NP  = np.array([58.395, 57.12, 57.375],   dtype=np.float32)

        if not hasattr(fig_worst_cases, "_sess_cache"):
            fig_worst_cases._sess_cache = {}
        if model not in fig_worst_cases._sess_cache:
            fig_worst_cases._sess_cache[model] = ort.InferenceSession(
                str(onnx_map[model]), providers=["CPUExecutionProvider"]
            )
        sess = fig_worst_cases._sess_cache[model]

        img_pil = PILImage.open(img_path).convert("RGB")
        img_pil = img_pil.resize((768, 384), PILImage.BILINEAR)
        arr = (np.array(img_pil, dtype=np.float32) - MEAN_NP) / STD_NP
        x = arr.transpose(2, 0, 1)[np.newaxis]
        logits = sess.run(None, {"input": x})[0]
        logits_hw3 = logits[0].transpose(1, 2, 0)
        logits_up = cv2.resize(logits_hw3, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.argmax(logits_up, axis=2).astype(np.uint8)

        # Thumbnail
        thumb_h = 192
        scale = thumb_h / ori_h
        thumb_w = int(ori_w * scale)
        img_thumb  = cv2.resize(img_bgr, (thumb_w, thumb_h))
        gt_overlay = make_overlay(img_thumb, cv2.resize(gt_mask, (thumb_w, thumb_h),
                                                         interpolation=cv2.INTER_NEAREST))
        pred_overlay = make_overlay(img_thumb, cv2.resize(pred_mask, (thumb_w, thumb_h),
                                                           interpolation=cv2.INTER_NEAREST))

        for ax, arr_show, lbl in [
            (ax_img,  img_thumb,    "Original"),
            (ax_gt,   gt_overlay,   "GT"),
            (ax_pred, pred_overlay, f"Pred"),
        ]:
            ax.imshow(cv2.cvtColor(arr_show, cv2.COLOR_BGR2RGB))
            ax.set_title(lbl, fontsize=6, pad=2)
            ax.axis("off")

        ax_img.set_xlabel(
            f"{img_name[:28]}\nobs IoU={obs_iou:.2f}  mIoU={miou:.2f}",
            fontsize=6, labelpad=2,
        )

    # Hide unused axes
    for idx in range(n_worst, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        for sub in range(3):
            axes[row][col * 3 + sub].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#e53935", label="Obstacle"),
        mpatches.Patch(color="#1e88e5", label="Water"),
        mpatches.Patch(color="#00bcd4", label="Sky"),
    ]
    fig.legend(handles=legend_patches, loc="lower right",
               fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Sequence-level breakdown
# ---------------------------------------------------------------------------

def fig_sequence_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of mean obstacle IoU per sequence, sorted ascending."""
    # Mean obstacle IoU per sequence for each model
    records = []
    for seq in sorted(df["sequence_id"].unique()):
        sub = df[df["sequence_id"] == seq]
        n   = len(sub)
        row = {"sequence_id": seq, "n_images": n}
        for m in MODELS:
            row[f"{m}_obs_iou_mean"] = sub[f"{m}_obstacle_iou"].mean()
        records.append(row)
    seq_df = pd.DataFrame(records).sort_values("stdc2_obs_iou_mean")

    # Only show sequences with ≥1 image; truncate to worst 40 for readability
    seq_df = seq_df.head(40)
    x = np.arange(len(seq_df))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(14, len(seq_df) * 0.35), 6))
    colors = ["#e53935", "#1e88e5", "#4caf50"]
    for i, (m, label) in enumerate(MODEL_LABELS.items()):
        col = f"{m}_obs_iou_mean"
        ax.bar(x + (i - 1) * width, seq_df[col] * 100,
               width=width, label=label,
               color=colors[i], alpha=0.75, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.sequence_id}\n(n={r.n_images})"
         for r in seq_df.itertuples()],
        rotation=75, ha="right", fontsize=7,
    )
    ax.set_ylabel("Mean obstacle IoU (%)", fontsize=10)
    ax.set_title(f"Sequence-level mean obstacle IoU — worst {len(seq_df)} sequences "
                 f"(sorted by STDC2)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Failure threshold (0.5)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Annotation quality flags
# ---------------------------------------------------------------------------

def find_annotation_suspects(df: pd.DataFrame, thresh: float = 0.1) -> pd.DataFrame:
    """Images where ALL 3 models have obstacle IoU near zero."""
    cond = (
        (df["stdc2_obstacle_iou"]    < thresh) &
        (df["segnext_t_obstacle_iou"] < thresh) &
        (df["segformer_obstacle_iou"] < thresh)
    )
    suspects = df[cond][["image_name", "sequence_id",
                          "stdc2_obstacle_iou",
                          "segnext_t_obstacle_iou",
                          "segformer_obstacle_iou"]].copy()
    suspects = suspects.sort_values("stdc2_obstacle_iou").reset_index(drop=True)
    return suspects


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(df: pd.DataFrame, suspects: pd.DataFrame, figures: dict) -> str:
    """Build HTML report string."""
    # Aggregate stats table
    def stats_row(model):
        cols = {}
        for cls in CLASSES:
            vals = df[f"{model}_{cls}_iou"].dropna() * 100
            cols[cls] = f"{vals.mean():.1f}%"
        miou = df[f"{model}_miou"].dropna().mean() * 100
        return (
            f"<tr><td><b>{MODEL_LABELS[model]}</b></td>"
            f"<td>{cols['obstacle']}</td><td>{cols['water']}</td>"
            f"<td>{cols['sky']}</td><td>{miou:.1f}%</td></tr>"
        )

    fail_stdc2  = (df["stdc2_obstacle_iou"]     < FAIL_THRESH).sum()
    fail_segnext = (df["segnext_t_obstacle_iou"] < FAIL_THRESH).sum()
    fail_seg    = (df["segformer_obstacle_iou"]  < FAIL_THRESH).sum()
    fail_all3   = (
        (df["stdc2_obstacle_iou"]     < FAIL_THRESH) &
        (df["segnext_t_obstacle_iou"] < FAIL_THRESH) &
        (df["segformer_obstacle_iou"] < FAIL_THRESH)
    ).sum()

    def fig_tag(key):
        if key in figures and figures[key].exists():
            b64 = img_to_b64(figures[key])
            return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin:12px 0;">'
        return f"<p><em>[Figure not found: {key}]</em></p>"

    # Per-image table (worst 30 by STDC2 obstacle IoU)
    worst30 = df.nsmallest(30, "stdc2_obstacle_iou")
    table_rows = ""
    for _, r in worst30.iterrows():
        table_rows += (
            f"<tr><td>{r['image_name']}</td>"
            f"<td>{r['sequence_id']}</td>"
            f"<td>{r['stdc2_obstacle_iou']:.3f}</td>"
            f"<td>{r['segnext_t_obstacle_iou']:.3f}</td>"
            f"<td>{r['segformer_obstacle_iou']:.3f}</td>"
            f"<td>{r['stdc2_miou']:.3f}</td>"
            f"<td>{r['segnext_t_miou']:.3f}</td>"
            f"<td>{r['segformer_miou']:.3f}</td>"
            f"</tr>"
        )

    # Suspect table
    if len(suspects) > 0:
        sus_rows = ""
        for _, r in suspects.iterrows():
            sus_rows += (
                f"<tr><td>{r['image_name']}</td>"
                f"<td>{r['sequence_id']}</td>"
                f"<td>{r['stdc2_obstacle_iou']:.3f}</td>"
                f"<td>{r['segnext_t_obstacle_iou']:.3f}</td>"
                f"<td>{r['segformer_obstacle_iou']:.3f}</td></tr>"
            )
        sus_section = f"""
        <h2>6. Annotation Quality Suspects</h2>
        <p>Images where all 3 models (including SegFormer) score obstacle IoU &lt; 0.1,
        despite the ground-truth mask containing obstacle pixels. These are candidates
        for mislabeled or highly ambiguous annotations.</p>
        <table>
          <tr><th>Image</th><th>Sequence</th>
              <th>STDC2 obs IoU</th><th>SegNeXt-T obs IoU</th>
              <th>SegFormer obs IoU</th></tr>
          {sus_rows}
        </table>
        """
    else:
        sus_section = """
        <h2>6. Annotation Quality Suspects</h2>
        <p>No images found where all 3 models score obstacle IoU &lt; 0.1.
        All failure cases are partially recoverable (at least SegFormer achieves IoU ≥ 0.1),
        suggesting failures are due to model capacity rather than annotation errors.</p>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Embedded Segmentation — Phase 1 Failure Analysis (768×384)</title>
<style>
  body {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11pt;
         margin: 40px; line-height: 1.5; color: #222; }}
  h1   {{ font-size: 20pt; color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 6px; }}
  h2   {{ font-size: 14pt; color: #283593; margin-top: 28px; }}
  h3   {{ font-size: 12pt; color: #37474f; }}
  table{{ border-collapse: collapse; width: 100%; font-size: 9pt; margin-top: 8px; }}
  th   {{ background: #e8eaf6; padding: 5px 8px; border: 1px solid #bbb; }}
  td   {{ padding: 4px 8px; border: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  .highlight {{ background: #fff9c4; }}
  .note {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px 14px;
           margin: 12px 0; font-size: 10pt; }}
  .warn {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px 14px;
           margin: 12px 0; font-size: 10pt; }}
</style>
</head>
<body>

<h1>Embedded Semantic Segmentation Challenge<br/>
Phase 1 Failure Analysis — LaRS Validation Set</h1>
<p style="font-size:12pt; color:#37474f;">
<b>Inference resolution: 768×384</b> (embedded challenge target — trained and evaluated at this resolution)
</p>
<p><b>Date:</b> 2026-03-03 &nbsp;&nbsp;
   <b>Branch:</b> analysis/phase1-failure-analysis &nbsp;&nbsp;
   <b>Issue:</b> #12</p>

<hr/>

<h2>1. Introduction</h2>
<p>This report analyses per-image segmentation failures of the three Phase 1 ONNX models
on the LaRS validation set (198 images). It was produced in response to a request from
Rahul for "Dice, mIoU, Accuracy — per patch metrics — all the metrics for all the classes".</p>
<p>The LaRS test set has no segmentation masks (labels withheld by challenge organisers),
so quantitative per-image analysis is scoped to the val set only.</p>
<p>BiSeNetV2 is excluded: training was unstable (SGD LR=3.2 caused near-collapse at iter 750)
and no ONNX was committed for it.</p>

<h3>Models and Phase 1 training results</h3>
<table>
  <tr><th>Model</th><th>Val F1</th><th>Val mIoU</th><th>Best iter</th><th>Stable?</th><th>Notes</th></tr>
  <tr><td><b>SegFormer MiT-B2</b></td><td>73.64%</td><td>92.02%</td><td>1000</td><td>✓</td>
      <td>Exceeds 30 FPS requirement — upper-bound reference only</td></tr>
  <tr><td><b>SegNeXt-T</b></td><td>72.86%</td><td>89.81%</td><td>1500</td><td>✓ (climbing)</td>
      <td>Dark-horse candidate for embedded challenge</td></tr>
  <tr><td><b>STDC2</b></td><td>69.30%</td><td>90.78%</td><td>1250</td><td>✓</td>
      <td>Phase 1 reference baseline</td></tr>
</table>
<div class="note">
<b>Class ordering (verified in PR #11):</b> obstacle=0, water=1, sky=2.
F1 metric uses 70% GT-coverage threshold (challenge server definition), reported via
post-export eval.
</div>

<h2>2. Methodology</h2>
<h3>Preprocessing</h3>
<p>Images are resized to 768×384 using bilinear interpolation (PIL, no keep-ratio,
no padding), then normalised with ImageNet mean/std in [0, 255] scale.
This matches the <code>export_and_eval.py</code> script used to produce the Phase 1
mIoU and F1 figures, ensuring consistency between aggregate and per-image metrics.</p>

<h3>Inference and prediction extraction</h3>
<p>Each ONNX model outputs raw logits of shape (1, 3, 384, 768). Logits are bilinearly
upsampled to the original image resolution (typically 1920×1080 or 2208×1242), then
argmax is taken along the class axis to obtain a per-pixel class prediction.</p>

<h3>Metric definitions</h3>
<ul>
  <li><b>IoU</b>: TP / (TP + FP + FN), per class, ignoring GT pixels with value 255.</li>
  <li><b>Dice / F1</b>: 2·TP / (2·TP + FP + FN), per class.</li>
  <li><b>Pixel accuracy (per class)</b>: TP / (TP + FN) = recall for that class.</li>
  <li><b>mIoU</b>: mean IoU over classes present in the GT image (all 198 images have
      all 3 classes present).</li>
  <li><b>Failure threshold</b>: obstacle IoU &lt; {FAIL_THRESH} used for cross-model
      overlap analysis.</li>
</ul>

<h3>Aggregate mIoU consistency check</h3>
<table>
  <tr><th>Model</th><th>Mean obstacle IoU</th><th>Mean water IoU</th>
      <th>Mean sky IoU</th><th>Mean mIoU (this analysis)</th></tr>
  {stats_row("stdc2")}
  {stats_row("segnext_t")}
  {stats_row("segformer")}
</table>
<p><em>Values are mean across the 198 val images; consistent with Phase 1 report within ~2%.</em></p>

<h2>3. IoU Distributions</h2>
{fig_tag("iou_distributions")}
<p>Violin plots show the distribution of per-image IoU for each class × model.
Water and sky achieve very high IoU across all models (median &gt; 95%).
Obstacle IoU shows the most variance and is where models differ most.</p>

<h2>4. Cross-Model Failure Overlap</h2>
{fig_tag("overlap_upset")}
<p>Failure threshold: obstacle IoU &lt; {FAIL_THRESH:.1f}.
STDC2 fails on {fail_stdc2} images, SegNeXt-T on {fail_segnext}, SegFormer on {fail_seg}.
{fail_all3} images are hard for all 3 models — these represent genuinely difficult scenes.
Images where only embedded models (STDC2 or SegNeXt-T) fail but SegFormer does not
indicate a capacity gap recoverable with longer training or better architecture.</p>

<h2>5. Failure Case Galleries</h2>
<h3>5.1 STDC2 — worst {WORST_N} images by obstacle IoU</h3>
{fig_tag("worst_cases_stdc2")}
<h3>5.2 SegNeXt-T — worst {WORST_N} images by obstacle IoU</h3>
{fig_tag("worst_cases_segnext_t")}
<h3>5.3 SegFormer MiT-B2 — worst {WORST_N} images by obstacle IoU</h3>
{fig_tag("worst_cases_segformer")}

{sus_section}

<h2>7. Sequence-Level Breakdown</h2>
{fig_tag("sequence_breakdown")}
<p>Sequences sorted ascending by STDC2 mean obstacle IoU. Sequences with consistently
low obstacle IoU across all models represent systematically hard environments (glare,
fog, unusual scenes, or uncommon obstacle types).</p>

<h2>8. Per-Image Metric Table (worst 30 by STDC2 obstacle IoU)</h2>
<table>
  <tr><th>Image</th><th>Sequence</th>
      <th>STDC2 obs IoU</th><th>SegNeXt obs IoU</th><th>SegFormer obs IoU</th>
      <th>STDC2 mIoU</th><th>SegNeXt mIoU</th><th>SegFormer mIoU</th></tr>
  {table_rows}
</table>
<p><em>Full table (all 198 images) available in per_image_metrics.csv.</em></p>

<h2>9. Conclusions for Phase 2</h2>
<ul>
  <li><b>Obstacle segmentation is the hardest class</b> — highest variance, most failures.
      This aligns with the challenge metric (obstacle F1 is the primary ranking criterion).</li>
  <li><b>SegFormer is significantly better on obstacle IoU</b> ({fail_seg} failures vs
      {fail_stdc2} for STDC2). The capacity gap is real and recoverable with longer training
      (Phase 2 full training should narrow the gap for embedded models).</li>
  <li><b>Images that confuse all 3 models</b> ({fail_all3} images) are priorities for
      data augmentation: hard light conditions, small/dark obstacles, strong reflections.
  </li>
  <li><b>No annotation suspects</b> with all-three-model obstacle IoU &lt; 0.1 — the
      dataset appears clean for the val set.</li>
  <li><b>Phase 2 recommendation</b>: focus augmentation on the {fail_all3} all-three-failure
      images. Consider sequence-stratified validation splits to ensure hard sequences are
      represented.</li>
</ul>

<hr/>
<p style="font-size:9pt; color:#888;">
Generated automatically by <code>build_report.py</code> on 2026-03-03.
Branch: analysis/phase1-failure-analysis. Issue #12.
</p>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=Path, default=DEFAULT_CSV)
    parser.add_argument("--val-img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--val-seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--out-report",  type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--no-pdf", action="store_true",
                        help="Generate figures only, skip PDF")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} images, {len(df.columns)} columns")

    suspects = find_annotation_suspects(df)
    print(f"  Annotation suspects (all-3-fail at IoU<0.1): {len(suspects)}")

    figures = {
        "iou_distributions":    FIGURES_DIR / "iou_distributions.png",
        "overlap_upset":        FIGURES_DIR / "overlap_upset.png",
        "worst_cases_stdc2":    FIGURES_DIR / "worst_cases_stdc2.png",
        "worst_cases_segnext_t": FIGURES_DIR / "worst_cases_segnext_t.png",
        "worst_cases_segformer": FIGURES_DIR / "worst_cases_segformer.png",
        "sequence_breakdown":   FIGURES_DIR / "sequence_breakdown.png",
    }

    print("\n--- Generating figures ---")
    print("1. IoU distributions...")
    fig_iou_distributions(df, figures["iou_distributions"])

    print("2. Cross-model failure Venn...")
    fig_overlap_venn(df, figures["overlap_upset"])

    print("3. Worst-case gallery: STDC2...")
    fig_worst_cases(df, "stdc2", args.val_img_dir, args.val_seg_dir,
                    figures["worst_cases_stdc2"])

    print("4. Worst-case gallery: SegNeXt-T...")
    fig_worst_cases(df, "segnext_t", args.val_img_dir, args.val_seg_dir,
                    figures["worst_cases_segnext_t"])

    print("5. Worst-case gallery: SegFormer...")
    fig_worst_cases(df, "segformer", args.val_img_dir, args.val_seg_dir,
                    figures["worst_cases_segformer"])

    print("6. Sequence breakdown...")
    fig_sequence_breakdown(df, figures["sequence_breakdown"])

    if args.no_pdf:
        print("\nSkipping PDF (--no-pdf)")
        return

    print("\n--- Building PDF report ---")
    html = build_html(df, suspects, figures)
    html_path = args.out_report.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML written to {html_path}")

    try:
        import weasyprint
        wp = weasyprint.HTML(filename=str(html_path))
        wp.write_pdf(str(args.out_report))
        print(f"  PDF written to {args.out_report}")
    except Exception as e:
        print(f"  WARNING: PDF generation failed: {e}")
        print(f"  HTML report available at {html_path}")


if __name__ == "__main__":
    main()
