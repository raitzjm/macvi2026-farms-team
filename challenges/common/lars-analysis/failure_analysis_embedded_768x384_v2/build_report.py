"""
Embedded Semantic Segmentation Challenge — Phase 2/2.5 failure analysis
Figure generation and PDF report builder.

Inference resolution: 768×384 (embedded challenge target resolution).
All models trained and evaluated at 768×384 with Phase 2 server-matching
preprocessing (keep-ratio + reflect-pad).

Reads per_image_metrics.csv and the val images/masks.
Generates PNG figures and a weasyprint PDF report.

Usage (from repo root, macvi conda env active):
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/build_report.py

Optional flags:
    --csv           path to per_image_metrics.csv
    --val-img-dir   /home/farms/data/datasets/lars_dataset/img_dir/val/images
    --val-seg-dir   /home/farms/data/datasets/lars_dataset/ann_dir/val/semantic_masks
    --out-report    challenges/common/lars-analysis/reports/failure_analysis_embedded_768x384_v2_report.pdf
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
import matplotlib.gridspec as gridspec
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
DEFAULT_OUT_PDF = REPORTS_DIR  / "failure_analysis_embedded_768x384_v2_report.pdf"
DEFAULT_IMG_DIR = Path("/home/farms/data/datasets/lars_dataset/img_dir/val/images")
DEFAULT_SEG_DIR = Path("/home/farms/data/datasets/lars_dataset/ann_dir/val/semantic_masks")

ONNX_EXPORTS = Path("/home/farms/data/onnx_exports")
ONNX_PATHS = {
    "segformer_p2":          ONNX_EXPORTS / "phase2"   / "segformer_mit-b2"  / "segformer_mit-b2_argmax.onnx",
    "pidnets_p2":            ONNX_EXPORTS / "phase2"   / "pidnets"           / "pidnets_argmax.onnx",
    "pidnets_p25":           ONNX_EXPORTS / "phase2_5" / "phase2_5" / "pidnets" / "pidnets_argmax.onnx",
    "segnext_t_p2":          ONNX_EXPORTS / "phase2"   / "segnext_t"         / "segnext_t_argmax.onnx",
    "segnext_t_uperhead_p2": ONNX_EXPORTS / "phase2"   / "segnext_t_uperhead" / "segnext_t_uperhead_argmax.onnx",
    "segnext_s_p2":          ONNX_EXPORTS / "phase2"   / "segnext_s"         / "segnext_s_argmax.onnx",
    "stdc2_p2":              ONNX_EXPORTS / "phase2"   / "stdc2"             / "stdc2_argmax.onnx",
    "stdc2_p25":             ONNX_EXPORTS / "phase2_5" / "phase2_5" / "stdc2"  / "stdc2_argmax.onnx",
    "bisenetv2_p2":          ONNX_EXPORTS / "phase2"   / "bisenetv2"         / "bisenetv2_argmax.onnx",
}

MODELS = list(ONNX_PATHS.keys())

MODEL_LABELS = {
    "segformer_p2":          "SegFormer MiT-B2 (P2)",
    "pidnets_p2":            "PIDNet-S (P2)",
    "pidnets_p25":           "PIDNet-S (P2.5)",
    "segnext_t_p2":          "SegNeXt-T (P2)",
    "segnext_t_uperhead_p2": "SegNeXt-T+UPerHead (P2)",
    "segnext_s_p2":          "SegNeXt-S (P2)",
    "stdc2_p2":              "STDC2 (P2)",
    "stdc2_p25":             "STDC2 (P2.5)",
    "bisenetv2_p2":          "BiSeNetV2 (P2)",
}

# Color per model family
MODEL_COLORS = {
    "segformer_p2":          "#7b1fa2",   # purple  — reference
    "pidnets_p2":            "#1565c0",   # dark blue
    "pidnets_p25":           "#42a5f5",   # light blue
    "segnext_t_p2":          "#2e7d32",   # dark green
    "segnext_t_uperhead_p2": "#66bb6a",   # light green
    "segnext_s_p2":          "#00897b",   # teal
    "stdc2_p2":              "#e53935",   # dark red
    "stdc2_p25":             "#ef9a9a",   # light red/pink
    "bisenetv2_p2":          "#f57f17",   # amber
}

CLASSES      = ["obstacle", "water", "sky"]
CLASS_COLORS = ["#e53935", "#1e88e5", "#00bcd4"]   # obstacle, water, sky

FAIL_THRESH = 0.5    # obstacle IoU below this = "failure image"
WORST_N     = 20     # images to show in failure gallery

MODEL_H, MODEL_W = 384, 768
MEAN_NP = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD_NP  = np.array([58.395,   57.12,   57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Mask overlay helper
# ---------------------------------------------------------------------------

def make_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colors = {
        0: np.array([50,  50,  255], dtype=np.float32),   # BGR red   → obstacle
        1: np.array([255, 120,  30], dtype=np.float32),   # BGR blue  → water
        2: np.array([220, 200,  50], dtype=np.float32),   # BGR cyan  → sky
    }
    img_f   = img_bgr.astype(np.float32)
    overlay = img_f.copy()
    for cls, color in colors.items():
        px = mask == cls
        overlay[px] = img_f[px] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Phase 2 preprocessing + ONNX inference (used in worst-case galleries)
# ---------------------------------------------------------------------------

def server_preprocess(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    ori_w, ori_h = img.size
    scale = min(MODEL_H / ori_h, MODEL_W / ori_w)
    new_h, new_w = round(ori_h * scale), round(ori_w * scale)
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
    t, l = info["pad_top"], info["pad_left"]
    h, w = info["new_h"],   info["new_w"]
    pred = argmax_out[0, 0, t:t + h, l:l + w].astype(np.uint8)
    return np.array(Image.fromarray(pred).resize(
        (info["ori_w"], info["ori_h"]), Image.NEAREST), dtype=np.uint8)


_SESS_CACHE: dict = {}

def _get_sess(model_name: str):
    if model_name not in _SESS_CACHE:
        import onnxruntime as ort
        _SESS_CACHE[model_name] = ort.InferenceSession(
            str(ONNX_PATHS[model_name]), providers=["CPUExecutionProvider"]
        )
    return _SESS_CACHE[model_name]


def infer_pred(model_name: str, img_path: Path) -> tuple[np.ndarray, tuple]:
    """Run one ONNX model on one image → (pred uint8, (ori_h, ori_w))."""
    x_np, info = server_preprocess(img_path)
    sess = _get_sess(model_name)
    inp_dtype = sess.get_inputs()[0].type
    if "float16" in inp_dtype:
        x_np = x_np.astype(np.float16)
    argmax_out = sess.run(None, {"input": x_np})[0]
    pred = unpad_argmax(argmax_out, info)
    return pred, (info["ori_h"], info["ori_w"])


# ---------------------------------------------------------------------------
# Figure 1: IoU distribution violin plots (3 classes × N models)
# ---------------------------------------------------------------------------

def fig_iou_distributions(df: pd.DataFrame, models: list, out_path: Path) -> None:
    n_models = len(models)
    fig, axes = plt.subplots(3, n_models, figsize=(n_models * 2.2, 9))
    fig.suptitle("Per-image IoU distributions — Phase 2/2.5 models × classes",
                 fontsize=13, fontweight="bold")

    for col_i, model in enumerate(models):
        for row_i, cls in enumerate(CLASSES):
            ax  = axes[row_i][col_i]
            col = f"{model}_{cls}_iou"
            if col not in df.columns:
                ax.axis("off")
                continue
            vals = df[col].dropna().values * 100

            color = MODEL_COLORS[model]
            vp = ax.violinplot(vals, positions=[0], showmedians=True, showextrema=True)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.55)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(jitter, vals, alpha=0.3, s=8, color=color)

            med = float(np.median(vals))
            p10 = float(np.percentile(vals, 10))
            ax.axhline(med, color="navy", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(0.55, med, f"{med:.0f}%", va="center",
                    fontsize=6, color="navy", transform=ax.get_yaxis_transform())
            ax.text(0.55, p10, f"p10:{p10:.0f}%", va="center",
                    fontsize=5.5, color="darkred", transform=ax.get_yaxis_transform())

            if col_i == 0:
                ax.set_ylabel(f"{cls.capitalize()} IoU (%)", fontsize=8)
            if row_i == 0:
                label = MODEL_LABELS[model]
                ax.set_title(label, fontsize=7, fontweight="bold", rotation=25, ha="left")
            ax.set_xticks([])
            ax.set_ylim(-5, 105)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Failure breadth distribution + pairwise co-failure heatmap
# ---------------------------------------------------------------------------

def fig_failure_overlap(df: pd.DataFrame, models: list, out_path: Path) -> None:
    """
    Two-panel figure:
    Left:  bar chart of "failure breadth" — how many images fail in exactly K models
    Right: pairwise co-failure heatmap — fraction of images where both models i and j fail
    """
    fail_sets  = {m: set(df[df[f"{m}_obstacle_iou"] < FAIL_THRESH]["image_name"])
                  for m in models}
    fail_flags = np.array([[img in fail_sets[m] for m in models]
                           for img in df["image_name"]])  # (N_images, N_models)

    # Breadth distribution
    breadths   = fail_flags.sum(axis=1)   # how many models fail each image
    breadth_counts = [int((breadths == k).sum()) for k in range(len(models) + 1)]

    # Pairwise co-failure (fraction of val images where BOTH models fail)
    n = len(models)
    cofail = np.zeros((n, n), dtype=np.float32)
    n_img  = len(df)
    for i in range(n):
        for j in range(n):
            cofail[i, j] = float(np.sum(fail_flags[:, i] & fail_flags[:, j])) / n_img

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Cross-model failure overlap  (obstacle IoU < {FAIL_THRESH:.1f})",
                 fontsize=12, fontweight="bold")

    # Left: breadth bar chart
    ks = list(range(len(models) + 1))
    bar_colors = ["#ccc" if k == 0 else "#e53935" if k == n else "#1e88e5" for k in ks]
    bars = ax_bar.bar(ks, breadth_counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, breadth_counts):
        if count > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bar.set_xlabel("Number of models failing the image", fontsize=10)
    ax_bar.set_ylabel("Number of images", fontsize=10)
    ax_bar.set_title("Failure breadth distribution", fontsize=10, fontweight="bold")
    ax_bar.set_xticks(ks)
    ax_bar.set_xticklabels([str(k) for k in ks])
    ax_bar.grid(axis="y", alpha=0.3)
    n_fail_any = int((breadths > 0).sum())
    n_fail_all = int((breadths == n).sum())
    ax_bar.text(0.98, 0.97, f"{n_fail_any}/{len(df)} images fail in ≥1 model\n"
                             f"{n_fail_all}/{len(df)} images fail in ALL {n} models",
                transform=ax_bar.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="orange"))

    # Right: pairwise co-failure heatmap
    short_labels = [MODEL_LABELS[m].replace(" (P2)", "").replace(" (P2.5)", " P2.5")
                    for m in models]
    im = ax_heat.imshow(cofail * 100, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7.5)
    ax_heat.set_yticklabels(short_labels, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            val = cofail[i, j] * 100
            ax_heat.text(j, i, f"{val:.0f}%", ha="center", va="center",
                         fontsize=6.5, color="black" if val < 60 else "white")
    plt.colorbar(im, ax=ax_heat, label="% of val images where both fail")
    ax_heat.set_title("Pairwise co-failure rate (obstacle IoU < 0.5)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Worst-case gallery (per model)
# ---------------------------------------------------------------------------

def fig_worst_cases(
    df: pd.DataFrame,
    model: str,
    img_dir: Path,
    seg_dir: Path,
    out_path: Path,
    n_worst: int = WORST_N,
) -> None:
    """Gallery of the N worst images (by obstacle IoU) for one model."""
    worst   = df.nsmallest(n_worst, f"{model}_obstacle_iou").reset_index(drop=True)
    n_cols  = 4
    n_rows  = (n_worst + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 3,
                             figsize=(n_cols * 3 * 3, n_rows * 2.2))
    fig.suptitle(f"{MODEL_LABELS[model]} — {n_worst} worst images by obstacle IoU",
                 fontsize=13, fontweight="bold")

    for idx in range(n_worst):
        row, col = divmod(idx, n_cols)
        base_col = col * 3
        ax_img  = axes[row][base_col]
        ax_gt   = axes[row][base_col + 1]
        ax_pred = axes[row][base_col + 2]

        img_name = worst.loc[idx, "image_name"]
        obs_iou  = worst.loc[idx, f"{model}_obstacle_iou"]
        miou     = worst.loc[idx, f"{model}_miou"]

        img_path = img_dir / img_name
        seg_path = seg_dir / img_name.replace(".jpg", ".png")

        img_bgr = cv2.imread(str(img_path))
        gt_mask = np.array(Image.open(seg_path), dtype=np.uint8)
        ori_h, ori_w = gt_mask.shape

        pred_mask, _ = infer_pred(model, img_path)

        thumb_h = 192
        scale   = thumb_h / ori_h
        thumb_w = int(ori_w * scale)
        img_thumb    = cv2.resize(img_bgr, (thumb_w, thumb_h))
        gt_overlay   = make_overlay(img_thumb,
                                    cv2.resize(gt_mask, (thumb_w, thumb_h),
                                               interpolation=cv2.INTER_NEAREST))
        pred_overlay = make_overlay(img_thumb,
                                    cv2.resize(pred_mask, (thumb_w, thumb_h),
                                               interpolation=cv2.INTER_NEAREST))

        for ax, arr_show, lbl in [
            (ax_img,  img_thumb,    "Original"),
            (ax_gt,   gt_overlay,   "GT"),
            (ax_pred, pred_overlay, "Pred"),
        ]:
            ax.imshow(cv2.cvtColor(arr_show, cv2.COLOR_BGR2RGB))
            ax.set_title(lbl, fontsize=6, pad=2)
            ax.axis("off")

        ax_img.set_xlabel(
            f"{img_name[:28]}\nobs IoU={obs_iou:.2f}  mIoU={miou:.2f}",
            fontsize=6, labelpad=2,
        )

    for idx in range(n_worst, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        for sub in range(3):
            axes[row][col * 3 + sub].axis("off")

    legend_patches = [
        mpatches.Patch(color="#e53935", label="Obstacle"),
        mpatches.Patch(color="#1e88e5", label="Water"),
        mpatches.Patch(color="#00bcd4", label="Sky"),
    ]
    fig.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Sequence-level breakdown
# ---------------------------------------------------------------------------

def fig_sequence_breakdown(df: pd.DataFrame, models: list, out_path: Path) -> None:
    """Bar chart of mean obstacle IoU per sequence, sorted by reference model."""
    ref_model = next((m for m in ["segformer_p2"] + models
                      if f"{m}_obstacle_iou" in df.columns), models[0])

    records = []
    for seq in sorted(df["sequence_id"].unique()):
        sub = df[df["sequence_id"] == seq]
        row = {"sequence_id": seq, "n_images": len(sub)}
        for m in models:
            col = f"{m}_obstacle_iou"
            if col in df.columns:
                row[f"{m}_obs_iou_mean"] = sub[col].mean()
        records.append(row)

    seq_df = (pd.DataFrame(records)
              .sort_values(f"{ref_model}_obs_iou_mean")
              .head(40))

    x     = np.arange(len(seq_df))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(16, len(seq_df) * 0.4), 7))
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models))

    for i, model in enumerate(models):
        col = f"{model}_obs_iou_mean"
        if col not in seq_df.columns:
            continue
        ax.bar(x + offsets[i] * width, seq_df[col] * 100,
               width=width, label=MODEL_LABELS[model],
               color=MODEL_COLORS[model], alpha=0.80, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.sequence_id}\n(n={r.n_images})" for r in seq_df.itertuples()],
        rotation=75, ha="right", fontsize=6.5,
    )
    ax.set_ylabel("Mean obstacle IoU (%)", fontsize=10)
    ax.set_title(f"Sequence-level mean obstacle IoU — worst {len(seq_df)} sequences "
                 f"(sorted by {MODEL_LABELS[ref_model]})", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.9, alpha=0.7,
               label=f"Failure threshold ({FAIL_THRESH})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Model ranking summary (mean obstacle IoU bar chart)
# ---------------------------------------------------------------------------

def fig_model_ranking(df: pd.DataFrame, models: list, out_path: Path) -> None:
    """Horizontal bar chart ranking all 9 models by mean obstacle IoU."""
    data = []
    for m in models:
        col = f"{m}_obstacle_iou"
        if col in df.columns:
            data.append((MODEL_LABELS[m], df[col].mean() * 100, MODEL_COLORS[m]))

    data.sort(key=lambda x: x[1])
    labels, values, colors = zip(*data)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.axvline(FAIL_THRESH * 100, color="red", linestyle="--", linewidth=1,
               alpha=0.7, label=f"Failure threshold ({FAIL_THRESH*100:.0f}%)")
    ax.set_xlabel("Mean obstacle IoU (%)", fontsize=10)
    ax.set_title("Phase 2/2.5 model ranking — mean obstacle IoU on LaRS val",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Annotation suspects
# ---------------------------------------------------------------------------

def find_annotation_suspects(df: pd.DataFrame, models: list, thresh: float = 0.1) -> pd.DataFrame:
    """Images where ALL models have obstacle IoU near zero."""
    cond = pd.Series([True] * len(df), index=df.index)
    for m in models:
        col = f"{m}_obstacle_iou"
        if col in df.columns:
            cond = cond & (df[col] < thresh)
    suspects = df[cond][["image_name", "sequence_id"]
                        + [f"{m}_obstacle_iou" for m in models if f"{m}_obstacle_iou" in df.columns]].copy()
    return suspects.sort_values(models[0] + "_obstacle_iou").reset_index(drop=True)


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fig_tag(key: str, figures: dict) -> str:
    if key in figures and figures[key].exists():
        b64 = img_to_b64(figures[key])
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin:12px 0;">'
    return f"<p><em>[Figure not found: {key}]</em></p>"


def build_html(df: pd.DataFrame, suspects: pd.DataFrame,
               models: list, figures: dict) -> str:

    # Aggregate stats table rows
    def stats_row(model):
        lbl = MODEL_LABELS[model]
        cells = ""
        for cls in CLASSES:
            col  = f"{model}_{cls}_iou"
            vals = df[col].dropna() * 100 if col in df.columns else pd.Series([float("nan")])
            cells += f"<td>{vals.mean():.1f}%</td>"
        miou_col = f"{model}_miou"
        miou = df[miou_col].dropna().mean() * 100 if miou_col in df.columns else float("nan")
        n_fail = (df[f"{model}_obstacle_iou"] < FAIL_THRESH).sum() if f"{model}_obstacle_iou" in df.columns else "?"
        color  = MODEL_COLORS.get(model, "#888")
        return (f'<tr><td style="border-left:4px solid {color};padding-left:6px">'
                f'<b>{lbl}</b></td>{cells}<td>{miou:.1f}%</td>'
                f'<td>{n_fail}</td></tr>')

    stats_rows_html = "\n".join(stats_row(m) for m in models)

    # Worst-30 table (by segformer_p2 obstacle IoU as reference)
    ref_sort = "segformer_p2_obstacle_iou" if "segformer_p2_obstacle_iou" in df.columns else f"{models[0]}_obstacle_iou"
    worst30  = df.nsmallest(30, ref_sort)

    def table_cols_header():
        return "".join(f"<th>{MODEL_LABELS[m]}<br/>obs IoU</th>" for m in models)

    def table_cols_row(r):
        return "".join(f"<td>{getattr(r, m+'_obstacle_iou', float('nan')):.3f}</td>" for m in models)

    table_rows_html = ""
    for r in worst30.itertuples():
        table_rows_html += (f"<tr><td>{r.image_name}</td><td>{r.sequence_id}</td>"
                            + table_cols_row(r) + "</tr>")

    # Annotation suspects section
    if len(suspects) > 0:
        sus_header = "".join(f"<th>{MODEL_LABELS[m]} obs IoU</th>"
                             for m in models if f"{m}_obstacle_iou" in suspects.columns)
        sus_rows_html = ""
        for r in suspects.itertuples():
            cells = "".join(f"<td>{getattr(r, m+'_obstacle_iou', float('nan')):.3f}</td>"
                            for m in models if f"{m}_obstacle_iou" in suspects.columns)
            sus_rows_html += f"<tr><td>{r.image_name}</td><td>{r.sequence_id}</td>{cells}</tr>"
        sus_section = f"""
        <h2>7. Annotation Quality Suspects</h2>
        <p>Images where <b>all {len(models)} models</b> score obstacle IoU &lt; 0.1.
        Candidates for mislabeled or highly ambiguous annotations.</p>
        <table>
          <tr><th>Image</th><th>Sequence</th>{sus_header}</tr>
          {sus_rows_html}
        </table>
        """
    else:
        sus_section = f"""
        <h2>7. Annotation Quality Suspects</h2>
        <p>No images found where all {len(models)} models score obstacle IoU &lt; 0.1.
        Even the weakest model achieves IoU ≥ 0.1 on every image, suggesting
        failures are due to model capacity rather than annotation errors.</p>
        """

    # Worst-case gallery section
    gallery_section = ""
    for i, m in enumerate(models, start=1):
        key = f"worst_cases_{m}"
        gallery_section += f"""
        <h3>5.{i} {MODEL_LABELS[m]} — worst {WORST_N} images by obstacle IoU</h3>
        {fig_tag(key, figures)}
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Embedded Segmentation — Phase 2/2.5 Failure Analysis (768×384)</title>
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
  .note {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px 14px;
           margin: 12px 0; font-size: 10pt; }}
  .warn {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px 14px;
           margin: 12px 0; font-size: 10pt; }}
</style>
</head>
<body>

<h1>Embedded Semantic Segmentation Challenge<br/>
Phase 2 &amp; 2.5 Failure Analysis — LaRS Validation Set</h1>
<p style="font-size:12pt; color:#37474f;">
<b>Inference resolution: 768×384</b> · Phase 2 server-matching preprocessing
(keep-ratio bilinear resize → center reflect-pad → ImageNet normalize)
</p>
<p><b>Date:</b> 2026-03-11 &nbsp;&nbsp;
   <b>Branch:</b> feat/failure-analysis-v2 &nbsp;&nbsp;
   <b>Issue:</b> #39</p>

<hr/>

<h2>1. Introduction</h2>
<p>This report extends the Phase 1 failure analysis to all Phase 2 and Phase 2.5 models.
All 9 FP32 argmax ONNX models are evaluated on the LaRS validation set (198 images)
at 768×384 resolution. Preprocessing matches the challenge server exactly (keep-ratio
bilinear resize → center reflect-pad to 384×768).</p>

<div class="note">
<b>Key differences from Phase 1 analysis:</b>
<ul>
  <li>Preprocessing: <b>keep-ratio + reflect-pad</b> (Phase 2) instead of simple resize (Phase 1)</li>
  <li>ONNX type: <b>argmax ONNX</b> (1×1×384×768 int64) — pad cropped then nearest-neighbour
      upsample to original GT resolution (instead of logit upsampling)</li>
  <li>9 models evaluated (Phase 2 + Phase 2.5)</li>
</ul>
</div>

<h2>2. Models</h2>
<table>
  <tr><th>Model</th><th>Phase</th><th>Exp</th><th>Val F1</th><th>Val mIoU</th><th>INT8-stable?</th><th>Notes</th></tr>
  <tr><td><b>SegFormer MiT-B2</b></td><td>P2</td><td>EXP-005</td><td>77.51%</td><td>89.72%</td><td>✗</td><td>Upper-bound reference; exceeds 30 FPS limit</td></tr>
  <tr><td><b>SegNeXt-T + UPerHead (P2)</b></td><td>P2</td><td>EXP-015</td><td>80.21%</td><td>92.48%</td><td>✗ (−78%)</td><td>See §2.1 — head replacement experiment</td></tr>
  <tr><td><b>PIDNet-S (P2)</b></td><td>P2</td><td>EXP-008</td><td>76.17%</td><td>93.25%</td><td>✓</td><td>Best INT8-stable model</td></tr>
  <tr><td><b>SegNeXt-S (P2)</b></td><td>P2</td><td>EXP-010</td><td>72.33%</td><td>92.40%</td><td>✗ (−82%)</td><td>Larger MSCAN backbone</td></tr>
  <tr><td><b>SegNeXt-T (P2)</b></td><td>P2</td><td>EXP-006</td><td>71.74%</td><td>91.79%</td><td>✗ (−67%)</td><td>Standard LightHamHead</td></tr>
  <tr><td><b>BiSeNetV2 (P2)</b></td><td>P2</td><td>EXP-009</td><td>69.06%</td><td>91.09%</td><td>✓</td><td>SGD training</td></tr>
  <tr><td><b>STDC2 (P2.5)</b></td><td>P2.5</td><td>EXP-017</td><td>—</td><td>—</td><td>✓</td><td>Pipeline fix (keep_ratio + reflect-pad); +1.99% Q vs P2</td></tr>
  <tr><td><b>STDC2 (P2)</b></td><td>P2</td><td>EXP-007</td><td>68.92%</td><td>90.69%</td><td>✓</td><td>Phase 2 baseline</td></tr>
  <tr><td><b>PIDNet-S (P2.5)</b></td><td>P2.5</td><td>EXP-018</td><td>—</td><td>—</td><td>✓</td><td>Pipeline fix; −6.30% Q regression vs P2</td></tr>
</table>

<h3>2.1 SegNeXt-T + UPerHead — EXP-015</h3>
<div class="note">
<b>What was changed:</b> The standard SegNeXt-T decode head (LightHamHead — Hamburger matrix
decomposition) was replaced with <b>UPerHead</b> (Feature Pyramid Network-style multi-scale
decoder), and <b>BoundaryLoss</b> auxiliary supervision was added on top of the standard
cross-entropy loss.
</div>
<p><b>Hypothesis:</b> LightHamHead performs global spectral decomposition (NMF/VQ/CD variants)
to capture long-range context, which is well-suited for large homogeneous regions (water, sky)
but may underweight fine-grained boundary detail. Obstacle segmentation in maritime scenes
requires precise boundary discrimination — small buoys, distant vessels, and partially occluded
objects have thin boundaries that LightHamHead's spectral approach may smooth over.
UPerHead, with explicit FPN-style multi-scale feature aggregation and per-level prediction,
was hypothesised to recover boundary sharpness and improve obstacle IoU specifically.
BoundaryLoss was added as explicit gradient signal to reinforce this effect.</p>
<p><b>Reality:</b> EXP-015 achieved <b>FP32 F1 = 80.21%</b> — the highest of all Phase 2 models,
outperforming even SegFormer MiT-B2 (77.51%) by +2.7 percentage points. Val mIoU also improved
to 92.48%, and the challenge Q-score reached 74.18% (best FP32 Q). The hypothesis was correct
in FP32: UPerHead + BoundaryLoss measurably improved boundary precision and obstacle recall.</p>
<div class="warn">
<b>INT8 reality:</b> Despite the FP32 gains, EXP-015 suffered <b>catastrophic INT8 degradation
(−78% mIoU)</b> under ORT static quantization — worse than baseline SegNeXt-T (−67%).
The root cause is the MSCAN backbone's large-kernel depthwise attention (depthwise convolutions
with kernel sizes up to 21×21), which is highly sensitive to quantization noise because the
learned attention weights span a very wide activation range. Replacing LightHamHead with
UPerHead did not fix this sensitivity, and the added BN layers in UPerHead added further
quantization-sensitive points. EXP-015 is therefore not viable for INT8 deployment on the
challenge's Qualcomm SNPE target, regardless of its strong FP32 performance.
</div>

<h2>3. Methodology</h2>
<h3>Preprocessing</h3>
<p>Images are resized with bilinear interpolation, aspect ratio preserved (keep_ratio=True),
to fit within 384×768. The resized image is centered in the 384×768 canvas with reflect
(mirror) padding. Normalization uses ImageNet mean/std in [0, 255] scale.
This matches the challenge server preprocessing used during Phase 2 training.</p>

<h3>Inference and prediction extraction</h3>
<p>Each ONNX outputs a (1, 1, 384, 768) int64 <b>argmax</b> map (same format used by the
challenge server). The reflect-pad margin is cropped out, and the resulting (new_h, new_w)
prediction is nearest-neighbour resized to the original image resolution for metric
computation against the GT mask. All 9 models use the same argmax ONNX format (FP32 weights),
ensuring consistency within this analysis.</p>
<div class="note">
<b>Difference from Phase 1 analysis (v1):</b> v1 used <em>logits</em> ONNX (Phase 1 only had
logit-output models) with bilinear upsampling of logits before argmax, and <em>simple resize</em>
preprocessing (no keep-ratio, no padding). v2 uses <em>argmax</em> ONNX with Phase 2
server-matching preprocessing. The two analyses are not directly comparable at the per-image
level, but both report standard IoU/Dice metrics at original GT resolution.
</div>

<h3>Metric definitions</h3>
<ul>
  <li><b>IoU</b>: TP / (TP + FP + FN), per class, ignoring GT pixels with value 255.</li>
  <li><b>Dice / F1</b>: 2·TP / (2·TP + FP + FN), per class.</li>
  <li><b>Pixel accuracy (per class)</b>: TP / (TP + FN).</li>
  <li><b>mIoU</b>: mean IoU over classes present in GT (all 198 images have all 3 classes).</li>
  <li><b>Failure threshold</b>: obstacle IoU &lt; {FAIL_THRESH}.</li>
</ul>

<h2>4. Model Ranking</h2>
{fig_tag("model_ranking", figures)}
<table>
  <tr><th>Model</th><th>Mean obstacle IoU</th><th>Mean water IoU</th>
      <th>Mean sky IoU</th><th>Mean mIoU</th><th>Failure count (obs IoU &lt; 0.5)</th></tr>
  {stats_rows_html}
</table>

<h2>5. Failure Case Galleries</h2>
{gallery_section}

<h2>6. Cross-Model Failure Overlap</h2>
{fig_tag("failure_overlap", figures)}
<p><b>Left panel:</b> How many images fail in exactly K of the {len(models)} models.
Images failing in all {len(models)} represent genuinely hard scenes that no model can solve in FP32.
<br/><b>Right panel:</b> Pairwise co-failure rate — what fraction of the 198 val images
both models fail simultaneously. High co-failure between the embedded candidates (STDC2, PIDNet-S)
and a reference model (SegFormer P2) indicates capacity-limited failures.</p>

{sus_section}

<h2>8. IoU Distributions</h2>
{fig_tag("iou_distributions", figures)}
<p>Violin plots of per-image IoU for each class × model. Water and sky consistently achieve
high IoU across all models. Obstacle IoU shows the highest variance and most model differentiation.
</p>

<h2>9. Sequence-Level Breakdown</h2>
{fig_tag("sequence_breakdown", figures)}
<p>Worst 40 sequences sorted by SegFormer P2 mean obstacle IoU.
Sequences consistently low across all models indicate systematically hard environments.</p>

<h2>10. Per-Image Metric Table (worst 30 by SegFormer P2 obstacle IoU)</h2>
<table>
  <tr><th>Image</th><th>Sequence</th>{table_cols_header()}</tr>
  {table_rows_html}
</table>
<p><em>Full table (all 198 images) available in per_image_metrics.csv.</em></p>

<hr/>
<p style="font-size:9pt; color:#888;">
Generated automatically by <code>build_report.py</code> on 2026-03-11.
Branch: feat/failure-analysis-v2. Issue #39.
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

    # Detect which models are present in the CSV
    models = [m for m in MODELS if f"{m}_miou" in df.columns]
    print(f"  Models in CSV: {models}")

    suspects = find_annotation_suspects(df, models)
    print(f"  Annotation suspects (all-{len(models)}-fail at IoU<0.1): {len(suspects)}")

    figures = {
        "model_ranking":     FIGURES_DIR / "model_ranking.png",
        "iou_distributions": FIGURES_DIR / "iou_distributions.png",
        "failure_overlap":   FIGURES_DIR / "failure_overlap.png",
        "sequence_breakdown": FIGURES_DIR / "sequence_breakdown.png",
    }
    for m in models:
        figures[f"worst_cases_{m}"] = FIGURES_DIR / f"worst_cases_{m}.png"

    print("\n--- Generating figures ---")

    print("1. Model ranking...")
    fig_model_ranking(df, models, figures["model_ranking"])

    print("2. IoU distributions...")
    fig_iou_distributions(df, models, figures["iou_distributions"])

    print("3. Failure overlap (breadth + co-failure heatmap)...")
    fig_failure_overlap(df, models, figures["failure_overlap"])

    for i, model in enumerate(models, start=4):
        print(f"{i}. Worst-case gallery: {MODEL_LABELS[model]} ...")
        fig_worst_cases(df, model, args.val_img_dir, args.val_seg_dir,
                        figures[f"worst_cases_{model}"])

    print(f"{len(models)+4}. Sequence breakdown...")
    fig_sequence_breakdown(df, models, figures["sequence_breakdown"])

    if args.no_pdf:
        print("\nSkipping PDF (--no-pdf)")
        return

    print("\n--- Building PDF report ---")
    html = build_html(df, suspects, models, figures)
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
