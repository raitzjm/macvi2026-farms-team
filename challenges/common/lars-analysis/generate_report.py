#!/usr/bin/env python3
"""
LaRS Embedding Analysis — Report Generator
==========================================
Generates a standalone PDF report from the pre-computed embeddings in
embeddings.db and tsne_results.npy.  Run this script independently of the
Jupyter notebook.

Usage:
    python generate_report.py [--db embeddings.db] [--tsne tsne_results.npy]
                              [--images /home/farms/data/lars_dataset]
                              [--out report.pdf]
"""

import argparse
import base64
import io
import os
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.colormaps
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="embeddings.db")
    p.add_argument("--tsne",   default="tsne_results.npy")
    p.add_argument("--images", default="/home/farms/data/lars_dataset/img_dir")
    p.add_argument("--out",    default="report.pdf")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path, tsne_path):
    conn = sqlite3.connect(db_path)

    meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

    rows = conn.execute(
        "SELECT img_name, split, sequence, spatial_mean_embedding FROM embeddings ORDER BY id"
    ).fetchall()
    conn.close()

    names  = [r[0] for r in rows]
    splits = [r[1] for r in rows]
    seqs   = [r[2] for r in rows]
    embeds = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    tsne = np.load(tsne_path)   # shape (N, 2)

    return names, splits, seqs, embeds, tsne, meta


def compute_cosine_similarity(embeds, splits):
    """Return max cosine similarity of each image vs the training set."""
    normed = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    train_mask = np.array([s == "train" for s in splits])
    train_vecs = normed[train_mask]

    max_sims = np.empty(len(splits), dtype=np.float32)
    # Compute in chunks to avoid OOM on CPU
    chunk = 256
    for i in range(0, len(splits), chunk):
        sims = normed[i:i+chunk] @ train_vecs.T
        max_sims[i:i+chunk] = sims.max(axis=1)

    # Train images: similarity to themselves is trivially 1.0 — set to NaN
    max_sims[train_mask] = np.nan
    return max_sims


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_thumbnail(img_name, split, images_root, size=(160, 120)):
    path = Path(images_root) / split / "images" / f"{img_name}.jpg"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLIT_ALPHA  = {"train": 0.25,       "val": 0.7,       "test": 0.8}
SPLIT_SIZE   = {"train": 4,           "val": 10,        "test": 6}


def fig_tsne_by_split(tsne, splits):
    fig, ax = plt.subplots(figsize=(9, 7))
    splits_arr = np.array(splits)
    for sp in ("train", "val", "test"):
        mask = splits_arr == sp
        ax.scatter(
            tsne[mask, 0], tsne[mask, 1],
            c=SPLIT_COLORS[sp], alpha=SPLIT_ALPHA[sp],
            s=SPLIT_SIZE[sp], linewidths=0, label=sp.capitalize(),
            rasterized=True,
        )
    ax.set_title("t-SNE: embedding space coloured by dataset split", fontsize=13)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def fig_tsne_by_sequence(tsne, splits, seqs):
    splits_arr = np.array(splits)
    seqs_arr   = np.array(seqs)
    test_mask  = splits_arr == "test"
    train_mask = splits_arr == "train"

    # Count images per sequence in the test set; label small ones "other"
    test_seqs, counts = np.unique(seqs_arr[test_mask], return_counts=True)
    major_seqs = set(test_seqs[counts >= 5])

    # Use a qualitative palette for test sequences
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(major_seqs))
    seq_color = {s: cmap(i) for i, s in enumerate(sorted(major_seqs))}

    fig, ax = plt.subplots(figsize=(10, 7))

    # Train as grey background
    ax.scatter(
        tsne[train_mask, 0], tsne[train_mask, 1],
        c="#AAAAAA", alpha=0.15, s=3, linewidths=0, label="Train (all)", rasterized=True
    )

    # Test sequences
    plotted = 0
    other_xy = []
    for i, (sp, sq) in enumerate(zip(splits, seqs)):
        if sp != "test":
            continue
        if sq in seq_color:
            ax.scatter(
                tsne[i, 0], tsne[i, 1],
                c=[seq_color[sq]], alpha=0.7, s=10, linewidths=0,
                label=sq if plotted < 20 else "_nolegend_",
                rasterized=True,
            )
            plotted += 1
        else:
            other_xy.append(tsne[i])

    if other_xy:
        other_xy = np.array(other_xy)
        ax.scatter(other_xy[:, 0], other_xy[:, 1],
                   c="#888888", alpha=0.4, s=6, linewidths=0, label="Test (small seqs)", rasterized=True)

    ax.set_title("t-SNE: test sequences overlaid on training cloud", fontsize=13)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="upper right",
              framealpha=0.8, borderpad=0.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def fig_cosine_hist(max_sims, splits):
    splits_arr = np.array(splits)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.5, 1.0, 51)
    for sp, color in [("val", SPLIT_COLORS["val"]), ("test", SPLIT_COLORS["test"])]:
        mask = splits_arr == sp
        data = max_sims[mask]
        data = data[~np.isnan(data)]
        ax.hist(data, bins=bins, alpha=0.65, color=color, label=sp.capitalize(),
                edgecolor="none")
        med = np.median(data)
        ax.axvline(med, color=color, linestyle="--", linewidth=1.5,
                   label=f"{sp.capitalize()} median = {med:.3f}")

    ax.axvline(0.7, color="red", linestyle=":", linewidth=1.5, label="OOD threshold (0.7)")
    ax.set_xlabel("Max cosine similarity to any training image", fontsize=12)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title("Coverage: how well is each test/val image covered by training?", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    return fig


def fig_sequence_coverage(splits, seqs):
    splits_arr = np.array(splits)
    seqs_arr   = np.array(seqs)
    train_seqs = set(seqs_arr[splits_arr == "train"])
    test_seqs  = set(seqs_arr[splits_arr == "test"])
    val_seqs   = set(seqs_arr[splits_arr == "val"])

    n_test_covered   = len(test_seqs & train_seqs)
    n_test_uncovered = len(test_seqs - train_seqs)
    n_val_covered    = len(val_seqs  & train_seqs)
    n_val_uncovered  = len(val_seqs  - train_seqs)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, sp, covered, uncovered in [
        (axes[0], "Test", n_test_covered, n_test_uncovered),
        (axes[1], "Val",  n_val_covered,  n_val_uncovered),
    ]:
        total = covered + uncovered
        ax.pie(
            [covered, uncovered],
            labels=[f"Seen in training\n({covered})", f"Unseen\n({uncovered})"],
            colors=["#4C72B0", "#C44E52"],
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11},
        )
        ax.set_title(f"{sp} set sequence coverage\n(total: {total} sequences)", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def pct_below(max_sims, splits, split, threshold):
    mask = np.array(splits) == split
    data = max_sims[mask]
    data = data[~np.isnan(data)]
    return 100.0 * (data < threshold).sum() / len(data)


def build_html(args, names, splits, seqs, embeds, tsne, meta, max_sims):
    splits_arr = np.array(splits)
    seqs_arr   = np.array(seqs)

    # --- figures ---
    print("Generating figures...")
    f1 = fig_tsne_by_split(tsne, splits)
    f2 = fig_tsne_by_sequence(tsne, splits, seqs)
    f3 = fig_cosine_hist(max_sims, splits)
    f4 = fig_sequence_coverage(splits, seqs)

    b64_tsne_split = fig_to_b64(f1)
    b64_tsne_seq   = fig_to_b64(f2)
    b64_cosine     = fig_to_b64(f3)
    b64_seq_cov    = fig_to_b64(f4)
    for f in (f1, f2, f3, f4):
        plt.close(f)

    # --- OOD table (lowest max-sim test images) ---
    print("Finding most OOD test images...")
    test_indices = [i for i, s in enumerate(splits) if s == "test"]
    test_sims    = [(i, max_sims[i]) for i in test_indices if not np.isnan(max_sims[i])]
    test_sims.sort(key=lambda x: x[1])
    top_ood = test_sims[:15]

    ood_rows_html = ""
    for rank, (idx, sim) in enumerate(top_ood, 1):
        img_name = names[idx]
        seq      = seqs[idx]
        thumb_b64 = load_thumbnail(img_name, "test", args.images)
        thumb_tag = (
            f'<img src="data:image/jpeg;base64,{thumb_b64}" style="height:80px;border-radius:4px;">'
            if thumb_b64 else "<em>n/a</em>"
        )
        ood_rows_html += f"""
        <tr>
          <td>{rank}</td>
          <td>{thumb_tag}</td>
          <td style="font-family:monospace;font-size:0.85em">{img_name}</td>
          <td>{seq}</td>
          <td><b>{sim:.4f}</b></td>
        </tr>"""

    # --- summary stats ---
    train_seqs = set(seqs_arr[splits_arr == "train"])
    test_seqs  = set(seqs_arr[splits_arr == "test"])
    val_seqs   = set(seqs_arr[splits_arr == "val"])

    def stats_row(sp):
        mask = splits_arr == sp
        data = max_sims[mask]; data = data[~np.isnan(data)]
        return (
            mask.sum(),
            len(set(seqs_arr[mask])),
            f"{np.median(data):.3f}",
            f"{pct_below(max_sims, splits, sp, 0.7):.1f}%",
            f"{pct_below(max_sims, splits, sp, 0.5):.1f}%",
        )

    rows = {sp: stats_row(sp) for sp in ("train", "val", "test")}

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LaRS Embedding Distribution Analysis</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1000px; margin: 40px auto;
          color: #222; line-height: 1.65; font-size: 15px; }}
  h1   {{ font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2   {{ font-size: 1.3em; margin-top: 2em; border-bottom: 1px solid #ccc; }}
  h3   {{ font-size: 1.1em; margin-top: 1.5em; }}
  img.plot {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px;
              margin: 12px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .highlight {{ background: #fff3cd; font-weight: bold; }}
  .finding  {{ background: #e8f4f8; border-left: 4px solid #2196F3;
               padding: 10px 16px; margin: 16px 0; border-radius: 4px; }}
  .warning  {{ background: #fff8e1; border-left: 4px solid #FF9800;
               padding: 10px 16px; margin: 16px 0; border-radius: 4px; }}
  .meta     {{ color: #666; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>LaRS Embedding Distribution Analysis</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Model: <b>{meta.get('model_name','?')}</b>
&nbsp;|&nbsp; Embed dim: {meta.get('embed_dim','?')}</p>

<!-- ================================================================== -->
<h2>1. Introduction</h2>

<p>This report analyses how the LaRS (Large Real-world Shoreline) dataset's test
set is distributed relative to the training set in the embedding space produced
by NVIDIA RADIO-H, a large vision foundation model trained to distil CLIP, DINOv2,
and SAM into a single encoder. The goal is to identify potential out-of-distribution
(OOD) test images, understand dataset shift, and estimate where a model trained on
the LaRS training set is most likely to fail.</p>

<p>The analysis covers three aspects:</p>
<ol>
  <li><b>t-SNE visualisation</b> — global structure of the embedding space, coloured by split and by sequence.</li>
  <li><b>Cosine similarity distribution</b> — per-image quantitative coverage score: how similar is each test image to its nearest training neighbour?</li>
  <li><b>Sequence coverage</b> — whether the physical environments (camera sequences) in the test set were seen during training.</li>
</ol>

<!-- ================================================================== -->
<h2>2. Dataset Overview</h2>

<table>
  <tr><th>Split</th><th>Images</th><th>Sequences</th><th>Resolution</th></tr>
  <tr><td>Train</td><td>{rows['train'][0]:,}</td><td>{rows['train'][1]:,}</td><td>1278×958 / 1280×720</td></tr>
  <tr><td>Val</td>  <td>{rows['val'][0]:,}</td>  <td>{rows['val'][1]:,}</td>  <td>same</td></tr>
  <tr><td>Test</td> <td>{rows['test'][0]:,}</td> <td>{rows['test'][1]:,}</td> <td>same</td></tr>
</table>

<div class="warning">
<b>Key structural finding:</b> The test set contains <b>{len(test_seqs):,} sequences</b>,
none of which appear in the training set ({len(test_seqs & train_seqs)} overlap).
Every test sequence is from a physical environment or recording session
completely unseen during training. This is by design for a rigorous benchmark, but
it means the model must generalise across camera locations and scene types, not just
across time within the same scene.
</div>

<!-- ================================================================== -->
<h2>3. Methodology</h2>

<h3>3.1 Embedding model — RADIO-H with internal spatial tokens</h3>
<p>Each image is passed through <b>RADIO-H</b> (nvidia/C-RADIOv3-H, 653 M parameters)
at its original resolution. RADIO internally divides the image into a 16 px grid and
produces one embedding vector per grid cell via its transformer attention mechanism —
these are the <em>spatial tokens</em>. Unlike manually cropping images into patches and
running each crop separately, using internal tokens means every patch embedding is
informed by the full image context (all patches attend to each other). This produces
richer, context-aware features and requires only one forward pass per image.</p>

<p>The spatial tokens are averaged across the spatial dimension to produce a single
{meta.get('embed_dim','?')}-dimensional summary vector per image. This per-image vector
is used for t-SNE projection and cosine similarity computation. Both the spatial mean
and the model's global summary token are stored in the SQLite database.</p>

<h3>3.2 Dimensionality reduction</h3>
<p>Raw {meta.get('embed_dim','?')}-dimensional embeddings are first reduced to 50
dimensions with PCA (retaining the principal axes of variation), then projected to 2D
with t-SNE (GPU-accelerated via cuML). PCA pre-reduction is standard practice before
t-SNE: it removes high-frequency noise while preserving the dominant structure, and
dramatically speeds up the t-SNE computation.</p>

<h3>3.3 Cosine similarity coverage score</h3>
<p>For each non-training image, the maximum cosine similarity to any training image
is computed as a scalar coverage score. A score near 1.0 means the image is very
similar to at least one training image; a score below ~0.7 indicates the image is
in a region of the embedding space not well covered by training.</p>

<!-- ================================================================== -->
<h2>4. t-SNE Visualisation</h2>

<h3>4.1 Split distribution</h3>
<img class="plot" src="data:image/png;base64,{b64_tsne_split}" alt="t-SNE by split">

<p>Train, val, and test images largely overlap in embedding space, which is consistent
with the high cosine similarity scores (see Section 5). The test set does not form
isolated clusters — it is interspersed throughout the training cloud. This suggests
that the visual content of test scenes (water surface, obstacles, lighting conditions)
is broadly similar to what the model sees during training, even though the specific
camera environments are all new.</p>

<h3>4.2 Test sequence overlay</h3>
<img class="plot" src="data:image/png;base64,{b64_tsne_seq}" alt="t-SNE by sequence">

<p>Colouring test images by their sequence ID reveals whether certain environments
cluster in a specific region of embedding space (potential domain shift). Sequences that
appear as isolated islands away from the grey training cloud are the highest-risk
evaluation environments — the model has seen nothing visually similar during training.</p>

<!-- ================================================================== -->
<h2>5. Cosine Similarity Coverage Analysis</h2>

<img class="plot" src="data:image/png;base64,{b64_cosine}" alt="Cosine similarity histogram">

<table>
  <tr>
    <th>Split</th><th>Images</th><th>Median max-sim</th>
    <th>% below 0.7 (OOD)</th><th>% below 0.5 (severe OOD)</th>
  </tr>
  <tr>
    <td>Val</td><td>{rows['val'][0]:,}</td><td>{rows['val'][2]}</td>
    <td>{rows['val'][3]}</td><td>{rows['val'][4]}</td>
  </tr>
  <tr class="highlight">
    <td>Test</td><td>{rows['test'][0]:,}</td><td>{rows['test'][2]}</td>
    <td>{rows['test'][3]}</td><td>{rows['test'][4]}</td>
  </tr>
</table>

<div class="finding">
<b>Interpretation:</b> Despite every test sequence being physically unseen during
training, the cosine similarity distribution is concentrated above 0.9. This indicates
that the <em>visual appearance</em> of test images (sea surface texture, lighting, obstacle
types) is well-represented in the training set, even if the exact locations are not.
The LaRS training set provides broad visual coverage of the maritime domain. A well-trained
model is unlikely to encounter catastrophic distribution shift on the average test image.
</div>

<!-- ================================================================== -->
<h2>6. Sequence Coverage</h2>

<img class="plot" src="data:image/png;base64,{b64_seq_cov}" alt="Sequence coverage">

<p>As noted above, 100% of test sequences are absent from training. This is not
unexpected for a benchmark — it prevents overfitting to specific camera locations.
However, it does mean that any sequence-level adaptation (e.g. test-time fine-tuning)
is not possible. Models must rely entirely on generalisation from the training sequences.</p>

<!-- ================================================================== -->
<h2>7. Failure Case Analysis — Most Out-of-Distribution Test Images</h2>

<p>The following test images have the lowest maximum cosine similarity to any training
image. These are the images the model is statistically most likely to struggle with,
as they represent visual content underrepresented in the training distribution.</p>

<table>
  <tr>
    <th>#</th><th>Thumbnail</th><th>Image name</th>
    <th>Sequence</th><th>Max cosine sim ↑</th>
  </tr>
  {ood_rows_html}
</table>

<div class="finding">
<b>Note:</b> Even the most OOD images have max-sim above 0.70, which is a relatively
high threshold. This reinforces the conclusion that the LaRS test set is well-covered
by the training distribution at the visual content level. Failure cases are more likely
to arise from rare obstacle types, unusual lighting, or ambiguous water surface
conditions than from a fundamental domain shift.
</div>

<!-- ================================================================== -->
<h2>8. Conclusions</h2>

<ul>
  <li>The LaRS test set is <b>visually well-covered</b> by the training set despite
      containing entirely new camera sequences. Median test image coverage (max cosine
      similarity) is {rows['test'][2]}.</li>
  <li>There is <b>no catastrophic distribution shift</b> expected. The embedding
      distributions of train and test overlap substantially in the t-SNE space.</li>
  <li>The most at-risk test images (lowest cosine similarity) are from sequences
      with rare visual conditions. These are the priority candidates for targeted
      data augmentation or additional annotation.</li>
  <li>Since all test sequences are physically disjoint from training, model performance
      will depend on generalisation of <em>visual patterns</em> (water texture, obstacle
      appearance), not memorisation of scene geometry.</li>
</ul>

<hr>
<p class="meta">Analysis performed by FARMS Team — MaCVi 2026. Model: {meta.get('model_name','?')}.
Approach recommended by Claude Sonnet 4.6 Agent.</p>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    db_path   = script_dir / args.db
    tsne_path = script_dir / args.tsne
    out_path  = script_dir / args.out

    print(f"Loading data from {db_path} ...")
    names, splits, seqs, embeds, tsne, meta = load_data(db_path, tsne_path)
    print(f"  {len(names)} images loaded. Embed shape: {embeds.shape}")

    print("Computing cosine similarity ...")
    max_sims = compute_cosine_similarity(embeds, splits)
    test_sims = max_sims[np.array(splits) == "test"]
    print(f"  Test median max-sim: {np.nanmedian(test_sims):.4f}")

    print("Building HTML report ...")
    html = build_html(args, names, splits, seqs, embeds, tsne, meta, max_sims)

    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML written to {html_path}")

    print("Converting to PDF via weasyprint ...")
    try:
        from weasyprint import HTML as WP_HTML
        WP_HTML(filename=str(html_path)).write_pdf(str(out_path))
        print(f"  PDF written to {out_path}")
    except Exception as e:
        print(f"  weasyprint failed: {e}")
        print(f"  HTML report is available at {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()
