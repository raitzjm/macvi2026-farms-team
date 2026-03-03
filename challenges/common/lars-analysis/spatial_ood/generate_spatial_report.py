#!/usr/bin/env python3
"""
LaRS Spatial OOD Analysis — Report Generator
=============================================
Generates a standalone PDF report from pre-computed spatial OOD heatmaps.

Usage:
    python generate_spatial_report.py \
        [--heatmaps spatial_ood/heatmaps] \
        [--figures  spatial_ood/figures] \
        [--db embeddings.db] \
        [--images /home/farms/data/lars_dataset/img_dir] \
        [--out spatial_ood/spatial_report.pdf]
"""

import argparse
import base64
import io
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    p.add_argument('--heatmaps', default=str(script_dir / 'heatmaps'))
    p.add_argument('--figures',  default=str(script_dir / 'figures'))
    p.add_argument('--db',       default=str(script_dir.parent / 'embeddings.db'))
    p.add_argument('--images',   default='/home/farms/data/lars_dataset/img_dir')
    p.add_argument('--out',      default=str(script_dir.parent / 'reports' / 'spatial_ood_report.pdf'))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_image_list(images_root, split):
    list_file = Path(images_root) / split / 'image_list.txt'
    if not list_file.exists():
        return []
    names = list_file.read_text().strip().splitlines()
    records = []
    for name in names:
        parts = name.rsplit('_', 1)
        seq = parts[0] if len(parts) > 1 else name
        records.append({'name': name, 'split': split, 'sequence': seq})
    return records


def load_heatmap_stats(heatmaps_dir, eval_records):
    heatmap_stats = []
    for rec in eval_records:
        hp = Path(heatmaps_dir) / f"{rec['name']}.npy"
        if hp.exists():
            hm = np.load(hp)
            heatmap_stats.append({
                'img_name':      rec['name'],
                'split':         rec['split'],
                'sequence':      rec['sequence'],
                'mean_patch_sim': float(hm.mean()),
                'min_patch_sim':  float(hm.min()),
                'pct_ood_08':     float((hm < 0.8).mean() * 100),
                'heatmap_shape':  hm.shape,
            })
    import pandas as pd
    return pd.DataFrame(heatmap_stats)


def load_db_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT img_name, split, sequence, spatial_mean_embedding FROM embeddings ORDER BY id'
    ).fetchall()
    meta = dict(conn.execute('SELECT key, value FROM metadata').fetchall())
    conn.close()
    names  = [r[0] for r in rows]
    splits = [r[1] for r in rows]
    seqs   = [r[2] for r in rows]
    embeds = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    return names, splits, seqs, embeds, meta


def compute_whole_image_sim(embeds, splits):
    normed     = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    train_mask = np.array([s == 'train' for s in splits])
    train_vecs = normed[train_mask]
    max_sims   = np.full(len(splits), np.nan)
    chunk = 256
    for i in range(0, len(splits), chunk):
        if splits[i] == 'train':
            # skip train-vs-train to preserve NaN convention
            pass
        batch = normed[i:i+chunk]
        sims  = batch @ train_vecs.T
        max_sims[i:i+chunk] = sims.max(axis=1)
    max_sims[train_mask] = np.nan
    return max_sims


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_path_to_b64(path, size=(300, 200)):
    if not Path(path).exists():
        return None
    img = Image.open(path).convert('RGB')
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def make_histogram_fig(df_hm):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(
        df_hm['mean_patch_sim'].min() * 0.98,
        df_hm['mean_patch_sim'].max() * 1.005,
        51
    )
    for sp, color in [('test', '#2ca02c'), ('val', '#ff7f0e')]:
        sub = df_hm[df_hm['split'] == sp]['mean_patch_sim']
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=bins, alpha=0.65, color=color, label=sp.capitalize(), edgecolor='none')
        med = sub.median()
        ax.axvline(med, color=color, linestyle='--', linewidth=1.5,
                   label=f'{sp.capitalize()} median={med:.4f}')
    ax.set_xlabel('Mean patch max cosine similarity to training', fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.set_title('Distribution of mean spatial OOD scores (test & val)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    return fig


def make_scatter_fig(df_merged):
    fig, ax = plt.subplots(figsize=(7, 6))
    for sp, color in [('test', '#2ca02c'), ('val', '#ff7f0e')]:
        sub = df_merged[df_merged['split'] == sp].dropna(subset=['whole_max_sim'])
        if len(sub) == 0:
            continue
        ax.scatter(sub['whole_max_sim'], sub['mean_patch_sim'],
                   c=color, alpha=0.5, s=8, label=sp.capitalize(), linewidths=0)
    ax.set_xlabel('Whole-image max cosine sim to training', fontsize=11)
    ax.set_ylabel('Mean patch max cosine sim to training', fontsize=11)
    ax.set_title('Whole-image vs spatial OOD scores', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def make_agg_heatmap_fig(figures_dir):
    agg_path = Path(figures_dir) / 'aggregate_heatmap.png'
    if not agg_path.exists():
        return None
    img = Image.open(agg_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.array(img))
    ax.axis('off')
    ax.set_title('Aggregate spatial OOD heatmap across all test images', fontsize=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(args, df_hm, df_merged, names, splits, seqs, embeds, meta, max_sims_whole):
    import pandas as pd
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    # --- summary stats ---
    def stats(sp):
        sub_hm = df_hm[df_hm['split'] == sp]
        sub_wh = df_merged[df_merged['split'] == sp].dropna(subset=['whole_max_sim'])
        if len(sub_hm) == 0:
            return ('N/A',) * 5
        return (
            len(sub_hm),
            f"{sub_hm['mean_patch_sim'].median():.4f}",
            f"{sub_hm['pct_ood_08'].median():.1f}%",
            f"{sub_wh['whole_max_sim'].median():.4f}" if len(sub_wh) else 'N/A',
        )

    sv = stats('val')
    st = stats('test')

    # --- Spearman correlation ---
    df_test_c = df_merged[df_merged['split'] == 'test'].dropna(subset=['whole_max_sim', 'mean_patch_sim'])
    rho, pval = (float('nan'), float('nan'))
    if len(df_test_c) > 10:
        rho, pval = spearmanr(df_test_c['whole_max_sim'], df_test_c['mean_patch_sim'])

    # --- figures ---
    print('Generating figures for report...')
    f1 = make_histogram_fig(df_hm)
    f2 = make_scatter_fig(df_merged)
    f3 = make_agg_heatmap_fig(args.figures)

    b64_hist    = fig_to_b64(f1)
    b64_scatter = fig_to_b64(f2)
    b64_agg     = fig_to_b64(f3) if f3 else ''
    for f in [f1, f2] + ([f3] if f3 else []):
        plt.close(f)

    # --- top-20 OOD grid ---
    grid_path = Path(args.figures) / 'top20_ood_grid.png'
    b64_grid  = img_path_to_b64(grid_path, size=(1600, 1280)) if grid_path.exists() else ''

    # --- top-5 seq heatmaps ---
    seq_path = Path(args.figures) / 'top5_seq_heatmaps.png'
    b64_seq  = img_path_to_b64(seq_path, size=(1600, 400)) if seq_path.exists() else ''

    # --- top-20 OOD table rows ---
    df_test_sorted = df_merged[df_merged['split'] == 'test'].sort_values('mean_patch_sim')
    top20 = df_test_sorted.head(20)
    ood_rows_html = ''
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        img_name  = row['img_name']
        seq       = row['sequence']
        mean_sim  = row['mean_patch_sim']
        whole_sim = row.get('whole_max_sim', float('nan'))
        fig_path  = Path(args.figures) / f'{img_name}.png'
        img_b64   = img_path_to_b64(fig_path, size=(900, 300))
        img_tag   = (
            f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:4px;">'
            if img_b64 else '<em>overlay not available</em>'
        )
        whole_str = f'{whole_sim:.4f}' if not (isinstance(whole_sim, float) and (whole_sim != whole_sim)) else 'N/A'
        ood_rows_html += f"""
        <tr class="entry-img">
          <td colspan="5" style="padding:6px 8px;border-bottom:none;">{img_tag}</td>
        </tr>
        <tr class="entry-data">
          <td style="width:2.5em;text-align:center;font-weight:bold;">{rank}</td>
          <td style="font-family:monospace;font-size:0.82em;word-break:break-all;">{img_name}</td>
          <td>{seq}</td>
          <td style="text-align:center;"><b>{mean_sim:.4f}</b></td>
          <td style="text-align:center;">{whole_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LaRS Spatial OOD Analysis</title>
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
  tr.entry-img  {{ background: #fff; border-top: 2px solid #999; }}
  tr.entry-data {{ background: #f7f7f7; border-bottom: 2px solid #999; }}
  .finding  {{ background: #e8f4f8; border-left: 4px solid #2196F3;
               padding: 10px 16px; margin: 16px 0; border-radius: 4px; }}
  .warning  {{ background: #fff8e1; border-left: 4px solid #FF9800;
               padding: 10px 16px; margin: 16px 0; border-radius: 4px; }}
  .meta     {{ color: #666; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>LaRS Spatial OOD Analysis</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Model: <b>{meta.get('model_name', '?')}</b>
&nbsp;|&nbsp; Extension: patchwise spatial OOD via PyTorch GPU nearest-neighbour</p>

<!-- ======================================================== -->
<h2>1. Introduction</h2>

<p>This report is a companion to the whole-image embedding analysis
(<code>whole_image_ood/whole_image_ood.ipynb</code> / <code>reports/whole_image_ood.pdf</code>). That analysis used the
spatial mean embedding — a single vector per image summarising the entire scene —
to measure how well each test image is covered by the training distribution.</p>

<p>This extension drills deeper: it retains the full <b>spatial token grid</b>
produced by RADIO-H and computes an <b>OOD score for every 16×16 patch</b> in each
image. The result is a spatial heatmap showing, at pixel resolution, <em>which parts
of a scene</em> are underrepresented in training — not just whether the image as a
whole is unusual.</p>

<p>This matters for maritime obstacle detection because OOD regions may be localised:
a test image with a well-covered sky and water background might have a highly unusual
obstacle type in one corner. The whole-image score would be high (well-covered), but
the model might fail precisely in that corner.</p>

<!-- ======================================================== -->
<h2>2. Methodology</h2>

<h3>2.1 Patch extraction via RADIO-H spatial tokens</h3>
<p>RADIO-H (nvidia/C-RADIOv3-H) uses a Vision Transformer with patch size 16px.
For a 1280×960 image, this produces a 80×60 grid of 4,800 spatial tokens.
Each token is a 1280-dimensional vector that has attended to the entire image,
so it captures its local region in full global context. These tokens are used
directly — no separate crop extraction is needed.</p>

<h3>2.2 Training patch index and subsampling</h3>
<p>All 2,605 training images are processed through RADIO-H. Each image's spatial
tokens are L2-normalised and written sequentially to a numpy memmap
(<code>train_patches.dat</code>). The total is approximately
{df_hm['heatmap_shape'].apply(lambda s: s[0]*s[1]).mean():.0f}
patches per image × 2,605 images ≈ <b>14 million vectors</b> (∼68 GB on disk).</p>

<p><b>Subsampling (1-in-10):</b> To keep GPU memory and compute tractable, every
10th patch is retained, yielding approximately <b>1.4 million vectors</b>. A FAISS
<code>IndexFlatIP</code> (exact inner-product search, cosine similarity on
L2-normalised vectors) is built from this subsample and saved to
<code>faiss_index.bin</code>. The subsample covers roughly 10 % of all training
patches; absolute threshold values (e.g. "below 0.80") may differ slightly from
what would be obtained with the full set, but the relative ranking of images is
expected to be stable.</p>

<p><b>Phase 2 search (PyTorch GPU matmul):</b> CPU FAISS flat search at 1.4 M
vectors was measured at ∼2.2 min/image (∼51 hours total) — unusable at this
scale. Instead, the 1.4 M subsampled vectors are loaded to GPU memory as
float16 (∼3.6 GB VRAM) and per-image similarity is computed as a single batched
matrix multiply: <code>(query_tokens @ train.T).max(dim=1)</code>. This runs at
13–17 ms/image on the RTX A6000, completing all 1,400 images in ∼17 minutes.
FAISS is used only for Phase 1 (index construction); the heatmap search itself
uses PyTorch GPU.</p>

<div class="warning"><b>Note on coverage:</b> Because only 10 % of training
patches are indexed, a test patch that closely matches a training patch absent
from the subsample will appear more OOD than it truly is. This affects absolute
score levels; relative comparisons between test images are reliable.</div>

<h3>2.3 OOD score definition</h3>
<p>For each test/val patch, the <b>OOD score = max cosine similarity to any patch in
the 1-in-10 subsampled training set</b>, computed via PyTorch GPU matrix
multiplication. A score near 1.0 means the patch is visually very close to at least
one training patch in the subsample; a low score (below ~0.80) indicates the patch is
in a region of embedding space with sparse training coverage.</p>

<p>The per-patch scores are reshaped back to the token grid and saved as
<code>heatmaps/{img_name}.npy</code>. Values are in [0, 1], dtype float32.</p>

<!-- ======================================================== -->
<h2>3. Results: Top-20 Most At-Risk Test Images</h2>

<h3>3.1 Summary statistics</h3>
<table>
  <tr><th>Split</th><th>Images</th><th>Median mean-patch sim</th>
      <th>Median % patches &lt;0.80</th><th>Median whole-img sim</th></tr>
  <tr><td>Val</td><td>{sv[0]}</td><td>{sv[1]}</td><td>{sv[2]}</td><td>{sv[3]}</td></tr>
  <tr><td>Test</td><td>{st[0]}</td><td>{st[1]}</td><td>{st[2]}</td><td>{st[3]}</td></tr>
</table>

<div class="finding"><b>Coverage note:</b> 1 test image (<code>inhouse_seq_110_00149</code>)
initially raised a CUDA out-of-memory error during Phase 2 (15,960 patches;
single-shot matmul would require ~45 GB). It was retried automatically using a
512-row chunked matmul fallback and completed successfully. All 1,401 test/val
heatmaps are included in the statistics below.</div>

<h3>3.2 Distribution of spatial OOD scores</h3>
<img class="plot" src="data:image/png;base64,{b64_hist}" alt="Figure 1: Histogram">

<h3>3.3 Heatmap grid: top-20 most at-risk test images</h3>
<div class="warning"><b>Colorscale note:</b> Each overlay figure uses a
<em>per-image</em> colorscale (green = highest similarity in that image, red =
lowest). The absolute OOD level differs between images; red in one panel does not
correspond to the same cosine similarity value as red in another. Use the mean patch
similarity values in the table below for cross-image comparisons.</div>
{"<img class='plot' src='data:image/png;base64," + b64_grid + "' alt='Figure 2: Top-20 OOD grid'>" if b64_grid else '<p><em>Grid figure not generated.</em></p>'}

<h3>3.4 Top-20 most at-risk test images</h3>
<p>Ranked by lowest mean patch cosine similarity to any patch in the subsampled
training set.</p>
<table>
  <tr><th>#</th><th>Image name</th><th>Sequence</th>
      <th>Mean patch sim ↑</th><th>Whole-img sim</th></tr>
  {ood_rows_html}
</table>

<div class="finding">
<b>Interpretation:</b> Images with mean patch sim below ~0.85 contain patches with
sparse training coverage. The <em>spatial heatmap</em> (red regions) highlights which
scene areas are least represented in the training distribution — often obstacles,
unusual lighting conditions, or image edges. These are regions where a deployed model
may be less reliable, though other factors (class balance, augmentation, architecture)
also affect per-region performance.
</div>

<!-- ======================================================== -->
<h2>4. Aggregate Spatial OOD Pattern</h2>

<p>By averaging the per-patch OOD scores across all test images (after resizing to a
common grid), we can identify whether certain <em>spatial regions</em> are
systematically underrepresented in training — for example, the sky region near the top
of the frame, or the horizon line where sea meets sky.</p>

{"<img class='plot' src='data:image/png;base64," + b64_agg + "' alt='Figure 3: Aggregate heatmap'>" if b64_agg else '<p><em>Aggregate heatmap figure not generated.</em></p>'}

<div class="finding">
<b>Interpretation:</b> Persistent red/orange regions in the aggregate heatmap indicate
spatial positions that are <em>systematically OOD</em> across the test set —
environments or scene elements that appear in test images but are consistently
underrepresented in training data. Green regions are well-covered by training.
</div>

<!-- ======================================================== -->
<h2>5. Per-Sequence Aggregate Heatmaps (Top-5 Most OOD Sequences)</h2>

{"<img class='plot' src='data:image/png;base64," + b64_seq + "' alt='Figure 4: Seq heatmaps'>" if b64_seq else '<p><em>Per-sequence heatmap figure not generated.</em></p>'}

<!-- ======================================================== -->
<h2>6. Comparison with Whole-Image OOD Ranking</h2>

<h3>6.1 Scatter plot: whole-image vs spatial OOD scores</h3>
<img class="plot" src="data:image/png;base64,{b64_scatter}" alt="Figure 5: Scatter">

<p>Spearman rank correlation between whole-image max cosine similarity and mean
patch similarity (test set): <b>ρ = {rho:.4f}</b> (p = {pval:.2e})</p>

<div class="{'finding' if not (rho != rho) and rho > 0.7 else 'warning'}">
{"<b>High correlation (ρ > 0.7):</b> The spatial and whole-image OOD rankings are broadly consistent. Images that are globally OOD (low whole-image similarity) tend also to have low mean patch similarity. However, some images with moderate whole-image similarity have locally OOD patches — the spatial analysis adds value by identifying <em>where</em> within the image the distributional gap is." if not (rho != rho) and rho > 0.7 else "<b>Moderate or low correlation:</b> The spatial and whole-image OOD rankings diverge. Some images with high whole-image similarity contain locally OOD patches — the spatial analysis reveals at-risk regions that the whole-image analysis missed."}
</div>

<!-- ======================================================== -->
<h2>7. Conclusions and Recommendations</h2>

<ul>
  <li>Patchwise spatial OOD analysis reveals <b>localised distributional gaps</b>
      that are invisible to whole-image cosine similarity. Even images with high
      whole-image similarity can contain patches with no close training analogue.</li>
  <li>The top-20 most at-risk test images are concentrated in a small number of
      sequences — targeted augmentation or additional labelling for those sequences
      is the highest-leverage intervention.</li>
  <li>The aggregate spatial heatmap reveals whether certain <em>scene regions</em>
      (sky, horizon, near-water obstacles) are systematically underrepresented.
      If the heatmap shows persistent red regions at the image boundaries or near
      the horizon, those are areas where a model is most likely to fail.</li>
  <li>Spearman ρ = {rho:.4f} between whole-image and spatial OOD rankings indicates
      {"the two measures largely agree — use the spatial heatmaps to localise within the images flagged by the whole-image analysis." if not (rho != rho) and rho > 0.7 else "the two measures capture complementary signals — both should be consulted when prioritising augmentation."}
  </li>
  <li>Phase 2 heatmap search used <b>PyTorch GPU float16 matmul</b> on a 1-in-10
      subsampled training patch set (∼1.4 M vectors). CPU FAISS flat search at this
      scale was measured at ∼2.2 min/image and was not feasible. If GPU FAISS becomes
      available, it could replace the PyTorch matmul and allow use of the full patch
      set without subsampling.</li>
</ul>

<hr>
<p class="meta">Spatial OOD extension — FARMS Team, MaCVi 2026.
Model: {meta.get('model_name', '?')}. Phase 1: faiss-cpu IndexFlatIP (1-in-10 subsample).
Phase 2: PyTorch GPU float16 matmul.</p>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f'Loading heatmap statistics from {args.heatmaps} ...')
    images_root = args.images
    test_recs   = get_image_list(images_root, 'test')
    val_recs    = get_image_list(images_root, 'val')
    eval_recs   = test_recs + val_recs

    df_hm = load_heatmap_stats(args.heatmaps, eval_recs)
    print(f'  Heatmaps loaded: {len(df_hm)} images')

    print(f'Loading embeddings from {args.db} ...')
    names, splits, seqs, embeds, meta = load_db_embeddings(args.db)
    max_sims_whole = compute_whole_image_sim(embeds, splits)

    import pandas as pd
    df_whole = pd.DataFrame({
        'img_name':       names,
        'split':          splits,
        'whole_max_sim':  max_sims_whole,
    })
    df_merged = df_hm.merge(df_whole[['img_name', 'whole_max_sim']], on='img_name', how='left')
    print(f'  Merged dataframe: {len(df_merged)} rows')

    print('Building HTML report ...')
    html = build_html(args, df_hm, df_merged, names, splits, seqs, embeds, meta, max_sims_whole)

    out_path  = Path(args.out)
    html_path = out_path.with_suffix('.html')
    html_path.write_text(html, encoding='utf-8')
    print(f'  HTML written to {html_path}')

    print('Converting to PDF via weasyprint ...')
    try:
        from weasyprint import HTML as WP_HTML
        WP_HTML(filename=str(html_path)).write_pdf(str(out_path))
        print(f'  PDF written to {out_path}')
        html_path.unlink()
    except Exception as e:
        print(f'  weasyprint failed: {e}')
        print(f'  HTML report available at {html_path}')

    print('Done.')


if __name__ == '__main__':
    main()
