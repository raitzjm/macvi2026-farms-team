# LaRS Embedding Analysis — Issues & Gotchas Log

A running record of every non-obvious problem encountered during this analysis,
why it happened, and how it was resolved. Intended as a reference for anyone
reproducing or extending this work.

---

## 1. Notebook SyntaxError from f-string line splitting (nbformat)

**When:** Initial notebook construction.

**What happened:** When building the notebook programmatically using `nbformat`,
each cell's source was stored as a list of strings (one per line). An f-string
containing `\n` inside it (e.g. `print(f"\n...")`) was split across the list at
the wrong point, producing an unterminated string literal that caused a
`SyntaxError` at kernel start.

**Fix:** Rewrote the notebook builder (`build_notebook.py`) to use string
concatenation instead of f-strings for cell source strings, avoiding escape
sequences that interact badly with the list-splitting.

---

## 2. Missing packages at RADIO-H import time

**When:** First execution of the embedding extraction cell.

**What happened:** `nvidia/C-RADIOv3-H` imports `timm`, `einops`, and
`open-clip-torch` at import time. None of these were in the initial environment
spec; the kernel crashed on `from transformers import AutoModel`.

**Fix:** Installed mid-execution:
```bash
conda run -n lars-analysis pip install timm einops open-clip-torch
```
Added to `setup.md`.

---

## 3. `report.html` accidentally committed

**When:** After the first PDF generation run.

**What happened:** `weasyprint` writes an intermediate HTML file before
converting to PDF. This file was not in `.gitignore` at the time and was
committed as part of the analysis output commit.

**Fix:** `git rm challenges/common/lars-analysis/report.html` and added the
path to `.gitignore`. Similarly added `spatial_report.html` pre-emptively.

---

## 4. weasyprint PDF generation failure — missing system library

**When:** First call to `weasyprint` for PDF export.

**What happened:** `weasyprint` requires `libpango` for text rendering, which
is not present by default in the base Ubuntu 22.04 container image.

**Error:** `OSError: cannot load library 'libpango-1.0.so.0'`

**Fix:**
```bash
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0
```
Added to `setup.md`.

---

## 5. `matplotlib.colormaps` import error

**When:** Running `generate_report.py` for the first time.

**What happened:** The script imported `matplotlib.colormaps` as a top-level
module before calling `matplotlib.use("Agg")`. On matplotlib 3.10.8 this raises
an `AttributeError` because the colormaps registry is not accessible as a module
attribute before the backend is set.

**Fix:** Removed the separate `import matplotlib.colormaps` and used
`matplotlib.cm.get_cmap()` directly after `matplotlib.use("Agg")`.

---

## 6. CPU FAISS flat search too slow for Phase 2

**When:** Initial Phase 2 implementation in `spatial_ood.ipynb`.

**What happened:** Phase 2 computes, for each test/val image, the max cosine
similarity of each of its patches to any training patch. The first implementation
used `faiss_cpu.search()` on the 1.4 M-vector index. Benchmarked at ~2.2
min/image → ~51 hours total for 1,401 images. Completely unusable.

**Root cause:** CPU FAISS flat search is O(N × D) per query. At N=1,408,492
and D=1,280, with ~hundreds of patches per image, the compute is dominated by
memory bandwidth on CPU.

**Fix:** Loaded the 1.4 M subsampled training vectors to GPU as float16 (~3.6 GB
VRAM) and replaced FAISS search with a single PyTorch GPU matmul:
```python
sims = (sp_gpu @ train_gpu.T).max(dim=1).values
```
Result: ~13–17 ms/image → ~17 minutes total for 1,401 images.

**Note:** `faiss-gpu` is not available via pip for Python 3.11 / CUDA 12.1.
The conda-forge `faiss-gpu=1.9.0` package was attempted but corrupted the
environment's numpy install. CPU FAISS is used only for Phase 1 (index
construction). Phase 2 uses PyTorch GPU exclusively.

---

## 7. CUDA OOM for one high-resolution image — chunked fallback

**When:** Phase 2 heatmap generation, image `inhouse_seq_110_00149`.

**What happened:** This image had an unusually high resolution, producing 15,960
RADIO-H spatial tokens. The single-shot GPU matmul would require an intermediate
result matrix of shape `(15960, 1,408,492)` in float16 = ~44.9 GB — exceeding
available VRAM (~37 GB free at runtime).

**Error log:**
```
WARNING ERROR inhouse_seq_110_00149: CUDA out of memory.
Tried to allocate 41.87 GiB. GPU 0 has a total capacity of 47.43 GiB
of which 37.02 GiB is free.
```

**Initial handling:** The bare `except Exception` in the original code caught the
error, logged a WARNING, and skipped the image. 1,400/1,401 heatmaps were
produced.

**Fix (v2):** Added an inner `except torch.cuda.OutOfMemoryError` block that:
1. Clears the CUDA cache.
2. Retries the matmul in 512-row query chunks:
   `(sp_n[i:i+512] @ train_gpu.T).max(dim=1)`
3. Accumulates results into a numpy array.

The result is **bit-identical** to the fast path — per-row max is mathematically
independent of other rows. After the fix, the image processed successfully in
~0.8 seconds. All 1,401 heatmaps are now present.

**Lesson:** Always catch OOM separately from generic exceptions so the fallback
can be attempted rather than silently skipping the image.

---

## 8. Incorrect methodology description in the spatial PDF report

**When:** Post-implementation self-review (independent review agent).

**What happened:** `generate_spatial_report.py` described the FAISS index as
`IndexIVFFlat` with `nlist=4096 / nprobe=64`, and implied that Phase 2 heatmap
scoring used FAISS CPU search. Both were wrong:
- Actual Phase 1 index: `IndexFlatIP` (exact, no IVF approximation).
- Phase 2 scoring: PyTorch GPU matmul (FAISS not used at all for heatmaps).

Additionally, the 10% training patch subsample was undisclosed, one skipped image
was not mentioned, per-image local colorscale was unexplained, and the
interpretation box overclaimed ("exactly where the model is most likely to fail").

**Fix:** Full rewrite of Section 2.2, added subsample caveat box, OOM recovery
note, colorscale warning, softened conclusion language. PDF regenerated.

---

## 9. `analysis.log` captures only `nbconvert` output, not cell stdout

**When:** Reviewing the committed log file.

**What happened:** The `analysis.log` file only contains 2 lines (the
`[NbConvertApp]` conversion/write lines). The notebook's cell-level logging
(embedding extraction progress, t-SNE status, etc.) was printed to the kernel's
stdout during execution but was not redirected to the log file — `nbconvert`
does not capture cell stdout into a separate log by default.

The `spatial_ood.ipynb` notebook explicitly configures a `FileHandler` writing
to `spatial_ood.log`, which is why that log is detailed. `analysis.ipynb` did
not have an equivalent handler, so its progress output is only preserved in the
notebook cell outputs (visible in the `.ipynb` file itself).

**Lesson:** For long-running notebooks, add a `logging.FileHandler` at the top
of the setup cell to ensure execution progress is captured independently of the
notebook output.

---

## 10. cuML GPU t-SNE output differs from sklearn CPU t-SNE

**When:** Planning and validation phase.

**What happened:** cuML's GPU t-SNE produces different (but equivalent) 2D
coordinates compared to sklearn's CPU implementation. Both minimise the same
KL divergence objective, but t-SNE is non-deterministic (random initialisation)
and the GPU/CPU code paths have different numerical paths.

**Impact for this analysis:** The committed `tsne_results.npy` was produced by
cuML (KL divergence 0.239, completed in seconds). If the environment is
reproduced without cuML, sklearn's fallback will produce a qualitatively
similar but numerically different embedding — visual cluster structure will be
preserved but exact coordinates will differ.

**Documented in:** `setup.md`, notebook Section 3.
