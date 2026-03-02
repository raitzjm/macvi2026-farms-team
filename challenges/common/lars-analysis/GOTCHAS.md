# LaRS Embedding Analysis — Issues & Gotchas Log

A running record of failures, unexpected blockers, and methodology changes made
in response to something encountered during the analysis. For package installs
and environment steps, see `setup.md`.

---

## 1. Notebook SyntaxError from f-string line splitting (nbformat)

**What failed:** Kernel crashed on first execution of `analysis.ipynb` with
`SyntaxError: unterminated string literal`.

**Root cause:** When building the notebook programmatically with `nbformat`,
cell source is stored as a list of strings (one per line). An f-string containing
`\n` inside it (e.g. `print(f"\n...")`) was split at the wrong boundary, producing
an unterminated string literal in the resulting cell.

**Fix:** Rewrote the notebook builder (`build_notebook.py`) to use string
concatenation instead of f-strings for all cell source strings, avoiding the
escape-sequence interaction with nbformat's line list.

---

## 2. Missing packages discovered at RADIO-H import time

**What failed:** Kernel crashed when the embedding cell imported `AutoModel` from
`transformers`. `nvidia/C-RADIOv3-H` silently imports `timm`, `einops`, and
`open-clip-torch` at load time — none were in the initial environment spec and
none are listed in the model card's documented requirements.

**Fix:** Installed mid-session and added to `setup.md`.

---

## 3. CPU FAISS flat search too slow — switched to PyTorch GPU matmul

**What changed and why:** The original Phase 2 implementation queried the 1.4 M
vector FAISS index per image using `cpu_idx.search()`. Benchmarked at ~2.2
min/image → ~51 hours total for 1,401 images — completely unusable.

**Root cause:** CPU FAISS flat search is O(N × D) per query, dominated by memory
bandwidth. At N=1,408,492 and D=1,280 with hundreds of patches per image, the
CPU simply cannot keep up.

**Methodology change:** Dropped FAISS for Phase 2 entirely. Loaded the 1.4 M
subsampled training vectors to GPU as float16 (~3.6 GB VRAM) and replaced the
per-image FAISS call with a single PyTorch matmul:
```python
sims = (sp_gpu @ train_gpu.T).max(dim=1).values
```
Result: ~13–17 ms/image → ~17 minutes total. FAISS is now used only for Phase 1
(index construction / persistence), not for heatmap scoring.

---

## 4. CUDA OOM for one high-resolution image — chunked fallback added

**What failed:** `inhouse_seq_110_00149` raised `torch.cuda.OutOfMemoryError`
during Phase 2. The image has 15,960 RADIO-H spatial tokens; the single-shot
matmul intermediate `(15960 × 1,408,492)` float16 would require ~45 GB, exceeding
the ~37 GB of free VRAM at runtime. The original bare `except Exception` caught
it, logged a warning, and skipped the image — 1,400/1,401 heatmaps were produced.

**Fix:** Added an inner `except torch.cuda.OutOfMemoryError` that clears the CUDA
cache and retries the matmul in 512-row query chunks (~1.44 GB per chunk). The
result is bit-identical to the fast path — per-row max is independent of other
rows, so chunking does not change the output. The image processed successfully in
~0.8 s via the chunked path. All 1,401 heatmaps are now present.

**Lesson:** Catch `OutOfMemoryError` explicitly rather than letting it fall into a
generic `except Exception` that silently skips work.

---

## 5. Wrong methodology description in the spatial PDF report

**What was wrong:** A self-review (independent clean-context agent) found that
`generate_spatial_report.py` described the FAISS index as `IndexIVFFlat` with
nlist=4096 / nprobe=64, and implied Phase 2 used FAISS CPU search — both
incorrect (actual: `IndexFlatIP`; Phase 2 uses PyTorch GPU matmul). Additionally:
- The 10% training patch subsample was never disclosed in the PDF.
- One skipped image was not mentioned.
- Per-image local colorscale was unexplained.
- "Exactly where the model is most likely to fail" overclaimed the result.

**Fix:** Full rewrite of Section 2.2 in the report generator; added subsample
caveat, OOM recovery note, colorscale warning, and softened conclusion language.
PDF regenerated.

**Lesson:** Report prose drifts from implementation when the implementation
changes post-draft (the PyTorch GPU switch happened after the report was first
written). A separate review pass by a clean-context agent caught all of this.
