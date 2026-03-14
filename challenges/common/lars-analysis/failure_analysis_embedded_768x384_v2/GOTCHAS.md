# GOTCHAS — Phase 2/2.5 Failure Analysis

Known issues, fixes, and methodology decisions encountered during implementation.

---

## 1. `/home/farms/data/` is mounted read-only

**Symptom:** `export_onnx.py` originally wrote the segnext_t_uperhead ONNX to
`/home/farms/data/onnx_exports/phase2/segnext_t_uperhead/` — this fails with a
permission error since the data mount is read-only.

**Resolution:** Output path changed to a local writable path inside the repo:
`failure_analysis_embedded_768x384_v2/onnx_exports/segnext_t_uperhead/segnext_t_uperhead_argmax.onnx`

The `*.onnx` pattern is gitignored at root, so this file is not committed.
`infer_and_eval.py` and `build_report.py` look for it at this local path.

---

## 2. `pandas` not installed in macvi env

**Symptom:** `ModuleNotFoundError: No module named 'pandas'` when running
`infer_and_eval.py` in the macvi env.

**Resolution:** `pip install pandas` installed pandas 3.0.1. See `setup.md`.

---

## 3. CUDA ORT provider unavailable despite GPU being present

**Symptom:** `infer_and_eval.py` selected `CPUExecutionProvider` even though the
machine has an RTX4050. Log shows CUDA provider not in available list.

**Root cause:** `libcudnn.so.9` was not on the library path in that session.
`onnxruntime-gpu` was installed but cudnn was not linked/found.

**Resolution for future runs:** Ensure `onnxruntime-gpu` is installed and
`libcudnn.so.9` is discoverable. In a properly configured container session
with the GPU driver mounted, CUDA should be available automatically.

**Impact on results:** None — ORT CPU and CUDA produce bit-identical results.
Only throughput is affected (CPU inference is slower).

---

## 4. segnext_t_uperhead uses SyncBN via UPerHead

**Symptom:** Uncertainty about whether `convert_syncbn()` is needed for
segnext_t_uperhead before ONNX export.

**Resolution:** UPerHead uses BatchNorm internally. When trained with DDP
(multi-GPU), mmseg automatically replaces BN with SyncBN. The saved checkpoint
contains SyncBN weights. `convert_syncbn()` must be called before export to
replace all `SyncBatchNorm` with standard `BatchNorm2d` for single-device ONNX
export. Omitting this causes a shape error at export time.

---

## 5. weasyprint requires pango/cairo system libraries

**Symptom:** `import weasyprint` raises `OSError: cannot load library 'libpango-1.0-0'`.

**Resolution:** Install system libs:
```bash
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info
```

---

## 6. Pairwise co-failure heatmap replaces 3-way Venn

With 9 models, a 3-circle Venn diagram is not feasible. The failure overlap
figure uses two panels instead:
- **Breadth distribution bar chart**: how many images fail in exactly K of the 9 models
- **Pairwise co-failure heatmap**: 9×9 matrix of fraction of val images where both models fail

---

## 7. All 198 val images contain obstacle pixels

Same as v1 — all images have at least one GT obstacle pixel (class=0). mIoU
always averages over all 3 classes. No NaN entries in any `*_obstacle_iou` column.

---

## 8. Failure threshold

**threshold = 0.5** used for breadth/co-failure analysis and worst-case galleries
(sorted by obstacle IoU ascending). Consistent with v1.
