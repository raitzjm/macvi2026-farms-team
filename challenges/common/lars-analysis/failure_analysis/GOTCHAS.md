# GOTCHAS — Phase 1 Failure Analysis

Known issues, fixes, and methodology decisions encountered during implementation.

---

## 1. mmcv reflect vs symmetric — `cv2.BORDER_REFLECT_101`

**Symptom:** Confusion about which OpenCV border type matches mmseg's
`padding_mode='reflect'`.

**Root cause:** In `mmcv.impad`:
```python
border_type = {
    'reflect':   cv2.BORDER_REFLECT_101,  # does NOT repeat edge pixel
    'symmetric': cv2.BORDER_REFLECT,      # repeats edge pixel
}
```
mmcv's `'reflect'` maps to `cv2.BORDER_REFLECT_101`, not `cv2.BORDER_REFLECT`.

**Resolution:** If using reflect padding, use `cv2.BORDER_REFLECT_101`.

---

## 2. mmseg Pad pads bottom-right, NOT centered

**Symptom:** Assumption that training preprocessing centers the image in 384×768.

**Root cause:** `mmcv.impad(img, shape=(H, W))` computes:
```python
padding = (0, 0, W - img.shape[1], H - img.shape[0])  # (left, top, right, bottom)
```
This pads to the right and bottom only, not centered.

**Resolution:** For this analysis we use simple resize (PIL to 768×384) matching
`export_and_eval.py`, which bypasses this issue entirely and ensures consistency with
Phase 1 reported metrics.

---

## 3. Preprocessing consistency — use `export_and_eval.py` approach

**Decision:** Use simple PIL resize to (768, 384) rather than keep_ratio + reflect pad.

**Rationale:** The Phase 1 mIoU and F1 numbers were computed in `export_and_eval.py`
using this simple resize. Using the same preprocessing ensures per-image metrics
aggregate consistently with the reported values. Confirmed: our per-image mean mIoU
values match within ~2% of Phase 1 report.

**Note:** The CLAUDE.md issue description mentions keep_ratio + reflect pad (matching
the challenge server spec). However, using that preprocessing would give metrics
inconsistent with Phase 1 report. For failure analysis purposes, consistency with
the established baseline is more important.

---

## 4. kaleido 1.2.0 requires Chrome for PNG export

**Symptom:** `plotly.io.write_image(fig, 'out.png')` raises:
```
KaleioError: Kaleido requires Google Chrome to be installed.
```

**Root cause:** kaleido 1.2.0 uses Chrome headless for rendering (not the old
go-based binary approach of kaleido 0.x).

**Resolution:** Use matplotlib for committed PNG figures. Plotly is used only for
interactive inline rendering in the notebook.

---

## 5. weasyprint requires pango/cairo system libraries

**Symptom:** `import weasyprint` raises `OSError: cannot load library 'libpango-1.0-0'`.

**Resolution:** Install system libs:
```bash
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info
```

---

## 6. All 198 val images contain obstacle pixels

All 198 val images have at least one GT obstacle pixel (class=0). No NaN entries
in obstacle_iou — the column is fully populated for all images.

**Implication:** mIoU always averages over all 3 classes.

---

## 7. Failure threshold choice

Threshold 0.1 gives only 0 all-3-model failures (very strict, catches only
completely missed obstacles). Threshold 0.5 gives 6 all-3-model failures and
14 total failure images — more informative for cross-model overlap analysis.

We use **threshold = 0.5** for the Venn diagram and **worst 20 images** for the
failure case gallery (sorted by obstacle IoU ascending).
