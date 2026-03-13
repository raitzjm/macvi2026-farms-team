# Setup Log — Phase 2/2.5 Failure Analysis

Environment: `macvi` conda env (Python 3.11, macvi-farms:latest container)

## New packages installed (2026-03-11)

The `macvi` env was missing `pandas` and `weasyprint`. `onnxruntime-gpu` was also
installed to enable CUDA inference. The following were installed:

```bash
# Python packages
pip install onnxruntime-gpu pandas weasyprint

# System libraries for weasyprint (requires pango/cairo)
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info
```

Versions confirmed working:
- onnxruntime-gpu 1.24.3
- pandas 3.0.1
- weasyprint 68.1

## Step 1 — segnext_t_uperhead ONNX export (macvi2 env)

Requires `macvi2` conda env (mmseg 1.2.2, PyTorch 2.1.2+cu121):

```bash
conda run --no-capture-output -n macvi2 \
    python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/export_onnx.py
```

Output: `failure_analysis_embedded_768x384_v2/onnx_exports/segnext_t_uperhead/segnext_t_uperhead_argmax.onnx`
(45.5 MB, PyTorch/ONNX pixel agreement: 99.93%)

## Notes

- `plotly` / `kaleido` not needed — no interactive notebook for v2 (PDF report is the primary output).
- `matplotlib` 3.10.8 already available in macvi env.
- CUDA ORT inference requires `libcudnn.so.9` to be discoverable. If CUDA is unavailable
  in the session, inference falls back to CPU automatically — results are bit-identical.

## Dockerfile update needed

Add to Dockerfile (macvi-farms image) to avoid manual installs on each container rebuild:

```dockerfile
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime-gpu pandas weasyprint
```
