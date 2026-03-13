"""
Export segnext_t_uperhead argmax ONNX from Phase 2 artifact checkpoint.

This is the only model not already present in /home/farms/data/onnx_exports/.
All other Phase 2/2.5 models already have pre-exported ONNX files there.

Run with macvi2 conda env active (mmseg 1.2.2, PyTorch 2.1.2+cu121):
    conda run --no-capture-output -n macvi2 \\
        python challenges/common/lars-analysis/failure_analysis_embedded_768x384_v2/export_onnx.py

Output:
    <this_dir>/onnx_exports/segnext_t_uperhead/segnext_t_uperhead_argmax.onnx
    (written to a writable path inside the repo since /home/farms/data/ is read-only)
"""

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT    = Path(__file__).resolve().parents[4]
CHALLENGE    = REPO_ROOT / "challenges" / "embedded-semantic-segmentation"
PHASE2_DIR   = CHALLENGE / "phase2"
ARTIFACT_DIR = PHASE2_DIR / "artifacts" / "segnext_t_uperhead"
CONFIG_PATH  = PHASE2_DIR / "configs" / "segnext_t_uperhead_768x384_lars_phase2.py"
CKPT_PATH    = ARTIFACT_DIR / "best_mIoU_iter_4250.pth"
OUT_ONNX     = Path(__file__).resolve().parent / "onnx_exports" / "segnext_t_uperhead" / "segnext_t_uperhead_argmax.onnx"

MODEL_H, MODEL_W = 384, 768
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def convert_syncbn(module):
    """Recursively replace SyncBatchNorm with BatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features, child.eps, child.momentum,
                                child.affine, child.track_running_stats)
            if child.affine:
                bn.weight.data = child.weight.data.clone()
                bn.bias.data   = child.bias.data.clone()
            bn.running_mean = child.running_mean.clone()
            bn.running_var  = child.running_var.clone()
            setattr(module, name, bn)
        else:
            convert_syncbn(child)
    return module


class SegWrapperArgmax(nn.Module):
    """backbone → decode_head → bilinear upsample → argmax → (1,1,H,W) int64."""
    def __init__(self, model, align_corners=False):
        super().__init__()
        self.model = model
        self.align_corners = align_corners

    def forward(self, img):
        feats  = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = F.interpolate(logits, size=(MODEL_H, MODEL_W),
                               mode="bilinear", align_corners=self.align_corners)
        return torch.argmax(logits, dim=1).unsqueeze(1)


def main():
    print(f"Device: {DEVICE}")

    if not CKPT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}")
        print("Run 'git lfs pull' to fetch the artifact checkpoint.")
        sys.exit(1)

    if OUT_ONNX.exists():
        print(f"ONNX already exists at {OUT_ONNX} — nothing to do.")
        print("Delete it and re-run to force re-export.")
        return

    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

    # Load model via mmseg API
    sys.path.insert(0, str(REPO_ROOT))
    from mmseg.apis import init_model

    print(f"Loading segnext_t_uperhead from {CKPT_PATH} ...")
    model = init_model(str(CONFIG_PATH), str(CKPT_PATH), device=DEVICE)
    model.eval()
    # segnext_t_uperhead uses SyncBN (due to UPerHead with BN) — convert for single-GPU export
    convert_syncbn(model)

    wrapper = SegWrapperArgmax(model, align_corners=False).to(DEVICE)
    wrapper.eval()

    # Dummy input — use model dtype (fp32 or fp16), avoid all-zeros for sanity check
    model_dtype = next(model.parameters()).dtype
    rng = np.random.default_rng(0)
    dummy_np = (rng.standard_normal((1, 3, MODEL_H, MODEL_W)).astype(np.float32)
                * 0.5 + 0.5)
    dummy = torch.from_numpy(dummy_np).to(device=DEVICE, dtype=model_dtype)

    # Sanity check: forward pass
    with torch.no_grad():
        out = wrapper(dummy)
    assert out.shape == (1, 1, MODEL_H, MODEL_W), f"Unexpected output shape: {out.shape}"
    assert out.dtype == torch.int64, f"Unexpected output dtype: {out.dtype}"
    print(f"  Forward pass OK — output shape {out.shape}, dtype {out.dtype}, model dtype {model_dtype}")

    # Export ONNX
    tmp_onnx = OUT_ONNX.with_suffix(".tmp.onnx")
    print(f"Exporting ONNX to {OUT_ONNX} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(tmp_onnx),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    shutil.move(str(tmp_onnx), str(OUT_ONNX))
    print(f"  Saved: {OUT_ONNX}  ({OUT_ONNX.stat().st_size / 1e6:.1f} MB)")

    # Sanity-check: load ONNX and compare argmax outputs (use same random input)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_np})[0]
    pt_out  = out.cpu().numpy()
    agree = np.mean(ort_out == pt_out) * 100
    print(f"  PyTorch vs ONNX pixel agreement on dummy input: {agree:.2f}%")
    if agree < 99.0:
        print("  WARNING: agreement < 99% — check model export!")
    else:
        print("  Export verified OK.")


if __name__ == "__main__":
    main()
