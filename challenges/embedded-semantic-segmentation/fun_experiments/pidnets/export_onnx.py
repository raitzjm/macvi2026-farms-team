"""
Export PIDNet-S to ONNX (mmseg 1.2.2 / mmcv 2.1.0 / PyTorch 2.1.2).

Usage:
    python export_onnx.py [--ckpt path/to/checkpoint.pth] [--out output.onnx]

Defaults to best_mIoU_iter_2000.pth in the same directory.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'configs', 'pidnets_768x384_lars_fun.py')
DEFAULT_CKPT = os.path.join(SCRIPT_DIR, 'best_mIoU_iter_10000.pth')
DEFAULT_OUT         = os.path.join(SCRIPT_DIR, 'pidnets_768x384.onnx')
DEFAULT_OUT_ARGMAX  = os.path.join(SCRIPT_DIR, 'pidnets_768x384_argmax.onnx')

OUT_H, OUT_W = 384, 768


class SegWrapper(nn.Module):
    """Thin wrapper: raw image tensor → full-res logits.

    Bypasses img_metas / data_samples entirely so torch.onnx.export works
    cleanly with PyTorch 2.1.x.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        # backbone returns single tensor (I-branch) in eval mode
        feats = self.model.backbone(img)
        # decode head returns single tensor in eval mode
        logits = self.model.decode_head(feats)
        # upsample to full resolution (same align_corners as PIDHead.predict)
        logits = F.interpolate(
            logits, size=(OUT_H, OUT_W), mode='bilinear', align_corners=True)
        return logits


class SegWrapperArgmax(nn.Module):
    """Same as SegWrapper but appends argmax — produces 1×1×H×W int64 class map.

    This is the format required by the challenge submission pipeline.
    Uses ArgMax(keepdims=0) + Unsqueeze rather than ArgMax(keepdims=1) because
    SNPE does not support keepdims=1 on ArgMax — matches the working PIDNet-L
    export pattern exactly.
    (SegWrapper logits output kept separately for local evaluation use.)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        feats = self.model.backbone(img)
        logits = self.model.decode_head(feats)
        logits = F.interpolate(
            logits, size=(OUT_H, OUT_W), mode='bilinear', align_corners=True)
        # keepdim=False → ArgMax(keepdims=0) in ONNX, then unsqueeze adds dim back
        # SNPE requires this two-step pattern; keepdims=1 is not supported
        return torch.argmax(logits, dim=1).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--out',  default=DEFAULT_OUT)
    parser.add_argument('--out-argmax', default=DEFAULT_OUT_ARGMAX)
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from mmseg.apis import init_model
    print(f'Config:     {CONFIG}')
    print(f'Checkpoint: {args.ckpt}')
    # Use CPU — training process already owns the GPU context
    model = init_model(CONFIG, args.ckpt, device='cpu')
    model.eval()

    wrapper        = SegWrapper(model).cpu().eval()
    wrapper_argmax = SegWrapperArgmax(model).cpu().eval()

    dummy = torch.randn(1, 3, OUT_H, OUT_W, device='cpu')
    with torch.no_grad():
        out        = wrapper(dummy)
        out_argmax = wrapper_argmax(dummy)
    print(f'PyTorch logits shape : {tuple(out.shape)}')         # (1, 3, 384, 768)
    print(f'PyTorch argmax shape : {tuple(out_argmax.shape)}')  # (1, 1, 384, 768)

    import onnxruntime as ort
    import numpy as np

    # ── Export logits (kept for local evaluation) ─────────────────────────────
    print(f'\nExporting logits to: {args.out}')
    torch.onnx.export(
        wrapper, dummy, args.out,
        opset_version=11,
        input_names=['input'], output_names=['output'],
    )
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {'input': dummy.numpy()})[0]
    max_diff = np.abs(out.numpy() - onnx_out).max()
    print(f'PyTorch vs ONNX max diff: {max_diff:.2e}')
    print('Logits export PASSED ✓' if max_diff < 1e-3 else 'WARNING: large diff')

    # ── Export argmax (challenge submission format) ───────────────────────────
    print(f'\nExporting argmax to: {args.out_argmax}')
    torch.onnx.export(
        wrapper_argmax, dummy, args.out_argmax,
        opset_version=13,          # challenge server (SNPE) requires opset 13
        input_names=['input'], output_names=['output'],
    )
    sess_am = ort.InferenceSession(args.out_argmax, providers=['CPUExecutionProvider'])
    onnx_am = sess_am.run(None, {'input': dummy.numpy()})[0]
    pt_am   = out_argmax.numpy()
    match   = np.array_equal(pt_am, onnx_am)
    print(f'Output shape : {onnx_am.shape}')   # expect (1, 1, 384, 768)
    print(f'Output dtype : {onnx_am.dtype}')   # expect int64
    print('Argmax export PASSED ✓' if match else 'WARNING: mismatch')


if __name__ == '__main__':
    sys.path.insert(0, '/home/farms/macvi2026-farms-team')
    main()
