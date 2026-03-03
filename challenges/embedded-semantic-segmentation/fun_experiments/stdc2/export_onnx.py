"""
Direct ONNX export for mmseg models, bypassing the img_metas PyTorch 2.1.x issue.
Wraps the model in a simple nn.Module that takes a raw image tensor.

Usage:
    python export_onnx_direct.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

WORK_DIR = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/outputs/fun_stdc2_768x384'
CONFIG   = '/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/configs/stdc2_768x384_lars_fun.py'
CKPT     = f'{WORK_DIR}/best_mIoU_iter_10000.pth'
OUTPUT   = f'{WORK_DIR}/stdc2_768x384.onnx'
H, W     = 384, 768


def convert_syncbn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features, child.eps, child.momentum,
                                child.affine, child.track_running_stats)
            if child.affine:
                bn.weight.data = child.weight.data.clone()
                bn.bias.data   = child.bias.data.clone()
            bn.running_mean = child.running_mean
            bn.running_var  = child.running_var
            setattr(module, name, bn)
        else:
            convert_syncbn(child)
    return module


class SegWrapper(nn.Module):
    """Takes (1, 3, H, W) float tensor, returns (1, num_classes, H, W) logits."""
    def __init__(self, model, out_h, out_w):
        super().__init__()
        self.model = model
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, img):
        feats = self.model.extract_feat(img)
        logits = self.model.decode_head.forward(feats)
        # Resize to input resolution (align_corners matches decode_head setting)
        logits = F.interpolate(logits, size=(self.out_h, self.out_w),
                               mode='bilinear', align_corners=False)
        return logits


def main():
    print(f'Config:     {CONFIG}')
    print(f'Checkpoint: {CKPT}')
    print(f'Output:     {OUTPUT}')

    cfg = mmcv.Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    cfg.model.train_cfg  = None

    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = convert_syncbn(model)
    load_checkpoint(model, CKPT, map_location='cpu')
    model.eval()

    wrapper = SegWrapper(model, H, W)
    wrapper.eval()

    dummy = torch.zeros(1, 3, H, W)

    print('Tracing model...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            OUTPUT,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            export_params=True,
        )
    print(f'Exported: {OUTPUT}')

    # Verify with onnxruntime
    print('Verifying with onnxruntime...')
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(OUTPUT, providers=['CPUExecutionProvider'])
    inp = np.zeros((1, 3, H, W), dtype=np.float32)
    out = sess.run(None, {'input': inp})[0]
    print(f'ONNX output shape: {out.shape}')  # expected: (1, 3, 384, 768)

    # Quick sanity: PyTorch vs ONNX outputs match
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()
    diff = np.abs(pt_out - out).max()
    print(f'Max abs diff PyTorch vs ONNX: {diff:.6f}')
    if diff < 1e-3:
        print('Verification PASSED')
    else:
        print('WARNING: outputs differ more than expected')


if __name__ == '__main__':
    main()
