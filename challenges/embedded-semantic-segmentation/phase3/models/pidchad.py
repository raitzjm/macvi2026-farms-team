"""PIDChad-S — PIDNet-S extended with AFFM + FPPM (EXP-021).

Architecture changes vs PIDNet-S:
  1. AFFM  — Attention-based Feature Fusion Module: I-branch context gates the
             P-branch before LightBag fusion. Suppresses water-reflection FPs
             by letting global context veto local texture predictions.
  2. FPPM  — Fast Pyramid Pooling Module: parallel AvgPool2d branches at grid
             sizes (1, 2, 3, 6, 12) replaces the cascaded PAPPM. Wider global
             anchor (scale-12) improves waterline estimation (WE_IoU).

Both modules use only 1×1 Conv, BN, ReLU/Sigmoid, AvgPool2d —
all proven INT8-safe ops (no LayerNorm, GELU, or Softmax).

ONNX note: FPPM uses explicit nn.AvgPool2d (not F.adaptive_avg_pool2d) with
kernel sizes precomputed for fixed 384×768 input (feature map = 12×24 at 1/32).
This avoids the PyTorch 2.1.2 ONNX symbolic failure for non-factor output sizes.

INT8 safety check is mandatory before full training (Step 4 in EXP-021 plan).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import ModuleList
from torch import Tensor

from mmseg.models.backbones.pidnet import PIDNet, LightBag
from mmseg.models.decode_heads.pid_head import PIDHead
from mmseg.registry import MODELS


# ── AFFM — Attention-based Feature Fusion Module ──────────────────────────────

class AFFM(nn.Module):
    """Attention-based Feature Fusion Module.

    Uses I-branch (context) to generate a spatial attention map that gates the
    P-branch (detail) before LightBag fusion. Suppresses water-reflection false
    positives: when global context says "open water here", local texture detail
    is down-weighted.

    Mechanism (applied BEFORE LightBag):
        attn = Sigmoid(conv1x1(BN(ReLU(conv1x1(x_i_upsampled)))))
        x_p_gated = x_p * attn          # context veto on detail
        [then LightBag(x_p_gated, x_i, x_d) as normal]

    Args:
        channels (int): Number of channels for both P and I branches
                        (channels×4 = 128 for PIDNet-S).
        norm_cfg (dict): Normalization config. Default: SyncBN.
    """

    def __init__(self,
                 channels: int,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # Two 1×1 convs: reduce → project attention logit
        self.attn_reduce = ConvModule(
            channels, channels // 4, kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.attn_proj = nn.Conv2d(channels // 4, channels, kernel_size=1, bias=True)

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        """Gate P-branch with context attention from I-branch.

        Args:
            x_p: P-branch feature map, shape (B, C, H, W).
            x_i: I-branch feature map, same spatial size as x_p.

        Returns:
            x_p_gated: Context-gated P-branch, same shape as x_p.
        """
        # Generate spatial attention from I-branch context
        attn = self.attn_reduce(x_i)           # (B, C//4, H, W)
        attn = self.attn_proj(attn)             # (B, C, H, W)
        attn = torch.sigmoid(attn)              # → [0, 1] gate
        return x_p * attn                       # gate detail with context


# ── FPPM — Fast Pyramid Pooling Module ────────────────────────────────────────

class FPPM(nn.Module):
    """Fast Pyramid Pooling Module (parallel, replaces cascaded PAPPM).

    Independent parallel branches at multiple grid scales, concatenated and
    compressed. Extended scale list (1, 2, 3, 6, 12) vs. PAPPM's (5 scales
    including global) provides a stronger vertical anchor for waterline
    estimation.

    All ops are INT8-safe: AvgPool2d + Conv1×1 + BN + ReLU.

    ONNX compatibility: uses explicit nn.AvgPool2d (not F.adaptive_avg_pool2d)
    with kernel sizes precomputed for feature_hw. PyTorch 2.1.2's ONNX symbolic
    for adaptive_avg_pool2d fails when output size does not evenly divide input
    size (e.g. scale=12 → output (12,12) from input (12,24), giving kernel
    w_stride=2 which ONNX cannot express as symmetric GlobalAveragePool).

    Args:
        in_channels (int): Input feature channels (channels×16 = 512 for PIDNet-S).
        branch_channels (int): Channels per pyramid branch (ppm_channels = 96).
        out_channels (int): Output channels (channels×4 = 128 for PIDNet-S).
        scales (list[int]): Grid output sizes per branch. Default: [1, 2, 3, 6, 12].
        feature_hw (tuple[int,int]): Spatial size of x at FPPM input (H, W).
                                     For 384×768 at stride 32: (12, 24).
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 scales: List[int] = [1, 2, 3, 6, 12],
                 feature_hw: Tuple[int, int] = (12, 24),
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True),
                 act_cfg: Dict = dict(type='ReLU', inplace=True)):
        super().__init__()
        self.scales = scales
        feat_h, feat_w = feature_hw

        # Build explicit poolers: AdaptiveAvgPool2d for scale=1 (→ GlobalAveragePool in ONNX),
        # explicit AvgPool2d for all others (kernel = feat // scale, evenly divides by design).
        self.poolers = ModuleList()
        for s in scales:
            if s == 1:
                self.poolers.append(nn.AdaptiveAvgPool2d(1))
            else:
                kh, kw = feat_h // s, feat_w // s
                self.poolers.append(nn.AvgPool2d(kernel_size=(kh, kw), stride=(kh, kw)))

        # One conv branch per scale: BN → ReLU → conv1×1 (pre-activation)
        self.branches = ModuleList()
        for _ in scales:
            self.branches.append(
                ConvModule(
                    in_channels, branch_channels, kernel_size=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                    order=('norm', 'act', 'conv')))

        # Compress all branches + shortcut → out_channels
        total_ch = branch_channels * len(scales) + in_channels
        self.compress = ConvModule(
            total_ch, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))

        # Shortcut: 1×1 conv on input (same as PAPPM/DAPPM)
        self.shortcut = ConvModule(
            in_channels, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        branch_outs = []
        for pooler, branch in zip(self.poolers, self.branches):
            pooled = pooler(x)
            feat = branch(pooled)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            branch_outs.append(feat)
        concat = torch.cat(branch_outs + [x], dim=1)
        return self.compress(concat) + self.shortcut(x)


# ── PIDChadNet — PIDNet-S with FPPM + AFFM ────────────────────────────────────

@MODELS.register_module()
class PIDChadNet(PIDNet):
    """PIDNet-S backbone extended with AFFM + FPPM.

    Structural changes:
      - FPPM replaces PAPPM (self.spp): parallel pooling at scales [1,2,3,6,12]
        for stronger waterline global anchor.
      - AFFM added before dfm (LightBag): I-branch context gates P-branch
        to suppress water-reflection false positives.

    All pretrained PIDNet-S weights load unchanged (FPPM and AFFM init fresh).
    Only applicable to num_stem_blocks=2 (PIDNet-S/M variant using LightBag).

    Args:
        fppm_scales (list[int]): Grid scales for FPPM.
                                 Default: [1, 2, 3, 6, 12].
        All other args: inherited from PIDNet.
    """

    def __init__(self,
                 fppm_scales: List[int] = [1, 2, 3, 6, 12],
                 **kwargs):
        # Call PIDNet.__init__ — this creates self.spp (PAPPM) and self.dfm (LightBag)
        super().__init__(**kwargs)

        # Extract channel dims from the already-constructed backbone
        channels = kwargs.get('channels', 64)
        ppm_channels = kwargs.get('ppm_channels', 96)

        # Replace PAPPM with FPPM (same in/out channel signature).
        # feature_hw=(12,24): FPPM input is at stride 32 for 384×768 input.
        # Must match training resolution — do not change without recomputing kernel sizes.
        self.spp = FPPM(
            in_channels=channels * 16,
            branch_channels=ppm_channels,
            out_channels=channels * 4,
            scales=fppm_scales,
            feature_hw=(12, 24),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Add AFFM for P-branch gating before LightBag
        # P-branch and I-branch both have channels*4 at the dfm stage
        self.affm = AFFM(channels=channels * 4, norm_cfg=self.norm_cfg)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward identical to PIDNet except AFFM is applied before dfm."""
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))

        # FPPM (replaces PAPPM)
        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # AFFM: I-branch context gates P-branch (BEFORE LightBag)
        x_p = self.affm(x_p, x_i)

        # LightBag: boundary-aware fusion (unchanged)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out


# ── PIDChadHead — PIDHead alias (class_weight natively supported) ─────────────

@MODELS.register_module()
class PIDChadHead(PIDHead):
    """PIDHead subclass for PIDChad-S config identification.

    class_weight in CrossEntropyLoss and OhemCrossEntropy is natively
    supported in mmseg 1.2.2 — no wrapper needed.

    This class exists solely to allow the config to reference a distinct
    decode_head type matching PIDChadNet.
    """
    pass
