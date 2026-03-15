"""SegChad — ConvNeXt-V2 backbone + 3-stream D/P/I decoder (EXP-022).

Architecture overview:
  Backbone: ConvNeXt-V2 Femto (timm, pretrained) → 4 feature stages
  D Branch: boundary refinement stream (1/4 res, 64 ch)
  P Branch: detail/parsing stream (1/8 res, 128 ch)
  I Branch: semantic context stream (1/8 res, 256 ch) with PAG fusion
  Boundary-Guided Fusion: D boundary gates semantic vs detail mix
  Decoder: lightweight bilinear upsampling to 1/1 resolution

Design decisions (EXP-022):
  - ConvNeXt-V2 Femto chosen over Atto: Femto has 5.2M params vs 3.7M for Atto.
    The 1.5M overhead is modest compared to the decoder, and Femto's wider channels
    [48, 96, 192, 384] provide better feature capacity for the 3-stream decoder.
    At 768x384, Femto runs ~15-20 it/s on A6000 — comfortable for 30 FPS on RVC4
    after INT8 quantization (ConvNeXt is highly efficient in INT8).
  - AdamW optimizer: standard for ConvNeXt architectures. With pretrained backbone
    and differential LR (backbone 0.1x), AdamW+cosine converges fast.
  - No explicit backbone freeze: differential LR (backbone 1e-5 vs head 1e-4)
    achieves similar effect with simpler implementation.

INT8 compatibility notes:
  - D/P/I branches and fusion: Conv, BN, ReLU, bilinear upsample, sigmoid, element-wise
    multiply/add — ALL INT8-safe.
  - PAG fusion: sigmoid(Conv * Conv) — element-wise ops, INT8-safe.
  - ConvNeXt-V2 backbone uses LayerNorm on channel dimension (not spatial). Channel-dim
    LayerNorm is typically INT8-safe on SNPE as it folds into per-channel affine scaling.
    GRN (Global Response Normalization) in V2 is element-wise multiply/divide — INT8-safe.
    HOWEVER: INT8 smoke test is mandatory before full training to validate this.
  - All spatial operations use fixed-size bilinear interpolation (no adaptive pooling
    with non-dividing sizes).

ONNX export notes:
  - forward() returns only segmentation logits at inference (no boundary head).
  - SegWrapperArgmax in export_and_eval.py calls backbone(img) then decode_head(feats).
  - All bilinear interpolations use align_corners=False and target known fixed sizes.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList


# ── Backbone ─────────────────────────────────────────────────────────────────

@MODELS.register_module()
class SegChadBackbone(nn.Module):
    """ConvNeXt-V2 backbone wrapper for SegChad using timm.

    Wraps a timm ConvNeXt-V2 model in features_only mode, returning 4 feature
    stages: F1 (1/4), F2 (1/8), F3 (1/16), F4 (1/32).

    Uses ConvNeXt-V2 Femto by default (channels [48, 96, 192, 384]).
    Recommended timm model: 'convnextv2_femto.fcmae_ft_in1k' (FCMAE pretrained,
    finetuned on ImageNet-1k). If available, 'convnextv2_femto.d1h_in1k'
    provides DINOv2 head-distilled weights for better features.

    Args:
        model_name (str): timm model identifier. Default: 'convnextv2_femto.fcmae_ft_in1k'.
        pretrained (bool): Load pretrained weights. Default: True.
        out_indices (tuple): Feature stage indices to extract. Default: (0, 1, 2, 3).
        init_cfg (dict, optional): Not used (timm handles init). Default: None.
    """

    def __init__(self,
                 model_name: str = 'convnextv2_femto.fcmae_ft_in1k',
                 pretrained: bool = True,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 init_cfg: Optional[dict] = None):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=list(out_indices))
        # Expose channel info for downstream modules
        self.feat_channels = self.backbone.feature_info.channels()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Extract multi-scale features.

        Returns:
            tuple[Tensor]: (F1, F2, F3, F4) at strides (4, 8, 16, 32).
                For Femto: channels [48, 96, 192, 384].
        """
        feats = self.backbone(x)
        return tuple(feats)


# ── PAG — Pixel Attention Guidance ───────────────────────────────────────────

class PAGFusion(nn.Module):
    """Pixel Attention Guidance module (INT8-safe).

    Fuses detail features (v_p) with context features (v_i) via a learned
    attention gate:
        gate = sigmoid(conv_p(v_p) * conv_i(v_i))          # gate_ch channels
        fused = gate * v_i + (1 - gate) * proj_p(v_p)      # i_channels

    When p_channels != i_channels, v_p is projected to i_channels before mixing.
    Gate is always computed at gate_ch channels (must equal i_channels for correct
    element-wise ops with v_i and proj_p).

    All ops: Conv1x1 + BN + sigmoid + element-wise mul/add — INT8-safe.

    Args:
        p_channels (int): Channels of the detail (P-like) input.
        i_channels (int): Channels of the context (I-like) input.
        gate_channels (int): Internal gate channels. MUST equal i_channels to avoid
            dimension mismatch in gate * v_i. Default: i_channels.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 p_channels: int,
                 i_channels: int,
                 gate_channels: Optional[int] = None,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # gate_ch must equal i_channels so gate * v_i is dimension-safe
        gate_ch = gate_channels if gate_channels is not None else i_channels
        self.conv_p = ConvModule(
            p_channels, gate_ch, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.conv_i = ConvModule(
            i_channels, gate_ch, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        # Project v_p to i_channels for residual mixing (no-op if same channels)
        if p_channels != i_channels:
            self.proj_p = ConvModule(
                p_channels, i_channels, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=None)
        else:
            self.proj_p = nn.Identity()

    def forward(self, v_p: Tensor, v_i: Tensor) -> Tensor:
        """Fuse detail (v_p) and context (v_i) via pixel attention gate.

        v_p and v_i must have the same spatial size.

        Returns:
            Tensor: Fused features, same spatial size, channels = i_channels.
        """
        gate = torch.sigmoid(self.conv_p(v_p) * self.conv_i(v_i))
        return gate * v_i + (1.0 - gate) * self.proj_p(v_p)


# ── D Branch — Boundary Refinement Stream ────────────────────────────────────

class DBranch(nn.Module):
    """D Branch: boundary refinement stream at 1/4 resolution.

    Processes F1 features through two Conv3x3-BN-ReLU blocks to produce
    boundary-aware features (D_feat) and a boundary prediction head.

    Args:
        in_channels (int): Input channels from F1 (e.g. 48 for Femto).
        mid_channels (int): Output feature channels. Default: 64.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int = 64,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels, mid_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.conv2 = ConvModule(
            mid_channels, mid_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        # Boundary prediction head: Conv3x3 → Conv1x1 → 1 channel (logits)
        self.boundary_conv = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.boundary_pred = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, f1: Tensor) -> Tuple[Tensor, Tensor]:
        """Process F1 features.

        Returns:
            d_feat: Boundary-aware features (B, mid_ch, H/4, W/4).
            boundary_logits: Raw boundary logits (B, 1, H/4, W/4).
        """
        x = self.conv1(f1)
        d_feat = self.conv2(x)
        bd = F.relu(self.boundary_conv(d_feat), inplace=True)
        boundary_logits = self.boundary_pred(bd)
        return d_feat, boundary_logits


# ── P Branch — Detail/Parsing Stream ─────────────────────────────────────────

class PBranch(nn.Module):
    """P Branch: detail parsing stream at 1/8 resolution.

    Fuses F2 and F3 features via projection + upsample + concat + refine.

    Args:
        f2_channels (int): Channels of F2 (e.g. 96 for Femto).
        f3_channels (int): Channels of F3 (e.g. 192 for Femto).
        out_channels (int): Output channels. Default: 128.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 f2_channels: int,
                 f3_channels: int,
                 out_channels: int = 128,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.f2_proj = ConvModule(
            f2_channels, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.f3_proj = ConvModule(
            f3_channels, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.refine = ConvModule(
            out_channels * 2, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, f2: Tensor, f3: Tensor) -> Tensor:
        """Fuse F2 (1/8) and F3 (1/16) into P-branch features at 1/8 res.

        Returns:
            Tensor: P features (B, out_ch, H/8, W/8).
        """
        f2_p = self.f2_proj(f2)
        f3_p = self.f3_proj(f3)
        # F3 (1/16) is exactly half the resolution of F2 (1/8) — use scale_factor
        # to avoid dynamic shape computation (SNPE/ONNX portability)
        f3_up = F.interpolate(f3_p, scale_factor=2.0,
                              mode='bilinear', align_corners=False)
        return self.refine(torch.cat([f2_p, f3_up], dim=1))


# ── I Branch — Semantic Context Stream ───────────────────────────────────────

class IBranch(nn.Module):
    """I Branch: semantic context stream with PAG fusion at 1/8 resolution.

    Processes F4 features, fuses with F3 and F2 via two PAG modules.

    Args:
        f2_channels (int): Channels of F2 (e.g. 96 for Femto).
        f3_channels (int): Channels of F3 (e.g. 192 for Femto).
        f4_channels (int): Channels of F4 (e.g. 384 for Femto).
        out_channels (int): Output channels. Default: 256.
        proj_channels (int): Intermediate projection channels for F2/F3.
            Default: 128 (matches P-branch projection).
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 f2_channels: int,
                 f3_channels: int,
                 f4_channels: int,
                 out_channels: int = 256,
                 proj_channels: int = 128,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.f4_proj = ConvModule(
            f4_channels, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.f3_proj = ConvModule(
            f3_channels, proj_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.f2_proj = ConvModule(
            f2_channels, proj_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        # PAG1: fuse F4_proj (upsampled to 1/16) with F3_proj (1/16)
        # gate_channels=out_channels so gate * v_i is dimension-safe (both 256ch)
        self.pag1 = PAGFusion(proj_channels, out_channels,
                              gate_channels=out_channels, norm_cfg=norm_cfg)
        self.pag1_refine = ConvModule(
            out_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        # PAG2: fuse I_mid (upsampled to 1/8) with F2_proj (1/8)
        self.pag2 = PAGFusion(proj_channels, out_channels,
                              gate_channels=out_channels, norm_cfg=norm_cfg)
        self.pag2_refine = ConvModule(
            out_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, f2: Tensor, f3: Tensor, f4: Tensor) -> Tensor:
        """Build semantic context features at 1/8 resolution.

        Returns:
            Tensor: I features (B, out_ch, H/8, W/8).
        """
        # Project
        f4_p = self.f4_proj(f4)          # (B, 256, H/32, W/32)
        f3_p = self.f3_proj(f3)          # (B, 128, H/16, W/16)
        f2_p = self.f2_proj(f2)          # (B, 128, H/8, W/8)

        # F4 → 1/16, PAG with F3
        f4_up = F.interpolate(f4_p, size=f3_p.shape[2:],
                              mode='bilinear', align_corners=False)
        i_mid = self.pag1(f3_p, f4_up)  # (B, 256, H/16, W/16)
        i_mid = self.pag1_refine(i_mid)

        # I_mid → 1/8, PAG with F2
        i_up = F.interpolate(i_mid, size=f2_p.shape[2:],
                             mode='bilinear', align_corners=False)
        i_feat = self.pag2(f2_p, i_up)  # (B, 256, H/8, W/8)
        i_feat = self.pag2_refine(i_feat)
        return i_feat


# ── Boundary-Guided Fusion ───────────────────────────────────────────────────

class BoundaryGuidedFusion(nn.Module):
    """Boundary-guided fusion of D, P, and I streams at 1/4 resolution.

    Uses the D-branch boundary prediction as an attention gate:
      B = sigmoid(boundary_logits)
      semantic_mix = (1 - B) * I_up
      detail_mix = B * P_up
      fused = concat(semantic_mix, detail_mix, D_feat) → refine → 256 ch

    Args:
        d_channels (int): D-branch feature channels (64).
        p_channels (int): P-branch feature channels (128).
        i_channels (int): I-branch feature channels (256).
        out_channels (int): Output channels after fusion. Default: 256.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 d_channels: int,
                 p_channels: int,
                 i_channels: int,
                 out_channels: int = 256,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        total_ch = p_channels + i_channels + d_channels
        self.refine = ConvModule(
            total_ch, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, d_feat: Tensor, p_feat: Tensor, i_feat: Tensor,
                boundary_logits: Tensor) -> Tensor:
        """Fuse all streams at 1/4 resolution.

        All inputs must be at 1/4 resolution (H/4, W/4).

        Args:
            d_feat: (B, d_ch, H/4, W/4) from D-branch.
            p_feat: (B, p_ch, H/4, W/4) P-branch upsampled.
            i_feat: (B, i_ch, H/4, W/4) I-branch upsampled.
            boundary_logits: (B, 1, H/4, W/4) raw boundary logits.

        Returns:
            Tensor: (B, out_ch, H/4, W/4) fused features.
        """
        b_gate = torch.sigmoid(boundary_logits)
        semantic_mix = (1.0 - b_gate) * i_feat
        detail_mix = b_gate * p_feat
        concat = torch.cat([semantic_mix, detail_mix, d_feat], dim=1)
        return self.refine(concat)


# ── Lightweight Decoder ──────────────────────────────────────────────────────

class SegChadDecoder(nn.Module):
    """Lightweight upsampling decoder: 1/4 → 1/1 resolution.

    Two stages of (Conv3x3 → BN → ReLU → bilinear 2x upsample).
    No transposed convolutions — bilinear is more INT8-friendly.

    Args:
        in_channels (int): Input channels (256 from fusion).
        mid_channels (int): Intermediate channels. Default: 128.
        out_channels (int): Output channels. Default: 64.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int = 128,
                 out_channels: int = 64,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # Stage 1: 1/4 → 1/2
        self.stage1_conv = ConvModule(
            in_channels, mid_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        # Stage 2: 1/2 → 1/1
        self.stage2_conv = ConvModule(
            mid_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Upsample features from 1/4 to full resolution (4x total).

        Two 2x bilinear upsamples using scale_factor — avoids dynamic shape
        computation for SNPE/ONNX portability.

        Args:
            x: (B, C, H/4, W/4) fused features.

        Returns:
            Tensor: (B, out_ch, H, W) at full resolution.
        """
        # Stage 1: conv then 2x upsample (1/4 → 1/2)
        x = self.stage1_conv(x)
        x = F.interpolate(x, scale_factor=2.0,
                          mode='bilinear', align_corners=False)
        # Stage 2: conv then 2x upsample (1/2 → 1/1)
        x = self.stage2_conv(x)
        x = F.interpolate(x, scale_factor=2.0,
                          mode='bilinear', align_corners=False)
        return x


# ── Segmentation Head ────────────────────────────────────────────────────────

class SegChadSegHead(nn.Module):
    """Final segmentation head: Conv3x3 → BN → ReLU → Conv1x1 → num_classes.

    Args:
        in_channels (int): Input channels from decoder (64).
        num_classes (int): Number of output classes.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 norm_cfg: Dict = dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.conv = ConvModule(
            in_channels, in_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))
        self.cls = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cls(self.conv(x))


# ── SegChadHead — mmseg decode head wrapping the full decoder ────────────────

@MODELS.register_module()
class SegChadHead(BaseDecodeHead):
    """SegChad decode head: 3-stream D/P/I decoder with boundary-guided fusion.

    Receives 4 feature maps from SegChadBackbone (F1-F4), processes them through
    D, P, I branches, fuses with boundary guidance, and produces segmentation
    logits.

    During training, also produces boundary logits for auxiliary boundary loss.
    During inference, returns only segmentation logits (ONNX-clean).

    The D branch boundary loss uses PIDNet's BoundaryLoss convention: boundary
    GT is generated on-the-fly from the segmentation label (class boundaries)
    rather than requiring a separate GenerateEdge pipeline transform.

    Args:
        in_channels (list[int]): Channels of backbone features [C1, C2, C3, C4].
            For Femto: [48, 96, 192, 384].
        channels (int): Internal working channels for the decoder. Default: 256.
        d_channels (int): D-branch feature channels. Default: 64.
        p_channels (int): P-branch output channels. Default: 128.
        i_channels (int): I-branch output channels. Default: 256.
        decoder_mid_channels (int): Decoder intermediate channels. Default: 128.
        decoder_out_channels (int): Decoder output channels. Default: 64.
        boundary_loss_weight (float): Weight for auxiliary boundary BCE loss.
            Default: 10.0.
        boundary_width (int): Width of boundary region in pixels (on GT mask).
            Default: 4.
        num_classes (int): Number of segmentation classes.
        norm_cfg (dict): Normalization config.
        align_corners (bool): align_corners for interpolation.
    """

    def __init__(self,
                 in_channels: List[int],
                 channels: int = 256,
                 *,
                 num_classes: int,
                 d_channels: int = 64,
                 p_channels: int = 128,
                 i_channels: int = 256,
                 decoder_mid_channels: int = 128,
                 decoder_out_channels: int = 64,
                 boundary_loss_weight: float = 10.0,
                 boundary_width: int = 4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 align_corners: bool = False,
                 **kwargs):
        # BaseDecodeHead expects in_channels, channels, num_classes as positional.
        # Use input_transform='multiple_select' so _transform_inputs passes through
        # the list of feature maps.
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            in_index=[0, 1, 2, 3],
            norm_cfg=norm_cfg,
            align_corners=align_corners,
            # Disable dropout (we have our own head)
            dropout_ratio=0.0,
            **kwargs)

        c1, c2, c3, c4 = in_channels
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_width = boundary_width

        # Three processing streams
        self.d_branch = DBranch(c1, d_channels, norm_cfg=norm_cfg)
        self.p_branch = PBranch(c2, c3, p_channels, norm_cfg=norm_cfg)
        self.i_branch = IBranch(c2, c3, c4, i_channels,
                                proj_channels=p_channels, norm_cfg=norm_cfg)

        # Boundary-guided fusion
        self.fusion = BoundaryGuidedFusion(
            d_channels, p_channels, i_channels, channels, norm_cfg=norm_cfg)

        # Lightweight decoder (1/4 → 1/1)
        self.decoder = SegChadDecoder(
            channels, decoder_mid_channels, decoder_out_channels,
            norm_cfg=norm_cfg)

        # Final segmentation head
        self.seg_head = SegChadSegHead(
            decoder_out_channels, num_classes, norm_cfg=norm_cfg)

        # Override BaseDecodeHead's conv_seg — we use our own seg_head instead
        # Set conv_seg to identity so cls_seg() is a no-op if accidentally called
        del self.conv_seg
        self.conv_seg = nn.Identity()

    def forward(self, inputs):
        """Forward pass through all streams and decoder.

        Args:
            inputs: tuple of 4 feature tensors from backbone (F1, F2, F3, F4).

        Returns:
            Training: tuple(seg_logits, boundary_logits) for loss computation.
            Inference: seg_logits tensor only (ONNX-clean).
        """
        # BaseDecodeHead._transform_inputs selects features by in_index
        x = self._transform_inputs(inputs)
        f1, f2, f3, f4 = x

        # D Branch: boundary features + boundary logits at 1/4 res
        d_feat, boundary_logits = self.d_branch(f1)

        # P Branch: detail features at 1/8 res
        p_feat = self.p_branch(f2, f3)

        # I Branch: semantic context at 1/8 res
        i_feat = self.i_branch(f2, f3, f4)

        # Upsample P and I to 1/4 resolution (D is already at 1/4)
        target_quarter = d_feat.shape[2:]
        p_up = F.interpolate(p_feat, size=target_quarter,
                             mode='bilinear', align_corners=False)
        i_up = F.interpolate(i_feat, size=target_quarter,
                             mode='bilinear', align_corners=False)

        # Boundary-guided fusion at 1/4 res
        fused = self.fusion(d_feat, p_up, i_up, boundary_logits)

        # Decoder: 1/4 → 1/1 (two 2x bilinear upsamples via scale_factor)
        decoded = self.decoder(fused)

        # Segmentation head
        seg_logits = self.seg_head(decoded)

        if self.training:
            return seg_logits, boundary_logits
        return seg_logits

    @staticmethod
    def _generate_boundary_gt(seg_label: Tensor, width: int = 4) -> Tensor:
        """Generate boundary ground truth from segmentation labels.

        Boundary = pixels near class transitions. Uses max-pool dilation of
        each class mask to find boundary regions.

        Args:
            seg_label: (B, H, W) long tensor with class labels.
                       ignore_index (255) regions are excluded.
            width: Boundary dilation width in pixels.

        Returns:
            Tensor: (B, 1, H, W) float boundary GT (1=boundary, 0=non-boundary).
        """
        ignore_mask = (seg_label == 255)
        # Treat ignore as a separate "class" for boundary computation
        # Find pixels where any neighbor has a different class
        kernel_size = 2 * width + 1
        padding = width

        b, h, w = seg_label.shape

        # For each unique class, dilate the mask and check overlap with other classes
        # Efficient approach: max and min pool the label map
        # Boundary = where max_pool != min_pool (ignoring ignore regions)
        seg_float = seg_label.float().unsqueeze(1)  # (B, 1, H, W)

        # Replace ignore with large val for min, small val for max
        seg_for_max = seg_float.clone()
        seg_for_min = seg_float.clone()
        seg_for_max[ignore_mask.unsqueeze(1)] = -1.0
        seg_for_min[ignore_mask.unsqueeze(1)] = 1000.0

        local_max = F.max_pool2d(seg_for_max, kernel_size, stride=1,
                                 padding=padding)
        local_min = -F.max_pool2d(-seg_for_min, kernel_size, stride=1,
                                  padding=padding)

        # Boundary where local max != local min (class transition)
        boundary = ((local_max != local_min) &
                    ~ignore_mask.unsqueeze(1)).float()
        return boundary

    def loss_by_feat(self, seg_logits, batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss + auxiliary boundary loss.

        Args:
            seg_logits: tuple(seg_logits, boundary_logits) during training.
            batch_data_samples: Data samples with GT.

        Returns:
            dict: Loss components.
        """
        seg_pred, boundary_pred = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()

        # Resize seg predictions to GT size
        seg_pred = resize(
            input=seg_pred,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label_squeezed = seg_label.squeeze(1)

        # Segmentation losses (from loss_decode config)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_pred, seg_label_squeezed)
        else:
            seg_weight = None

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_pred, seg_label_squeezed,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_pred, seg_label_squeezed,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # Boundary loss (auxiliary)
        # Generate boundary GT from segmentation labels
        boundary_gt = self._generate_boundary_gt(
            seg_label_squeezed, width=self.boundary_width)

        # Resize boundary prediction to GT size
        boundary_pred_resized = resize(
            input=boundary_pred,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # Weighted BCE: balance pos/neg boundary pixels (same as BoundaryLoss)
        bd_pred_flat = boundary_pred_resized.permute(0, 2, 3, 1).contiguous().view(1, -1)
        bd_gt_flat = boundary_gt.view(1, -1).float()

        pos_index = (bd_gt_flat == 1)
        neg_index = (bd_gt_flat == 0)
        weight = torch.zeros_like(bd_pred_flat)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        if sum_num > 0:
            weight[pos_index] = neg_num.float() / sum_num.float()
            weight[neg_index] = pos_num.float() / sum_num.float()

        loss_bd = F.binary_cross_entropy_with_logits(
            bd_pred_flat, bd_gt_flat, weight, reduction='mean')
        loss['loss_boundary'] = self.boundary_loss_weight * loss_bd

        # Accuracy metric
        loss['acc_seg'] = accuracy(
            seg_pred, seg_label_squeezed, ignore_index=self.ignore_index)

        return loss

