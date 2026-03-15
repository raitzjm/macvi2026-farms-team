"""Phase 3 custom model components — PIDChad-S (EXP-021) + SegChad (EXP-022)."""

from .pidchad import AFFM, FPPM, PIDChadNet, PIDChadHead
from .segchad import (SegChadBackbone, SegChadHead, PAGFusion,
                      DBranch, PBranch, IBranch, BoundaryGuidedFusion,
                      SegChadDecoder, SegChadSegHead)

__all__ = [
    'AFFM', 'FPPM', 'PIDChadNet', 'PIDChadHead',
    'SegChadBackbone', 'SegChadHead', 'PAGFusion',
    'DBranch', 'PBranch', 'IBranch', 'BoundaryGuidedFusion',
    'SegChadDecoder', 'SegChadSegHead',
]
