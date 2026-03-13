"""
Manual QAT model preparation script.
Bypasses symbolic tracing issues by directly wrapping layers.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, QConfig

def prepare_qat(model):
    """
    Manually prepares a model for QAT by adding FakeQuantize modules.
    """
    # Define QConfig
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False
        )
    )

    def _attach_qconfig(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                child.qconfig = qconfig
            else:
                _attach_qconfig(child)

    _attach_qconfig(model)
    
    # Use torch.ao.quantization.prepare_qat_inplace or similar if available
    # But for a more surgical approach, we can just use FakeQuant wrappers
    
    from torch.ao.quantization import prepare_qat_inplace
    # Note: prepare_qat_inplace might still use some tracing/mapping internally
    # Let's try a simpler wrapper approach if it fails.
    
    prepare_qat_inplace(model)
    return model

# ── Custom QAT Wrapper ────────────────────────────────────────────────────────

class FakeQuantWrapper(nn.Module):
    def __init__(self, mod, qconfig):
        super().__init__()
        self.mod = mod
        self.weight_fake_quant = qconfig.weight()
        self.activation_fake_quant = qconfig.activation()

    def forward(self, x):
        # Fake quantize activation
        x = self.activation_fake_quant(x)
        
        # Fake quantize weight
        q_weight = self.weight_fake_quant(self.mod.weight)
        
        # Dispatch to functional call depending on module type
        if isinstance(self.mod, nn.Conv2d):
            return torch.nn.functional.conv2d(
                x, q_weight, self.mod.bias, self.mod.stride,
                self.mod.padding, self.mod.dilation, self.mod.groups)
        elif isinstance(self.mod, nn.Linear):
            return torch.nn.functional.linear(x, q_weight, self.mod.bias)
        else:
            return self.mod(x)

def wrap_model_qat(model):
    """
    Surgically wraps Conv/Linear layers with FakeQuant without FX tracing.
    """
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False
        )
    )

    def r_wrap(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, FakeQuantWrapper(child, qconfig))
            else:
                r_wrap(child)
    
    r_wrap(model)
    return model
