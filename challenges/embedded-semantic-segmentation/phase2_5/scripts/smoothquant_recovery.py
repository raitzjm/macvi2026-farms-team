"""
Manual SmoothQuant implementation with explicit layer mapping for MSCAN/PIDNet.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add repo root to path for mmseg imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)

from mmseg.apis import init_model

# ── Configuration ────────────────────────────────────────────────────────────
LARS_IMG_DIR = '/home/farms/lars_linked/train/images'
MODEL_H, MODEL_W = 384, 768
MEAN = [123.675, 116.28, 103.53]
STD  = [58.395, 57.12, 57.375]

MODELS = {
    'segnext_t': dict(
        config='challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_768x384_lars_phase2.py',
        ckpt='challenges/embedded-semantic-segmentation/phase2/artifacts/segnext_t/best_mIoU_iter_4250.pth',
        align_corners=False,
    ),
    'segnext_t_uperhead': dict(
        config='challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_uperhead_768x384_lars_phase2.py',
        ckpt='challenges/embedded-semantic-segmentation/phase2/artifacts/segnext_t_uperhead/best_mIoU_iter_4250.pth',
        align_corners=False,
    ),
    'pidnets': dict(
        config='challenges/embedded-semantic-segmentation/phase2/configs/pidnets_768x384_lars_phase2.py',
        ckpt='challenges/embedded-semantic-segmentation/phase2/artifacts/pidnets/best_mIoU_iter_4750.pth',
        align_corners=True,
    ),
    'pidnets_p2_5': dict(
        config='challenges/embedded-semantic-segmentation/phase2_5/configs/pidnets_768x384_lars_phase2_5.py',
        ckpt='challenges/embedded-semantic-segmentation/phase2_5/artifacts/pidnets/best_mIoU_iter_4750.pth',
        align_corners=True,
    ),
}

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    ori_w, ori_h = img.size
    scale = min(MODEL_H / ori_h, MODEL_W / ori_w)
    new_h, new_w = round(ori_h * scale), round(ori_w * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    pad_top = (MODEL_H - new_h) // 2
    pad_bottom = MODEL_H - new_h - pad_top
    pad_left = (MODEL_W - new_w) // 2
    pad_right = MODEL_W - new_w - pad_left
    arr = np.pad(arr, [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode='reflect')
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

# ── Manual SmoothQuant ───────────────────────────────────────────────────────

class ActivationObserver:
    def __init__(self):
        self.max_abs = None

    def update(self, x):
        if len(x.shape) == 4:
            # [B, C, H, W] -> max over [B, H, W]
            curr_max = x.abs().transpose(0, 1).reshape(x.shape[1], -1).max(dim=1)[0]
        elif len(x.shape) == 3:
            # [B, L, C] -> max over [B, L]
            curr_max = x.abs().reshape(-1, x.shape[-1]).max(dim=0)[0]
        else:
            return

        if self.max_abs is None:
            self.max_abs = curr_max
        else:
            self.max_abs = torch.max(self.max_abs, curr_max)

def get_smooth_map(model, model_type):
    """Returns a list of (prev_name, next_name) tuples to smooth."""
    mapping = []
    if 'segnext' in model_type:
        # MSCAN stages
        for stage in [1, 2, 3, 4]:
            for name, m in model.named_modules():
                # norm2 -> mlp.fc1 is SAFE (no shortcut in Mlp)
                if f'backbone.block{stage}.' in name and name.endswith('.norm2'):
                    block_prefix = name.replace('.norm2', '')
                    mapping.append((f'{block_prefix}.norm2', f'{block_prefix}.mlp.fc1'))
                    # Also try fc1 -> dwconv (dwconv is the next in Mlp)
                    mapping.append((f'{block_prefix}.mlp.fc1', f'{block_prefix}.mlp.dwconv'))
            
            # Stage transition norms: norm{n} -> patch_embed{n+1}.proj is SAFE
            if stage < 4:
                mapping.append((f'backbone.norm{stage}', f'backbone.patch_embed{stage+1}.proj'))
        
        # Stem: proj.1 -> proj.3 is SAFE (Sequential)
        mapping.append(('backbone.patch_embed1.proj.1', 'backbone.patch_embed1.proj.3'))
        
        # Head: norm4 -> conv_seg is SAFE
        if 'uperhead' not in model_type:
            mapping.append(('backbone.norm4', 'decode_head.conv_seg'))

    elif 'pidnets' in model_type:
        # PIDNet blocks
        for name, m in model.named_modules():
            if '.conv1.bn' in name:
                prefix = name.replace('.conv1.bn', '')
                mapping.append((f'{prefix}.conv1.bn', f'{prefix}.conv2.conv'))
            if '.conv2.bn' in name:
                prefix = name.replace('.conv2.bn', '')
                if hasattr(dict(model.named_modules()).get(f'{prefix}.conv3.conv'), 'weight'):
                    mapping.append((f'{prefix}.conv2.bn', f'{prefix}.conv3.conv'))
    
    return mapping

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()), required=True)
    parser.add_argument('--alpha', type=float, default=0.5, help='SmoothQuant alpha')
    parser.add_argument('--num-samples', type=int, default=100, help='Calibration samples')
    parser.add_argument('--output', type=str, help='Output .pth path')
    args = parser.parse_args()

    m_cfg = MODELS[args.model]
    config_path = os.path.join(REPO_ROOT, m_cfg['config'])
    ckpt_path = os.path.join(REPO_ROOT, m_cfg['ckpt'])
    
    print(f"Loading model: {args.model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_path, ckpt_path, device=device)
    model.eval()

    smooth_map = get_smooth_map(model, args.model)
    print(f"Identified {len(smooth_map)} layer transitions to smooth.")

    # Register observers
    observers = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if name not in observers:
                observers[name] = ActivationObserver()
            observers[name].update(input[0].detach())
        return hook

    # We need to observe the INPUT to the 'next' layer
    for _, next_name in smooth_map:
        module = dict(model.named_modules())[next_name]
        hooks.append(module.register_forward_hook(get_hook(next_name)))

    # Calibration loop
    img_names = sorted(os.listdir(LARS_IMG_DIR))[:args.num_samples]
    print(f"Calibrating on {len(img_names)} images...")
    with torch.no_grad():
        for img_name in tqdm(img_names):
            img_path = os.path.join(LARS_IMG_DIR, img_name)
            x = preprocess(img_path).to(device)
            _ = model(x)

    for h in hooks:
        h.remove()

    print("Applying smoothing transformations...")
    smoothed_count = 0
    for prev_name, next_name in smooth_map:
        if next_name not in observers: continue
        
        act_max = observers[next_name].max_abs
        if act_max is None: continue
        
        prev_mod = dict(model.named_modules())[prev_name]
        next_mod = dict(model.named_modules())[next_name]
        
        weight = next_mod.weight.data
        if isinstance(next_mod, nn.Conv2d):
            if next_mod.groups > 1:
                # DW conv: [C, 1, H, W] -> reduce over [1, H, W] -> [C]
                weight_max = weight.abs().reshape(weight.shape[0], -1).max(dim=1)[0]
            else:
                # [O, I, H, W] -> max over [O, H, W] -> [I]
                weight_max = weight.abs().transpose(0, 1).reshape(weight.shape[1], -1).max(dim=1)[0]
        else:
            # [O, I] -> max over [O] -> [I]
            weight_max = weight.abs().max(dim=0)[0]
        
        # Calculate per-channel scale
        scale = (act_max.pow(args.alpha) / (weight_max.pow(1 - args.alpha) + 1e-5)).clamp(min=1e-5)
        
        # 1. Scale next layer weights
        if isinstance(next_mod, nn.Conv2d):
            if next_mod.groups > 1:
                # DW conv: scale per output channel [C]
                s_broad = scale.view(-1, 1, 1, 1)
            else:
                # Std conv: scale per input channel [I]
                s_broad = scale.view(1, -1, 1, 1)
        else:
            s_broad = scale.view(1, -1)
        next_mod.weight.data.mul_(s_broad)
        
        # 2. Absorb into prev layer
        if isinstance(prev_mod, (nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d, nn.GroupNorm)):
            prev_mod.weight.data.div_(scale)
            if prev_mod.bias is not None:
                prev_mod.bias.data.div_(scale)
        elif isinstance(prev_mod, (nn.Conv2d, nn.Linear)):
            # Migrating variance from activations to weights of PREVIOUS layer
            # X_new = X / scale  => Y_prev = W_prev * X_prev / scale
            # So we scale prev_mod weights by 1/scale on its OUTPUT channels
            if isinstance(prev_mod, nn.Conv2d):
                s_broad_prev = scale.view(-1, 1, 1, 1)
            else:
                s_broad_prev = scale.view(-1, 1)
            prev_mod.weight.data.div_(s_broad_prev)
            if prev_mod.bias is not None:
                prev_mod.bias.data.div_(scale)
        
        smoothed_count += 1

    print(f"Smoothed {smoothed_count} layer transitions.")
    
    # Load original checkpoint to preserve meta info
    orig_ckpt = torch.load(ckpt_path, map_location='cpu')
    orig_ckpt['state_dict'] = model.state_dict()
    
    output_path = args.output or f"{args.model}_smoothed_manual_alpha{args.alpha}.pth"
    torch.save(orig_ckpt, output_path)
    print(f"Smoothed model saved to: {output_path}")

if __name__ == '__main__':
    main()
