"""
Phase 3 pipeline sanity check script.

Runs 5 programmatic checks on the training pipeline. Exit code 0 = all checks
passed (safe to train). Exit code 1 = one or more checks failed (fix before
training).

Usage:
    python phase3/scripts/visualize_pipeline.py \\
        --config phase3/configs/pidnets_768x384_lars_phase3.py \\
        --n 50 --out /tmp/aug_vis/
    # Or for STDC2:
    python phase3/scripts/visualize_pipeline.py \\
        --config phase3/configs/stdc2_768x384_lars_phase3.py \\
        --n 50 --out /tmp/aug_vis/

Checks:
  1. Paste occurred:   >=60% of samples had obstacles pasted.
  2. Boundary blur:    Mean Laplacian on paste boundary < 2× natural boundary Laplacian.
  3. GT mask integrity: All GT values in {0, 1, 2, 255}.
  4. Edge mask (PIDNet): Zero values in padded border of gt_edge_map.
  5. Loss smoke test:  50-iter overfit on 8 images; loss must decrease monotonically
                       or near-monotonically and final < initial × 0.5; no NaN/Inf.
"""

import argparse
import os
import sys
import copy
import random

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_dataset_from_config(config_path, n_samples, repo_root):
    """Build the training dataset from a mmseg config and return n_samples."""
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)

    # Resolve custom_imports relative to repo root (same as training)
    os.chdir(repo_root)

    # Trigger custom_imports so our transforms register themselves
    from mmengine.utils import import_modules_from_strings
    if hasattr(cfg, 'custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)

    # Build train dataset
    from mmseg.datasets import build_dataset  # mmseg 1.x
    # Override lazy to get a concrete dataset
    train_ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    # Limit to first n_samples for speed
    dataset = build_dataset(train_ds_cfg)
    return dataset


def check_paste_occurred(samples, threshold=0.6):
    """Check 1: At least 60% of samples had obstacles pasted."""
    paste_count = 0
    for s in samples:
        gt = s['gt_seg_map'] if 'gt_seg_map' in s else s.get('inputs', None)
        if gt is None:
            continue
        # Count obstacle pixels
        if (gt == 0).sum() > 0:
            paste_count += 1
    rate = paste_count / max(len(samples), 1)
    if rate < threshold:
        return False, (f"Check 1 FAILED: only {rate:.1%} of samples had "
                       f"obstacles pasted (threshold: {threshold:.0%}). "
                       f"Check DB path and water-mask detection in CopyPasteObstacles.")
    return True, f"Check 1 PASSED: {rate:.1%} of samples had obstacles pasted."


def check_boundary_blur(samples, factor=2.0):
    """Check 2: Mean Laplacian on paste boundary < factor × natural boundary Laplacian."""
    paste_boundary_laps = []
    natural_boundary_laps = []

    for s in samples:
        gt = s.get('gt_seg_map')
        img = s.get('img')
        if gt is None or img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        lap = np.abs(cv2.Laplacian(img_gray, cv2.CV_64F))

        # Natural boundary: pixels where GT transitions between classes (not paste)
        # We use a simple edge-of-GT metric for natural boundaries
        gt_f = gt.astype(np.float32)
        # Natural boundary pixels: non-obstacle adjacent to water or sky
        water_mask = (gt == 1)
        sky_mask = (gt == 2)
        # Dilate water+sky to find their boundary with anything
        nat_bound = np.zeros_like(gt, dtype=bool)
        for mask in [water_mask, sky_mask]:
            dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            nat_bound |= ((dilated > 0) & ~mask)
        if nat_bound.sum() > 10:
            natural_boundary_laps.append(lap[nat_bound].mean())

        # Paste boundary: thin ring around obstacle pixels
        obs_mask = (gt == 0).astype(np.uint8)
        if obs_mask.sum() == 0:
            continue
        dilated3 = cv2.dilate(obs_mask, np.ones((3, 3), np.uint8))
        eroded3 = cv2.erode(obs_mask, np.ones((3, 3), np.uint8))
        ring = (dilated3 - eroded3).astype(bool)
        if ring.sum() > 5:
            paste_boundary_laps.append(lap[ring].mean())

    if not paste_boundary_laps or not natural_boundary_laps:
        return True, "Check 2 SKIPPED: insufficient paste or natural boundary data."

    mean_paste = np.mean(paste_boundary_laps)
    mean_nat = np.mean(natural_boundary_laps)
    ratio = mean_paste / max(mean_nat, 1e-6)
    if ratio > factor:
        return False, (f"Check 2 FAILED: paste boundary Laplacian ({mean_paste:.2f}) "
                       f"is {ratio:.2f}× natural boundary ({mean_nat:.2f}) "
                       f"(threshold: {factor:.1f}×). "
                       f"Increase boundary_blur_radius to 3 and re-run.")
    return True, (f"Check 2 PASSED: paste boundary Laplacian ratio {ratio:.2f}× "
                  f"(paste={mean_paste:.2f}, natural={mean_nat:.2f}).")


def check_gt_integrity(samples):
    """Check 3: GT must contain only values {0, 1, 2, 255}."""
    valid = {0, 1, 2, 255}
    for i, s in enumerate(samples):
        gt = s.get('gt_seg_map')
        if gt is None:
            continue
        unique = set(np.unique(gt).tolist())
        bad = unique - valid
        if bad:
            return False, (f"Check 3 FAILED: GT mask has invalid values {bad} "
                           f"in sample {i}. Bug in CopyPasteObstacles GT update logic.")
    return True, "Check 3 PASSED: all GT masks contain only values {0, 1, 2, 255}."


def check_edge_mask_padding(samples):
    """Check 4 (PIDNet only): Edge map must be zero in padded border."""
    edge_key = 'gt_edge_map'
    checked = 0
    for i, s in enumerate(samples):
        edge = s.get(edge_key)
        padding = s.get('img_padding_size')
        if edge is None or padding is None:
            continue
        left, right, top, bottom = padding
        H, W = edge.shape
        if top > 0 and edge[:top, :].sum() != 0:
            return False, (f"Check 4 FAILED: edge map has non-zero values in "
                           f"top padded region (top={top}) in sample {i}. "
                           f"GenerateEdgeMasked is not reading img_padding_size correctly.")
        if bottom > 0 and edge[H - bottom:, :].sum() != 0:
            return False, (f"Check 4 FAILED: edge map has non-zero values in "
                           f"bottom padded region (bottom={bottom}) in sample {i}.")
        if left > 0 and edge[:, :left].sum() != 0:
            return False, (f"Check 4 FAILED: edge map has non-zero values in "
                           f"left padded region (left={left}) in sample {i}.")
        if right > 0 and edge[:, W - right:].sum() != 0:
            return False, (f"Check 4 FAILED: edge map has non-zero values in "
                           f"right padded region (right={right}) in sample {i}.")
        checked += 1
    if checked == 0:
        return True, "Check 4 SKIPPED: no padded samples found (config may not use GenerateEdgeMasked)."
    return True, f"Check 4 PASSED: edge map padding suppressed in all {checked} padded samples."


def check_loss_smoke_test(config_path, repo_root, n_iters=50, n_images=8):
    """Check 5: 50-iter overfit on n_images; loss must decrease and reach < initial×0.5."""
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    import tempfile

    cfg = Config.fromfile(config_path)
    os.chdir(repo_root)

    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    # Patch config for single-GPU overfit
    cfg.launcher = 'none'
    cfg.train_dataloader.batch_size = n_images
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False

    # Limit dataset to n_images (InfiniteSampler will repeat them)
    # We use a short fixed seed for reproducibility
    from mmengine.utils import import_modules_from_strings
    from mmseg.datasets import build_dataset
    import copy
    train_ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    full_ds = build_dataset(train_ds_cfg)
    indices = list(range(min(n_images, len(full_ds))))

    # Smoke: manually run forward passes
    from torch.utils.data import DataLoader, Subset
    subset = Subset(full_ds, indices)

    from mmseg.models import build_segmentor
    model_cfg = copy.deepcopy(cfg.model)
    model = build_segmentor(model_cfg).cuda()
    model.train()

    # Build dataloader with collate
    from mmengine.dataset import default_collate
    loader = DataLoader(
        subset,
        batch_size=n_images,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate,
    )

    from mmengine.optim import build_optim_wrapper
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    from mmseg.utils import register_all_modules
    register_all_modules()

    losses = []
    data_iter = iter(loader)
    for i in range(n_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move to GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        optim_wrapper.zero_grad()
        loss_dict = model.train_step(batch, optim_wrapper)
        loss_val = loss_dict.get('loss', loss_dict.get('decode.loss_ce', None))
        if loss_val is None:
            # Try to get total loss from any key
            total = sum(v for k, v in loss_dict.items() if 'loss' in k and isinstance(v, torch.Tensor))
            loss_val = total

        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()

        if np.isnan(loss_val) or np.isinf(loss_val):
            return False, (f"Check 5 FAILED: NaN/Inf loss at iter {i}. "
                           f"Augmentation is destabilising the loss. "
                           f"Investigate GT corruption or extreme photometric distortion.")
        losses.append(loss_val)

    initial = losses[0]
    final = losses[-1]
    # Check monotonically decreasing (allowing 10% non-monotone tolerance)
    violations = sum(1 for a, b in zip(losses, losses[1:]) if b > a * 1.10)
    if final >= initial * 0.5:
        return False, (f"Check 5 FAILED: final loss ({final:.4f}) >= initial ({initial:.4f}) × 0.5. "
                       f"Loss is not decreasing sufficiently. "
                       f"Loss trajectory: {[f'{l:.3f}' for l in losses[::10]]}")
    if violations > n_iters * 0.3:
        return False, (f"Check 5 FAILED: loss increased (>10%) in {violations}/{n_iters-1} "
                       f"consecutive steps. Training is unstable.")
    return True, (f"Check 5 PASSED: loss decreased from {initial:.4f} → {final:.4f} "
                  f"({final/initial:.1%} of initial) over {n_iters} iters.")


def save_visualizations(samples, out_dir, n_save=10):
    """Save side-by-side original | augmented | GT mask visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    palette = {0: (247, 195, 37), 1: (41, 167, 224), 2: (90, 75, 164), 255: (50, 50, 50)}
    saved = 0
    for i, s in enumerate(samples[:n_save]):
        img = s.get('img')
        gt = s.get('gt_seg_map')
        if img is None or gt is None:
            continue
        # GT mask visualisation
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            gt_vis[gt == cls_id] = color

        H, W = img.shape[:2]
        gt_vis = cv2.resize(gt_vis, (W, H))
        combined = np.concatenate([img, gt_vis], axis=1)
        cv2.imwrite(os.path.join(out_dir, f'sample_{i:03d}.jpg'), combined)
        saved += 1
    print(f"Saved {saved} visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Phase 3 pipeline sanity check')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--n', type=int, default=50, help='Number of samples to check')
    parser.add_argument('--out', default='/tmp/aug_vis/', help='Output dir for visualizations')
    parser.add_argument('--skip-smoke', action='store_true',
                        help='Skip the loss smoke test (checks 1-4 only)')
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Config:    {config_path}")
    print(f"Samples:   {args.n}")
    print(f"Repo root: {REPO_ROOT}")
    print()

    # Trigger custom_imports
    os.chdir(REPO_ROOT)
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, 'custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)

    # Build dataset and gather samples
    from mmseg.datasets import build_dataset
    import copy
    train_ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    print("Building dataset...")
    dataset = build_dataset(train_ds_cfg)
    print(f"Dataset size: {len(dataset)}")

    # Sample randomly
    idxs = random.sample(range(len(dataset)), min(args.n, len(dataset)))
    print(f"Processing {len(idxs)} samples...")
    samples = []
    for idx in idxs:
        try:
            s = dataset[idx]
            samples.append(s)
        except Exception as e:
            print(f"  WARNING: failed to load sample {idx}: {e}")

    print(f"Successfully loaded {len(samples)} samples.")
    print()

    # Run checks
    all_passed = True
    results = []

    ok, msg = check_paste_occurred(samples)
    results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_boundary_blur(samples)
    results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_gt_integrity(samples)
    results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_edge_mask_padding(samples)
    results.append((ok, msg))
    all_passed &= ok

    if not args.skip_smoke:
        print("Running loss smoke test (50 iters on 8 images)...")
        try:
            ok, msg = check_loss_smoke_test(config_path, REPO_ROOT)
        except Exception as e:
            ok, msg = False, f"Check 5 ERROR: {e}"
        results.append((ok, msg))
        all_passed &= ok
    else:
        print("Smoke test skipped (--skip-smoke).")

    # Print results
    print()
    print("=" * 60)
    print("PIPELINE SANITY CHECK RESULTS")
    print("=" * 60)
    for ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")
    print("=" * 60)

    # Save visualizations (informational only)
    save_visualizations(samples, args.out)

    if all_passed:
        print("\nAll checks PASSED. Safe to launch training.")
        sys.exit(0)
    else:
        print("\nOne or more checks FAILED. Fix issues before launching training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
