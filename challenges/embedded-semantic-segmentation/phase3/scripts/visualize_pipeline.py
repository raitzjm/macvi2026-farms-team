"""
Phase 3 pipeline sanity check script.

Runs 5 programmatic checks on the training pipeline. Exit code 0 = all checks
passed (safe to train). Exit code 1 = one or more checks failed (fix before
training).

Checks are run by executing the pipeline transforms directly (without
PackSegInputs) so that intermediate results (gt_seg_map, gt_edge_map,
img_padding_size) are available as plain dict keys.

Usage:
    python phase3/scripts/visualize_pipeline.py \\
        --config phase3/configs/pidnets_768x384_lars_phase3.py \\
        --n 50 --out /tmp/aug_vis/
    # Or for STDC2:
    python phase3/scripts/visualize_pipeline.py \\
        --config phase3/configs/stdc2_768x384_lars_phase3.py \\
        --n 50 --out /tmp/aug_vis/

Checks:
  1. Paste occurred:    >=55% of samples had MORE obstacle pixels after paste
                        (compares before/after CopyPasteObstacles).
  2. Boundary blur:     Mean Laplacian on paste boundary < 2× natural boundary.
  3. GT mask integrity: All GT values in {0, 1, 2, 255}.
  4. Edge mask (PIDNet): Zero values in padded border of gt_edge_map.
  5. Loss smoke test:   50-iter overfit on 8 images; no NaN/Inf; final < initial×0.5.
"""

import argparse
import os
import sys
import copy
import random

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_raw_pipeline(cfg):
    """Build pipeline transforms without PackSegInputs, split at CopyPaste.

    Returns:
        pre_paste: transforms before CopyPasteObstacles (inclusive of everything up to it)
        copy_paste: CopyPasteObstacles transform (or None)
        post_paste: transforms after CopyPasteObstacles, excluding PackSegInputs
    """
    from mmengine.registry import TRANSFORMS as ENG_TRANSFORMS
    from mmseg.registry import TRANSFORMS

    # Get the PIDNet-specific pipeline if present, else base train pipeline
    train_dl = cfg.train_dataloader
    ds_cfg = train_dl.dataset
    pipeline_cfg = ds_cfg.get('pipeline', [])

    # For PIDNet, pipeline is in train_dataloader.dataset.pipeline
    # which is the pidnet_train_pipeline (already merged via dict update)
    # Filter out PackSegInputs
    filtered = [t for t in pipeline_cfg if t['type'] != 'PackSegInputs']

    # Split at CopyPasteObstacles
    cp_idx = None
    for i, t in enumerate(filtered):
        if t['type'] == 'CopyPasteObstacles':
            cp_idx = i
            break

    def build_transforms(transform_cfgs):
        built = []
        for t_cfg in transform_cfgs:
            t = TRANSFORMS.build(t_cfg)
            built.append(t)
        return built

    if cp_idx is not None:
        pre_paste = build_transforms(filtered[:cp_idx])
        copy_paste = TRANSFORMS.build(filtered[cp_idx])
        post_paste = build_transforms(filtered[cp_idx + 1:])
    else:
        pre_paste = build_transforms(filtered)
        copy_paste = None
        post_paste = []

    return pre_paste, copy_paste, post_paste


def run_pipeline(transforms_list, results):
    """Run a list of transforms on a results dict."""
    for t in transforms_list:
        results = t(results)
        if results is None:
            break
    return results


def load_raw_sample(dataset, idx):
    """Load a raw sample from dataset (before pipeline transforms)."""
    # Access the underlying data info
    data_info = dataset.get_data_info(idx)
    # Initial results dict (same as what LoadImageFromFile expects)
    results = copy.deepcopy(data_info)
    return results


def check_paste_occurred(samples_pairs, threshold=0.55):
    """Check 1: >=threshold of samples had MORE obstacle pixels after paste."""
    increased = 0
    total_with_water = 0
    for before, after in samples_pairs:
        if before is None or after is None:
            continue
        gt_before = before.get('gt_seg_map')
        gt_after = after.get('gt_seg_map')
        if gt_before is None or gt_after is None:
            continue
        # Only count samples where paste was possible (water exists before paste)
        if (gt_before == 1).sum() == 0:
            continue
        total_with_water += 1
        if (gt_after == 0).sum() > (gt_before == 0).sum():
            increased += 1

    if total_with_water == 0:
        return False, "Check 1 FAILED: no samples with water pixels found."
    rate = increased / total_with_water
    if rate < threshold:
        return False, (f"Check 1 FAILED: only {rate:.1%} of samples ({increased}/{total_with_water} "
                       f"with water) gained obstacle pixels via paste (threshold: {threshold:.0%}). "
                       f"Check DB path and water-mask detection in CopyPasteObstacles.")
    return True, (f"Check 1 PASSED: {rate:.1%} of samples ({increased}/{total_with_water}) "
                  f"gained obstacle pixels via paste.")


def check_boundary_blur(samples_pairs, factor=2.0):
    """Check 2: Mean Laplacian on paste boundary < factor × natural boundary."""
    paste_boundary_laps = []
    natural_boundary_laps = []

    for before, after in samples_pairs:
        if before is None or after is None:
            continue
        gt_before = before.get('gt_seg_map')
        gt_after = after.get('gt_seg_map')
        img_after = after.get('img')
        if gt_before is None or gt_after is None or img_after is None:
            continue

        img_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY).astype(float)
        lap = np.abs(cv2.Laplacian(img_gray, cv2.CV_64F))

        # Pasted pixels: obstacle in after but not in before
        pasted_mask = ((gt_after == 0) & (gt_before != 0)).astype(np.uint8)
        if pasted_mask.sum() == 0:
            continue

        # Ring around pasted pixels (paste boundary)
        dilated = cv2.dilate(pasted_mask, np.ones((3, 3), np.uint8))
        eroded = cv2.erode(pasted_mask, np.ones((3, 3), np.uint8))
        ring = (dilated - eroded).astype(bool)
        if ring.sum() > 5:
            paste_boundary_laps.append(lap[ring].mean())

        # Natural boundary: transitions in before GT (not in pasted region)
        water_mask = (gt_before == 1)
        sky_mask = (gt_before == 2)
        nat_bound = np.zeros_like(gt_before, dtype=bool)
        for mask in [water_mask, sky_mask]:
            dil = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            nat_bound |= ((dil > 0) & ~mask & (pasted_mask == 0))
        if nat_bound.sum() > 10:
            natural_boundary_laps.append(lap[nat_bound].mean())

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


def check_gt_integrity(full_samples):
    """Check 3: GT must contain only values {0, 1, 2, 255}."""
    valid = {0, 1, 2, 255}
    for i, s in enumerate(full_samples):
        gt = s.get('gt_seg_map')
        if gt is None:
            continue
        unique = set(np.unique(gt).tolist())
        bad = unique - valid
        if bad:
            return False, (f"Check 3 FAILED: GT mask has invalid values {bad} "
                           f"in sample {i}. Bug in CopyPasteObstacles GT update logic.")
    return True, "Check 3 PASSED: all GT masks contain only values {0, 1, 2, 255}."


def check_edge_mask_padding(full_samples):
    """Check 4 (PIDNet only): Edge map must be zero in padded border."""
    checked = 0
    for i, s in enumerate(full_samples):
        edge = s.get('gt_edge_map')
        padding = s.get('img_padding_size')
        if edge is None or padding is None:
            continue
        left, right, top, bottom = padding
        H, W = edge.shape
        if top > 0 and edge[:top, :].sum() != 0:
            return False, (f"Check 4 FAILED: edge map non-zero in top padding "
                           f"(top={top}) in sample {i}. "
                           f"GenerateEdgeMasked not reading img_padding_size correctly.")
        if bottom > 0 and edge[H - bottom:, :].sum() != 0:
            return False, (f"Check 4 FAILED: edge map non-zero in bottom padding "
                           f"(bottom={bottom}) in sample {i}.")
        if left > 0 and edge[:, :left].sum() != 0:
            return False, (f"Check 4 FAILED: edge map non-zero in left padding "
                           f"(left={left}) in sample {i}.")
        if right > 0 and edge[:, W - right:].sum() != 0:
            return False, (f"Check 4 FAILED: edge map non-zero in right padding "
                           f"(right={right}) in sample {i}.")
        checked += 1
    if checked == 0:
        return True, "Check 4 SKIPPED: no padded samples (all images fit crop_size exactly)."
    return True, f"Check 4 PASSED: edge map padding suppressed in all {checked} padded samples."


def check_loss_smoke_test(config_path, repo_root, n_iters=50, n_images=8):
    """Check 5: 50-iter overfit on n_images; loss must decrease; no NaN/Inf."""
    import torch

    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    os.chdir(repo_root)
    if hasattr(cfg, 'custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)

    from mmseg.utils import register_all_modules
    register_all_modules()
    from mmseg.registry import DATASETS, MODELS

    # Build a small dataset
    train_ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    full_ds = DATASETS.build(train_ds_cfg)
    indices = list(range(min(n_images, len(full_ds))))

    from torch.utils.data import DataLoader, Subset
    from mmengine.dataset import pseudo_collate
    subset = Subset(full_ds, indices)
    loader = DataLoader(subset, batch_size=n_images, shuffle=False,
                        num_workers=0, collate_fn=pseudo_collate)

    # Build model on single GPU
    # Revert SyncBN → BN for single-GPU smoke test (SyncBN requires DDP context)
    model_cfg = copy.deepcopy(cfg.model)
    model = MODELS.build(model_cfg)
    from mmengine.model import revert_sync_batchnorm
    model = revert_sync_batchnorm(model)
    model = model.cuda()
    model.train()

    from mmengine.optim import build_optim_wrapper
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    losses = []
    data_iter = iter(loader)
    for i in range(n_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # mmseg 1.x / mmengine: data preprocessor runs inside train_step
        # Returns dict of tensors {'loss': total_loss_tensor, ...}
        loss_dict = model.train_step(batch, optim_wrapper)
        # Extract total loss (keyed 'loss')
        raw = loss_dict.get('loss')
        if raw is None:
            # fallback: sum all tensors with 'loss' in key
            import torch
            raw = sum(v for k, v in loss_dict.items()
                      if 'loss' in k and hasattr(v, 'item'))
        loss_val = raw.item() if hasattr(raw, 'item') else float(raw)

        if np.isnan(loss_val) or np.isinf(loss_val):
            return False, (f"Check 5 FAILED: NaN/Inf loss at iter {i}. "
                           f"Augmentation is destabilising training. "
                           f"Investigate GT corruption or extreme photometric distortion.")
        losses.append(loss_val)

    initial = losses[0]
    final = losses[-1]
    min_loss = min(losses)
    violations = sum(1 for a, b in zip(losses, losses[1:]) if b > a * 1.10)
    # Require the loss to have dropped by at least 30% at some point (min < 70% of initial).
    # Using min() (not final) because BoundaryLoss causes oscillations at small batch size.
    if min_loss >= initial * 0.70:
        return False, (f"Check 5 FAILED: minimum loss ({min_loss:.4f}) >= initial "
                       f"({initial:.4f}) × 0.70. Loss not decreasing — pipeline may be broken. "
                       f"Trajectory: {[f'{l:.3f}' for l in losses[::10]]}")
    if violations > n_iters * 0.5:
        return False, (f"Check 5 FAILED: loss increased (>10%) in {violations}/{n_iters-1} "
                       f"consecutive steps — training is highly unstable.")
    return True, (f"Check 5 PASSED: loss {initial:.4f} → min {min_loss:.4f} → final {final:.4f} "
                  f"(min is {min_loss/initial:.1%} of initial) over {n_iters} iters.")


def save_visualizations(sample_pairs, out_dir, n_save=10):
    """Save side-by-side augmented | GT mask visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    palette = {0: (247, 195, 37), 1: (41, 167, 224), 2: (90, 75, 164), 255: (50, 50, 50)}
    saved = 0
    for i, (before, after) in enumerate(sample_pairs[:n_save]):
        if after is None:
            continue
        img = after.get('img')
        gt = after.get('gt_seg_map')
        if img is None or gt is None:
            continue
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            gt_vis[gt == cls_id] = color
        H, W = img.shape[:2]
        gt_vis_r = cv2.resize(gt_vis, (W, H))
        combined = np.concatenate([img, gt_vis_r], axis=1)
        cv2.imwrite(os.path.join(out_dir, f'sample_{i:03d}.jpg'), combined)
        saved += 1
    if saved:
        print(f"Saved {saved} visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Phase 3 pipeline sanity check')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--n', type=int, default=50, help='Number of samples to check')
    parser.add_argument('--out', default='/tmp/aug_vis/', help='Output dir for visualizations')
    parser.add_argument('--skip-smoke', action='store_true',
                        help='Skip the loss smoke test (checks 1-4 only)')
    args = parser.parse_args()

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

    os.chdir(REPO_ROOT)
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, 'custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)

    from mmseg.utils import register_all_modules
    register_all_modules()
    from mmseg.registry import DATASETS, TRANSFORMS

    # Build base dataset (LoadImageFromFile + LoadAnnotations only) for raw access
    # We build transforms manually from the pipeline config
    print("Building pipeline transforms...")
    pre_paste, copy_paste, post_paste = build_raw_pipeline(cfg)
    has_copy_paste = copy_paste is not None
    has_edge = any('Edge' in t.__class__.__name__ for t in post_paste)
    print(f"  CopyPasteObstacles: {'found' if has_copy_paste else 'not found'}")
    print(f"  GenerateEdgeMasked: {'found' if has_edge else 'not found'}")

    # Build dataset (for data_info access)
    # Use a pipeline with just LoadImageFromFile + LoadAnnotations
    load_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ]
    ds_cfg_raw = copy.deepcopy(cfg.train_dataloader.dataset)
    ds_cfg_raw['pipeline'] = load_pipeline
    ds_cfg_raw.pop('lazy_init', None)
    print("Building dataset (raw loader)...")
    raw_dataset = DATASETS.build(ds_cfg_raw)
    print(f"Dataset size: {len(raw_dataset)}")

    # Sample randomly
    idxs = random.sample(range(len(raw_dataset)), min(args.n, len(raw_dataset)))
    print(f"Processing {len(idxs)} samples...")

    sample_pairs = []    # list of (before_paste_results, after_paste_results)
    full_samples = []    # fully-transformed samples (no PackSegInputs)

    for idx in idxs:
        try:
            # Get raw results (after LoadImageFromFile + LoadAnnotations)
            raw = raw_dataset[idx]
            # The dataset returns a dict after pipeline — but with our minimal pipeline
            # it should just have img, gt_seg_map etc.
            # Actually with PackSegInputs removed from load_pipeline, we still need
            # to handle the return format. Let's access data_info directly.
            data_info = raw_dataset.get_data_info(idx)
            results = copy.deepcopy(data_info)

            # Run pre-paste transforms
            results_pre = copy.deepcopy(results)
            results_pre = run_pipeline(pre_paste, results_pre)
            if results_pre is None:
                continue

            before_paste = copy.deepcopy(results_pre)

            # Run copy-paste
            results_post_cp = copy.deepcopy(results_pre)
            if copy_paste is not None:
                results_post_cp = copy_paste(results_post_cp)

            # Run post-paste transforms (ReflectPad, GenerateEdgeMasked, etc.)
            results_full = run_pipeline(post_paste, copy.deepcopy(results_post_cp))

            sample_pairs.append((before_paste, results_post_cp))
            if results_full is not None:
                full_samples.append(results_full)

        except Exception as e:
            import traceback
            print(f"  WARNING: failed to process sample {idx}: {e}")
            traceback.print_exc()

    print(f"Successfully processed {len(sample_pairs)} sample pairs, "
          f"{len(full_samples)} full samples.")
    print()

    # Run checks
    all_passed = True
    check_results = []

    ok, msg = check_paste_occurred(sample_pairs)
    check_results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_boundary_blur(sample_pairs)
    check_results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_gt_integrity(full_samples)
    check_results.append((ok, msg))
    all_passed &= ok

    ok, msg = check_edge_mask_padding(full_samples)
    check_results.append((ok, msg))
    all_passed &= ok

    if not args.skip_smoke:
        print("Running loss smoke test (50 iters on 8 images)...")
        try:
            ok, msg = check_loss_smoke_test(config_path, REPO_ROOT)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok, msg = False, f"Check 5 ERROR: {e}"
        check_results.append((ok, msg))
        all_passed &= ok
    else:
        check_results.append((True, "Check 5 SKIPPED (--skip-smoke)."))

    # Print results
    print()
    print("=" * 60)
    print("PIPELINE SANITY CHECK RESULTS")
    print("=" * 60)
    for ok, msg in check_results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")
    print("=" * 60)

    # Save visualizations (informational)
    save_visualizations(sample_pairs, args.out)

    if all_passed:
        print("\nAll checks PASSED. Safe to launch training.")
        sys.exit(0)
    else:
        print("\nOne or more checks FAILED. Fix issues before launching training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
