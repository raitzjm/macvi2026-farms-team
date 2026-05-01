#!/usr/bin/env python3
"""
Co-DETR smoke trainer for FathomNet-CLEF 2026.

Loads the Co-DETR R50 (5-scale, no LSJ) config from the bundled CO-DETR project,
remaps non-contiguous COCO category IDs to a contiguous 0..N-1 space, uses a
shared explicit train/val split manifest, and runs a short MMEngine training
loop with AMP and WandB logging via the visualizer.

Designed to run on:
- Local container (RTX 3050 4GB) — 1-batch smoke
- Kaggle kernels (T4/P100)
- TensorDock (set FATHOMNET_DATA_DIR)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- Make CO-DETR project importable BEFORE importing mmdet ----------------
HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent
CODETR_PROJ = CHALLENGE_ROOT / "envs" / "mmdet-src" / "projects" / "CO-DETR"
if CODETR_PROJ.exists():
    sys.path.insert(0, str(CODETR_PROJ))
# Also add the parent so `projects.CO-DETR.codetr` style imports resolve.
MMDET_SRC = CHALLENGE_ROOT / "envs" / "mmdet-src"
if MMDET_SRC.exists():
    sys.path.insert(0, str(MMDET_SRC))

sys.path.insert(0, str(HERE))
from _paths import TRAIN_JSON, TRAIN_IMGS, DATASET_ROOT  # noqa: E402
from run_tracking import (  # noqa: E402
    copy_if_exists,
    git_commit,
    relpath_str,
    resolve_artifact_dir,
    tee_output,
    tracked_run_name,
    write_json,
)
from split_manifest import (  # noqa: E402
    default_split_manifest_path,
    ensure_manifest,
    load_manifest,
)


def setup_wandb(disabled: bool) -> bool:
    if disabled:
        os.environ["WANDB_DISABLED"] = "true"
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    print("[warn] no WANDB_API_KEY found; wandb logging will be skipped.")
    os.environ["WANDB_DISABLED"] = "true"
    return False


def prepare_splits(train_json: Path, out_dir: Path, split_manifest_path: Path):
    """Build contiguous category map; write shared split COCO JSONs."""
    with open(train_json) as f:
        coco = json.load(f)

    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    new_categories = [
        {"id": i, "name": c["name"], "supercategory": c.get("supercategory", "")}
        for i, c in enumerate(cats_sorted)
    ]
    class_names = [c["name"] for c in new_categories]

    # Remap annotations
    remapped_anns = []
    for ann in coco["annotations"]:
        a = dict(ann)
        a["category_id"] = cat_id_to_idx[ann["category_id"]]
        remapped_anns.append(a)

    manifest = load_manifest(split_manifest_path)
    images_by_id = {int(im["id"]): im for im in coco["images"]}
    train_imgs = [images_by_id[i] for i in manifest["train_ids"]]
    val_imgs = [images_by_id[i] for i in manifest["val_ids"]]
    val_ids = {im["id"] for im in val_imgs}

    train_anns = [a for a in remapped_anns if a["image_id"] not in val_ids]
    val_anns = [a for a in remapped_anns if a["image_id"] in val_ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "coco_train.json"
    val_path = out_dir / "coco_val.json"
    with open(train_path, "w") as f:
        json.dump(
            {"images": train_imgs, "annotations": train_anns,
             "categories": new_categories, "info": coco.get("info", {}),
             "licenses": coco.get("licenses", [])}, f)
    with open(val_path, "w") as f:
        json.dump(
            {"images": val_imgs, "annotations": val_anns,
             "categories": new_categories, "info": coco.get("info", {}),
             "licenses": coco.get("licenses", [])}, f)
    print(f"[data] train={len(train_imgs)} imgs / {len(train_anns)} anns")
    print(f"[data] val  ={len(val_imgs)} imgs / {len(val_anns)} anns")
    print(f"[data] {len(class_names)} classes")
    return train_path, val_path, class_names


def _set_num_classes(node, num_classes: int):
    """Recursively set every dict key 'num_classes' encountered."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "num_classes":
                node[k] = num_classes
            else:
                _set_num_classes(v, num_classes)
    elif isinstance(node, list):
        for v in node:
            _set_num_classes(v, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--name", default="smoke-codetr")
    ap.add_argument("--width", type=int, default=2000,
                    help="Resize long-side cap (mmdet scale=(width, height) with keep_ratio)")
    ap.add_argument("--height", type=int, default=640,
                    help="Resize short-side target")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable AMP. AMP is on by default (with mmcv NMS dtype patch)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--run-id", default=None,
                    help="Optional tracked training ID (for example T-004)")
    ap.add_argument("--artifact-dir", default=None,
                    help="Optional durable artifact directory override")
    ap.add_argument("--split-manifest", default=str(default_split_manifest_path()),
                    help="Path to shared train/val split manifest JSON")
    args = ap.parse_args()
    args.amp = not args.no_amp

    artifact_dir = resolve_artifact_dir(CHALLENGE_ROOT, "training", args.run_id, args.artifact_dir)
    log_path = artifact_dir / "stdout.log" if artifact_dir else None

    with tee_output(log_path):
        _main(args, artifact_dir)


def _main(args, artifact_dir: Path | None):

    if args.amp:
        # mmcv 2.1.0 + Co-DETR + AMP triggers two known issues:
        #
        # 1. batched_nms dtype mismatch: rpn_head scores buffer is fp16 but NMS
        #    op returns fp32 dets. Fix: cast inputs to fp32 before NMS.
        import mmcv.ops as _ops_mod
        import mmcv.ops.nms as _nms_mod
        _orig_bn = _ops_mod.batched_nms

        def _bn_fp32_safe(boxes, scores, idxs, nms_cfg, class_agnostic=False):
            return _orig_bn(boxes.float(), scores.float(), idxs,
                            nms_cfg, class_agnostic)

        _ops_mod.batched_nms = _bn_fp32_safe
        _nms_mod.batched_nms = _bn_fp32_safe
        try:
            import mmdet.models.dense_heads.rpn_head as _rpn
            _rpn.batched_nms = _bn_fp32_safe
        except Exception:
            pass

        # 2. FocalLoss NaN under fp16: CoATSSHead's sigmoid_focal_loss
        #    overflows in fp16 (exp(-pred) exceeds fp16 range → NaN), producing
        #    NaN loss_cls1 and a NaN cost matrix in the Hungarian assigner.
        #    Fix: cast only pred to fp32; target must remain torch.long for the
        #    mmcv CUDA kernel (it asserts dtype == long).
        try:
            import mmdet.models.losses.focal_loss as _fl_mod
            _orig_sfl = _fl_mod.sigmoid_focal_loss

            def _sfl_fp32_safe(pred, target, weight=None, gamma=2.0, alpha=0.25,
                               reduction="mean", avg_factor=None):
                return _orig_sfl(pred.float(), target,
                                 weight, gamma, alpha, reduction, avg_factor)

            _fl_mod.sigmoid_focal_loss = _sfl_fp32_safe
            # patch the binding already imported inside focal_loss.py itself
            import mmdet.models.losses.focal_loss as _fl2
            _fl2.sigmoid_focal_loss = _sfl_fp32_safe
        except Exception:
            pass

    use_wandb = setup_wandb(args.no_wandb)

    run_name = tracked_run_name(args.name, args.run_id)

    out_dir = CHALLENGE_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = ensure_manifest(Path(TRAIN_JSON), Path(args.split_manifest))
    train_json, val_json, class_names = prepare_splits(Path(TRAIN_JSON), out_dir, split_manifest_path)
    num_classes = len(class_names)

    # Import CO-DETR so its registries populate, then load config.
    # The config's `custom_imports` references `projects.CO-DETR.codetr`,
    # which is not a valid Python dotted name (hyphen). Pre-register the
    # alias in sys.modules so mmengine's import_module call short-circuits.
    import types
    import codetr  # noqa: F401
    if "projects" not in sys.modules:
        sys.modules["projects"] = types.ModuleType("projects")
    if "projects.CO-DETR" not in sys.modules:
        sys.modules["projects.CO-DETR"] = types.ModuleType("projects.CO-DETR")
    sys.modules["projects.CO-DETR.codetr"] = codetr

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg_path = (CODETR_PROJ / "configs" / "codino"
                / "co_dino_5scale_r50_8xb2_1x_coco.py")
    cfg = Config.fromfile(str(cfg_path))

    # --- Model overrides --------------------------------------------------
    _set_num_classes(cfg.model, num_classes)
    # Disable LSJ batch augments for smoke (already done in this config, but be safe)
    if "data_preprocessor" in cfg.model:
        cfg.model.data_preprocessor.batch_augments = None
        cfg.model.data_preprocessor.pad_mask = False
    cfg.model.use_lsj = False

    # --- Pipelines (640 short side, rect, long capped at 2000) -----------
    scale = (args.width, args.height)
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Resize", scale=scale, keep_ratio=True),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=scale, keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs",
             meta_keys=("img_id", "img_path", "ori_shape",
                        "img_shape", "scale_factor")),
    ]

    metainfo = dict(classes=tuple(class_names))
    data_root = str(DATASET_ROOT)

    cfg.train_dataloader = dict(
        batch_size=args.batch,
        num_workers=2,
        persistent_workers=False,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            metainfo=metainfo,
            ann_file=str(train_json),
            data_prefix=dict(img="train_set/"),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline,
        ),
    )
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            metainfo=metainfo,
            ann_file=str(val_json),
            data_prefix=dict(img="train_set/"),
            test_mode=True,
            pipeline=test_pipeline,
        ),
    )
    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator = dict(
        type="CocoMetric",
        ann_file=str(val_json),
        metric="bbox",
        format_only=False,
    )
    cfg.test_evaluator = cfg.val_evaluator

    # --- Training loop ----------------------------------------------------
    cfg.train_cfg = dict(type="EpochBasedTrainLoop",
                         max_epochs=args.epochs, val_interval=1)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")

    cfg.optim_wrapper = dict(
        type="AmpOptimWrapper" if args.amp else "OptimWrapper",
        optimizer=dict(type="AdamW", lr=args.lr, weight_decay=1e-4),
        clip_grad=dict(max_norm=0.1, norm_type=2),
        paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
    )
    if args.amp:
        cfg.optim_wrapper["loss_scale"] = "dynamic"
    cfg.param_scheduler = [
        dict(type="MultiStepLR", begin=0, end=args.epochs,
             by_epoch=True, milestones=[max(1, args.epochs - 1)], gamma=0.1)
    ]

    cfg.default_hooks = dict(
        timer=dict(type="IterTimerHook"),
        logger=dict(type="LoggerHook", interval=10),
        param_scheduler=dict(type="ParamSchedulerHook"),
        checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2),
        sampler_seed=dict(type="DistSamplerSeedHook"),
        visualization=dict(type="DetVisualizationHook"),
    )

    # --- Visualizer (handles WandB) --------------------------------------
    vis_backends = [dict(type="LocalVisBackend")]
    if use_wandb:
        vis_backends.append(dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project="fathomnet2026",
                entity="little_farms",
                name=run_name,
            ),
        ))
    cfg.visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer",
    )

    cfg.work_dir = str(out_dir / "mmdet_run")
    cfg.launcher = "none"
    cfg.load_from = None
    cfg.resume = False
    cfg.randomness = dict(seed=42, deterministic=False)
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

    if artifact_dir:
        cfg_dump_path = artifact_dir / "resolved-config.py"
        cfg.dump(str(cfg_dump_path))
        write_json(
            artifact_dir / "metadata.json",
            {
                "artifact_dir": relpath_str(artifact_dir, CHALLENGE_ROOT),
                "config_source": relpath_str(cfg_path, CHALLENGE_ROOT),
                "git_commit": git_commit(CHALLENGE_ROOT),
                "kind": "training",
                "metrics_artifact": relpath_str(artifact_dir / "metrics.jsonl", CHALLENGE_ROOT),
                "output_root": relpath_str(out_dir, CHALLENGE_ROOT),
                "platform": "local",
                "resolved_config": relpath_str(cfg_dump_path, CHALLENGE_ROOT),
                "run_id": args.run_id,
                "run_name": run_name,
                "split_manifest": relpath_str(split_manifest_path, CHALLENGE_ROOT),
                "stdout_log": relpath_str(artifact_dir / "stdout.log", CHALLENGE_ROOT),
                "track": "Co-DETR",
            },
        )

    print(f"[cfg] model={cfg.model.type}  num_classes={num_classes}")
    print(f"[cfg] work_dir={cfg.work_dir}")
    print(f"[cfg] run_name={run_name}")
    print(f"[cfg] split_manifest={split_manifest_path}")
    print(f"[cfg] epochs={args.epochs} batch={args.batch} scale={scale} wandb={use_wandb}")

    runner = Runner.from_cfg(cfg)
    runner.train()

    if artifact_dir:
        run_root = Path(cfg.work_dir)
        vis_data_root = max(run_root.glob("*/vis_data"), default=None)
        if vis_data_root is not None:
            copy_if_exists(vis_data_root / "scalars.json", artifact_dir / "metrics.jsonl")
            copy_if_exists(vis_data_root / "config.py", artifact_dir / "visualizer-config.py")
        latest_log = max(run_root.glob("*/*.log"), default=None)
        if latest_log is not None:
            copy_if_exists(latest_log, artifact_dir / "framework.log")

        best_ckpts = sorted(run_root.glob("best_*.pth"))
        selected_ckpt = best_ckpts[-1] if best_ckpts else None
        if selected_ckpt is not None:
            write_json(
                artifact_dir / "selected-checkpoint.json",
                {
                    "relative_path": relpath_str(selected_ckpt, CHALLENGE_ROOT),
                    "selection": "training framework best checkpoint",
                },
            )
    print("OK")


if __name__ == "__main__":
    main()
