#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — Co-DETR R50 baseline (Kaggle T4x2).

Setup required ONCE in the Kaggle UI before pushing/running:
  1. Add a Kaggle Secret named WANDB_API_KEY (Add-ons > Secrets).
  2. Set Accelerator to "GPU T4 x2" (kernel-metadata.json cannot pin
     dual-T4 — only enable_gpu=true. Set this in the UI before run).
  3. Internet ON, Persistence Off.

Inputs (from kernel-metadata.json):
  /kaggle/input/fathomnet-2026/                  -> JSONs only (no images)
  /kaggle/input/fathomnetclef2026/images/{train,test}/<uuid>.jpg -> image mirror

Outputs:
  /kaggle/working/idx_to_cat_id.json   (idx -> original COCO cat_id)
  /kaggle/working/run/epoch_*.pth      mmengine checkpoints
  /kaggle/working/submission.csv

Notes:
  - AMP is ENABLED with two runtime patches:
      1. batched_nms: mmcv 2.1.0 dtype mismatch (fp16 scores → fp32 cast).
      2. sigmoid_focal_loss: CoATSSHead FocalLoss fp16 overflow → fp32 cast
         on pred only (target must remain torch.long for the CUDA kernel).
  - Training is launched in single-GPU mode; mmengine multi-GPU on Kaggle
    requires torchrun and is disabled here for robustness. T4x2 still
    benefits because torchvision uses GPU0 and inference can also use it.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Install deps (cite challenge requirements-mmdet.txt; pinned subset).
#    mmcv must use the openmim wheel matching torch 2.x + cu12.
# ---------------------------------------------------------------------------
print("[setup] pip installing deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openmim==0.3.9",
                       "wandb==0.26.1", "pycocotools==2.0.11"])
# Use mim for torch-aware mmcv install
subprocess.check_call([sys.executable, "-m", "mim", "install", "-q",
                       "mmengine==0.10.7", "mmcv==2.1.0", "mmdet==3.3.0"])

# ---------------------------------------------------------------------------
# 2. Clone mmdetection v3.3.0 to get the Co-DETR project source.
# ---------------------------------------------------------------------------
MMDET_SRC = Path("/kaggle/working/mmdetection-src")
if not MMDET_SRC.exists():
    print("[setup] cloning mmdetection v3.3.0")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "v3.3.0",
        "https://github.com/open-mmlab/mmdetection.git", str(MMDET_SRC),
    ])

CODETR_PROJ = MMDET_SRC / "projects" / "CO-DETR"
sys.path.insert(0, str(CODETR_PROJ))
sys.path.insert(0, str(MMDET_SRC))

# ---------------------------------------------------------------------------
# 3. WandB key from Kaggle Secrets.
# ---------------------------------------------------------------------------
USE_WANDB = False
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    USE_WANDB = True
    print("[setup] WANDB_API_KEY loaded from Kaggle Secrets")
except Exception as e:
    print(f"[setup] WandB disabled (no secret): {e}")
    os.environ["WANDB_DISABLED"] = "true"

# ---------------------------------------------------------------------------
# 4. Paths.
# ---------------------------------------------------------------------------
COMP_ROOT = Path("/kaggle/input/fathomnet-2026")
IMG_ROOT = Path("/kaggle/input/fathomnetclef2026/images")
TRAIN_JSON_RAW = COMP_ROOT / "train_dataset.json"
TEST_JSON_RAW = COMP_ROOT / "test_dataset.json"
WORK = Path("/kaggle/working")

assert TRAIN_JSON_RAW.exists()
assert TEST_JSON_RAW.exists()
assert IMG_ROOT.exists(), f"Missing image mirror at {IMG_ROOT}"

# Build symlink farm so mmdet sees a single image root with file_name == on-disk name.
# JSON references *.png; mirror is *.jpg. We'll rewrite file_name to match the
# resolved filename in the symlink farm.
IMG_FARM = WORK / "imgs"
TRAIN_FARM = IMG_FARM / "train"
TEST_FARM = IMG_FARM / "test"
TRAIN_FARM.mkdir(parents=True, exist_ok=True)
TEST_FARM.mkdir(parents=True, exist_ok=True)


def _build_stem_index(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir()}


print("[data] indexing image mirrors")
TRAIN_IDX = _build_stem_index(IMG_ROOT / "train")
TEST_IDX = _build_stem_index(IMG_ROOT / "test")


def _link_into_farm(src: Path, dst_dir: Path) -> str:
    dst = dst_dir / src.name
    if not dst.exists():
        try:
            dst.symlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
    return src.name


# ---------------------------------------------------------------------------
# 5. Build contiguous category map + 90/10 split COCO JSONs (from train_mmdet.py).
# ---------------------------------------------------------------------------
def prepare_splits(train_json: Path, out_dir: Path, seed: int = 42):
    coco = json.loads(train_json.read_text())
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    idx_to_cat_id = {i: c["id"] for i, c in enumerate(cats_sorted)}
    new_categories = [
        {"id": i, "name": c["name"], "supercategory": c.get("supercategory", "")}
        for i, c in enumerate(cats_sorted)
    ]
    class_names = [c["name"] for c in new_categories]

    # Re-link images and rewrite file_name to match the symlink farm.
    new_images = []
    missing = 0
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        src = TRAIN_IDX.get(stem)
        if src is None:
            missing += 1
            continue
        fname = _link_into_farm(src, TRAIN_FARM)
        new_im = dict(im)
        new_im["file_name"] = fname
        new_images.append(new_im)
    print(f"[data] linked {len(new_images)} train imgs, missing {missing}")
    valid_ids = {im["id"] for im in new_images}

    remapped_anns = [
        {**a, "category_id": cat_id_to_idx[a["category_id"]]}
        for a in coco["annotations"] if a["image_id"] in valid_ids
    ]

    rng = random.Random(seed)
    images = list(new_images)
    rng.shuffle(images)
    n_val = max(1, int(round(0.1 * len(images))))
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]
    val_ids = {im["id"] for im in val_imgs}

    train_anns = [a for a in remapped_anns if a["image_id"] not in val_ids]
    val_anns = [a for a in remapped_anns if a["image_id"] in val_ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "coco_train.json"
    val_path = out_dir / "coco_val.json"
    common = dict(categories=new_categories, info=coco.get("info", {}),
                  licenses=coco.get("licenses", []))
    train_path.write_text(json.dumps(
        {"images": train_imgs, "annotations": train_anns, **common}))
    val_path.write_text(json.dumps(
        {"images": val_imgs, "annotations": val_anns, **common}))
    print(f"[data] train={len(train_imgs)} / val={len(val_imgs)} / classes={len(class_names)}")
    return train_path, val_path, class_names, idx_to_cat_id


def prepare_test(test_json: Path, out_dir: Path):
    coco = json.loads(test_json.read_text())
    new_images = []
    missing = 0
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        src = TEST_IDX.get(stem)
        if src is None:
            missing += 1
            continue
        fname = _link_into_farm(src, TEST_FARM)
        new_im = dict(im)
        new_im["file_name"] = fname
        new_images.append(new_im)
    print(f"[data] linked {len(new_images)} test imgs, missing {missing}")
    test_path = out_dir / "coco_test.json"
    test_path.write_text(json.dumps(
        {"images": new_images, "annotations": [],
         "categories": coco.get("categories", []),
         "info": coco.get("info", {}), "licenses": coco.get("licenses", [])}))
    return test_path


train_json, val_json, class_names, idx_to_cat_id = prepare_splits(TRAIN_JSON_RAW, WORK)
test_json = prepare_test(TEST_JSON_RAW, WORK)
num_classes = len(class_names)
(WORK / "idx_to_cat_id.json").write_text(
    json.dumps({str(k): v for k, v in idx_to_cat_id.items()})
)

# ---------------------------------------------------------------------------
# 6. Register CO-DETR project (custom_imports key uses hyphenated path, alias it).
# ---------------------------------------------------------------------------
import codetr  # noqa: F401 — populates registries
sys.modules.setdefault("projects", types.ModuleType("projects"))
sys.modules.setdefault("projects.CO-DETR", types.ModuleType("projects.CO-DETR"))
sys.modules["projects.CO-DETR.codetr"] = codetr

from mmengine.config import Config
from mmengine.runner import Runner

# ---------------------------------------------------------------------------
# AMP patches — applied once before Runner builds the model.
# ---------------------------------------------------------------------------
# 1. batched_nms dtype mismatch: rpn_head scores buffer is fp16 but NMS op
#    returns fp32 dets. Cast inputs to fp32 before calling the underlying op.
import mmcv.ops as _ops_mod
import mmcv.ops.nms as _nms_mod
_orig_bn = _ops_mod.batched_nms

def _bn_fp32_safe(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    return _orig_bn(boxes.float(), scores.float(), idxs, nms_cfg, class_agnostic)

_ops_mod.batched_nms = _bn_fp32_safe
_nms_mod.batched_nms = _bn_fp32_safe
try:
    import mmdet.models.dense_heads.rpn_head as _rpn
    _rpn.batched_nms = _bn_fp32_safe
except Exception:
    pass

# 2. FocalLoss fp16 overflow: CoATSSHead sigmoid_focal_loss overflows in fp16
#    (large-magnitude pred values cause exp overflow → NaN). Cast only pred to
#    fp32; target must stay torch.long for the mmcv CUDA kernel assertion.
try:
    import mmdet.models.losses.focal_loss as _fl_mod
    _orig_sfl = _fl_mod.sigmoid_focal_loss

    def _sfl_fp32_safe(pred, target, weight=None, gamma=2.0, alpha=0.25,
                       reduction="mean", avg_factor=None):
        return _orig_sfl(pred.float(), target, weight, gamma, alpha,
                         reduction, avg_factor)

    _fl_mod.sigmoid_focal_loss = _sfl_fp32_safe
    import mmdet.models.losses.focal_loss as _fl2
    _fl2.sigmoid_focal_loss = _sfl_fp32_safe
except Exception:
    pass


def _set_num_classes(node, n: int):
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "num_classes":
                node[k] = n
            else:
                _set_num_classes(v, n)
    elif isinstance(node, list):
        for v in node:
            _set_num_classes(v, n)


# ---------------------------------------------------------------------------
# 7. Build runner config.
# ---------------------------------------------------------------------------
cfg_path = CODETR_PROJ / "configs" / "codino" / "co_dino_5scale_r50_8xb2_1x_coco.py"
cfg = Config.fromfile(str(cfg_path))

_set_num_classes(cfg.model, num_classes)
if "data_preprocessor" in cfg.model:
    cfg.model.data_preprocessor.batch_augments = None
    cfg.model.data_preprocessor.pad_mask = False
cfg.model.use_lsj = False

scale = (2000, 800)
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

cfg.train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CocoDataset",
        data_root=str(IMG_FARM),
        metainfo=metainfo,
        ann_file=str(train_json),
        data_prefix=dict(img="train/"),
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
        data_root=str(IMG_FARM),
        metainfo=metainfo,
        ann_file=str(val_json),
        data_prefix=dict(img="train/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
cfg.test_dataloader = cfg.val_dataloader
cfg.val_evaluator = dict(type="CocoMetric", ann_file=str(val_json),
                         metric="bbox", format_only=False)
cfg.test_evaluator = cfg.val_evaluator

EPOCHS = 12
cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=EPOCHS, val_interval=1)
cfg.val_cfg = dict(type="ValLoop")
cfg.test_cfg = dict(type="TestLoop")

# eff_bs = 4 (bs=2 single-GPU x grad-accum not used here);
# linear-scaled lr from base 2e-4 @ eff_bs=16 -> 5e-5.
cfg.optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)
cfg.param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=EPOCHS, by_epoch=True,
         milestones=[11], gamma=0.1)
]

cfg.default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2,
                    save_best="coco/bbox_mAP", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

RUN_NAME = "fathomnet-codetr-r50-baseline"
vis_backends = [dict(type="LocalVisBackend")]
if USE_WANDB:
    vis_backends.append(dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="fathomnet2026", entity="little_farms",
                         name=RUN_NAME),
    ))
cfg.visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends,
                      name="visualizer")

cfg.work_dir = str(Path("/kaggle/working/run"))
cfg.launcher = "none"
cfg.load_from = None
cfg.resume = False
cfg.randomness = dict(seed=42, deterministic=False)
cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

print(f"[cfg] num_classes={num_classes} scale={scale} epochs={EPOCHS}")
runner = Runner.from_cfg(cfg)
runner.train()

# ---------------------------------------------------------------------------
# 8. Test-set inference -> submission.csv.
# ---------------------------------------------------------------------------
print("[infer] running test-set inference")
import glob
import torch
from mmdet.apis import init_detector, inference_detector

# Pick best checkpoint (or last).
ckpts = sorted(glob.glob(str(Path(cfg.work_dir) / "best_*.pth"))) or \
        sorted(glob.glob(str(Path(cfg.work_dir) / "epoch_*.pth")))
assert ckpts, f"No checkpoint found in {cfg.work_dir}"
ckpt = ckpts[-1]
print(f"[infer] using checkpoint {ckpt}")

# Use a test_dataloader-style cfg for init.
# init_detector wants the config + checkpoint.
cfg.dump(str(Path(cfg.work_dir) / "config_used.py"))
detector = init_detector(str(Path(cfg.work_dir) / "config_used.py"),
                         ckpt, device="cuda:0")

test_coco = json.loads(test_json.read_text())
rows: list[dict] = []
ann_id = 1
SCORE_THR = 0.001
for im in test_coco["images"]:
    img_path = TEST_FARM / im["file_name"]
    if not img_path.exists():
        continue
    result = inference_detector(detector, str(img_path))
    pred = result.pred_instances
    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    for (x1, y1, x2, y2), s, c in zip(bboxes, scores, labels):
        if s < SCORE_THR:
            continue
        rows.append({
            "annotation_id": ann_id,
            "image_id": int(im["id"]),
            "category_id": int(idx_to_cat_id[int(c)]),
            "bbox_x": float(x1),
            "bbox_y": float(y1),
            "bbox_width": float(x2 - x1),
            "bbox_height": float(y2 - y1),
            "score": float(s),
        })
        ann_id += 1

import csv
SUB = WORK / "submission.csv"
with SUB.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score",
    ])
    w.writeheader()
    w.writerows(rows)
print(f"[infer] wrote {len(rows)} rows")
print(f"Submission CSV: /kaggle/working/submission.csv")
