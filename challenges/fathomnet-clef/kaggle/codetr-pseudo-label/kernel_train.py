#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — Co-DETR R50 pseudo-label training (Kaggle T4x2).

Matches T-004 config exactly: scale=(768,432), RandomFlip only, LR=1e-4,
AMP, effective batch=10 via accumulative_counts=5 on 2xT4 (1 sample/GPU).
Uses shared-train-val-v1 split manifest (embedded inline).

Flags at top of file:
  DRY_RUN         — True=1 epoch no inference, False=12 epoch full run
  PSEUDO_LABEL_JSON — path to pseudo-label COCO JSON, or None for GT-only
  RUN_ID          — training run ID string (e.g. "T-009")

Setup required ONCE in Kaggle UI before pushing:
  1. Add Kaggle Secret: WANDB_API_KEY
  2. Accelerator: GPU T4 x2
  3. Internet: ON, Persistence: OFF
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CONFIG — edit these before pushing
# ---------------------------------------------------------------------------
DRY_RUN = True          # True = 1 epoch sanity check; False = full 12-epoch run
PSEUDO_LABEL_JSON = None  # path to pseudo-label COCO JSON once generated
RUN_ID = "T-009"
# ---------------------------------------------------------------------------

import atexit
import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Shared-train-val-v1 split manifest (embedded from artifacts/split-manifests/)
# seed=42, 5816 train / 647 val
# ---------------------------------------------------------------------------
_TRAIN_IDS_SET: set[int] | None = None
_VAL_IDS_SET: set[int] | None = None

def _load_split() -> tuple[set[int], set[int]]:
    global _TRAIN_IDS_SET, _VAL_IDS_SET
    if _TRAIN_IDS_SET is not None:
        return _TRAIN_IDS_SET, _VAL_IDS_SET
    # Inline manifest — generated from artifacts/split-manifests/shared-train-val-v1.json
    import importlib.util, pathlib
    _here = pathlib.Path(__file__).parent
    _mf = _here / "split_manifest.json"
    if _mf.exists():
        _m = json.loads(_mf.read_text())
        _TRAIN_IDS_SET = set(_m["train_ids"])
        _VAL_IDS_SET = set(_m["val_ids"])
    else:
        # fallback: download from repo if internet available
        import urllib.request
        _url = ("https://raw.githubusercontent.com/raitzjm/macvi2026-farms-team"
                "/main/challenges/fathomnet-clef/artifacts/split-manifests"
                "/shared-train-val-v1.json")
        print(f"[split] fetching manifest from {_url}")
        _data = json.loads(urllib.request.urlopen(_url).read())
        _TRAIN_IDS_SET = set(_data["train_ids"])
        _VAL_IDS_SET = set(_data["val_ids"])
    print(f"[split] train={len(_TRAIN_IDS_SET)} val={len(_VAL_IDS_SET)}")
    return _TRAIN_IDS_SET, _VAL_IDS_SET


# ---------------------------------------------------------------------------
# 1. Install deps
# ---------------------------------------------------------------------------
print("[setup] detecting torch/CUDA for mmcv wheel")
import torch as _torch
_cuda_tag = "cu" + _torch.version.cuda.replace(".", "")
_torch_tag = "torch" + ".".join(_torch.__version__.split(".")[:2])
_mmcv_url = f"https://download.openmmlab.com/mmcv/dist/{_cuda_tag}/{_torch_tag}/index.html"
print(f"[setup] torch={_torch.__version__} cuda={_torch.version.cuda}")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "mmengine==0.10.7", "mmdet==3.3.0",
                       "wandb==0.26.1", "pycocotools==2.0.11",
                       "ensemble-boxes==1.0.9"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "mmcv==2.1.0", "-f", _mmcv_url])

# ---------------------------------------------------------------------------
# 2. Clone mmdetection v3.3.0 for CO-DETR project source
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
# 3. Logging + WandB
# ---------------------------------------------------------------------------
WORK = Path("/kaggle/working")

class _TeeWriter:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_raw_stdout = sys.stdout
_raw_stderr = sys.stderr
_log_file = (WORK / "train.log").open("a", encoding="utf-8", buffering=1)
sys.stdout = _TeeWriter(_raw_stdout, _log_file)
sys.stderr = _TeeWriter(_raw_stderr, _log_file)
atexit.register(_log_file.close)

USE_WANDB = False
if not DRY_RUN:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        USE_WANDB = True
        print("[setup] WandB enabled")
    except Exception:
        os.environ.pop("WANDB_API_KEY", None)
        print("[setup] WandB disabled: secret not available")
else:
    os.environ["WANDB_DISABLED"] = "true"
    print("[setup] WandB disabled (dry run)")

# ---------------------------------------------------------------------------
# 4. Paths — competition JSONs + image mirror
# ---------------------------------------------------------------------------
_candidates = sorted(Path("/kaggle/input").rglob("train_dataset.json"))
assert _candidates, f"train_dataset.json not found. /kaggle/input: {list(Path('/kaggle/input').iterdir())}"
COMP_ROOT = _candidates[0].parent
print(f"[data] competition root: {COMP_ROOT}")
TRAIN_JSON_RAW = COMP_ROOT / "train_dataset.json"
TEST_JSON_RAW = COMP_ROOT / "test_dataset.json"

_img_candidates = [
    Path("/kaggle/input/fathomnetclef2026/images"),
    Path("/kaggle/input/fathomnetclef2026"),
]
IMG_ROOT = next((p for p in _img_candidates if (p / "train").exists()), None)
if IMG_ROOT is None:
    for _p in Path("/kaggle/input").rglob("train"):
        if _p.is_dir() and any(_p.iterdir()):
            IMG_ROOT = _p.parent
            break
assert IMG_ROOT is not None, f"Image mirror not found. /kaggle/input: {list(Path('/kaggle/input').iterdir())}"
print(f"[data] image root: {IMG_ROOT}")

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
# 5. Build COCO JSONs using shared split manifest
# ---------------------------------------------------------------------------
def prepare_data(train_json_raw: Path, pseudo_label_json: Path | None,
                 out_dir: Path) -> tuple[Path, Path, list[str], dict[int, int]]:
    coco = json.loads(train_json_raw.read_text())
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    idx_to_cat_id = {i: c["id"] for i, c in enumerate(cats_sorted)}
    new_categories = [
        {"id": i, "name": c["name"], "supercategory": c.get("supercategory", "")}
        for i, c in enumerate(cats_sorted)
    ]
    class_names = [c["name"] for c in new_categories]

    train_ids, val_ids = _load_split()

    new_images, missing = [], 0
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        src = TRAIN_IDX.get(stem)
        if src is None:
            missing += 1
            continue
        fname = _link_into_farm(src, TRAIN_FARM)
        new_images.append({**im, "file_name": fname})
    print(f"[data] linked {len(new_images)} train imgs, missing {missing}")

    valid_ids = {im["id"] for im in new_images}
    remapped_anns = [
        {**a, "category_id": cat_id_to_idx[a["category_id"]]}
        for a in coco["annotations"] if a["image_id"] in valid_ids
    ]

    train_imgs = [im for im in new_images if im["id"] in train_ids]
    val_imgs   = [im for im in new_images if im["id"] in val_ids]
    train_img_ids = {im["id"] for im in train_imgs}
    val_img_ids   = {im["id"] for im in val_imgs}

    train_anns = [a for a in remapped_anns if a["image_id"] in train_img_ids]
    val_anns   = [a for a in remapped_anns if a["image_id"] in val_img_ids]

    # Merge pseudo-labels into training annotations (already remapped to 0-based idx)
    if pseudo_label_json is not None and Path(pseudo_label_json).exists():
        pl = json.loads(Path(pseudo_label_json).read_text())
        pl_anns = pl.get("annotations", [])
        # pseudo-labels use original cat_ids — remap to idx
        pl_anns_remapped = []
        for a in pl_anns:
            if a["image_id"] in train_img_ids and a["category_id"] in cat_id_to_idx:
                pl_anns_remapped.append({**a, "category_id": cat_id_to_idx[a["category_id"]]})
        print(f"[data] merged {len(pl_anns_remapped)} pseudo-label annotations")
        train_anns = train_anns + pl_anns_remapped
    else:
        print(f"[data] no pseudo-labels (GT-only)")

    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(categories=new_categories, info=coco.get("info", {}),
                  licenses=coco.get("licenses", []))
    train_path = out_dir / "coco_train.json"
    val_path   = out_dir / "coco_val.json"
    train_path.write_text(json.dumps({"images": train_imgs, "annotations": train_anns, **common}))
    val_path.write_text(json.dumps({"images": val_imgs, "annotations": val_anns, **common}))
    print(f"[data] train={len(train_imgs)} imgs / {len(train_anns)} anns  "
          f"val={len(val_imgs)} imgs / {len(val_anns)} anns  classes={len(class_names)}")
    return train_path, val_path, class_names, idx_to_cat_id


def prepare_test(test_json_raw: Path, out_dir: Path) -> Path:
    coco = json.loads(test_json_raw.read_text())
    new_images, missing = [], 0
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        src = TEST_IDX.get(stem)
        if src is None:
            missing += 1
            continue
        fname = _link_into_farm(src, TEST_FARM)
        new_images.append({**im, "file_name": fname})
    print(f"[data] linked {len(new_images)} test imgs, missing {missing}")
    test_path = out_dir / "coco_test.json"
    test_path.write_text(json.dumps({
        "images": new_images, "annotations": [],
        "categories": coco.get("categories", []),
        "info": coco.get("info", {}), "licenses": coco.get("licenses", []),
    }))
    return test_path


train_json, val_json, class_names, idx_to_cat_id = prepare_data(
    TRAIN_JSON_RAW, PSEUDO_LABEL_JSON, WORK)
test_json = prepare_test(TEST_JSON_RAW, WORK) if not DRY_RUN else None
num_classes = len(class_names)
(WORK / "idx_to_cat_id.json").write_text(
    json.dumps({str(k): v for k, v in idx_to_cat_id.items()}))

# ---------------------------------------------------------------------------
# 6. Register CO-DETR project
# ---------------------------------------------------------------------------
import codetr  # noqa: F401
sys.modules.setdefault("projects", types.ModuleType("projects"))
sys.modules.setdefault("projects.CO-DETR", types.ModuleType("projects.CO-DETR"))
sys.modules["projects.CO-DETR.codetr"] = codetr

from mmengine.config import Config
from mmengine.runner import Runner

# ---------------------------------------------------------------------------
# 7. AMP patches (same as T-003, required for mmcv 2.1.0 + mmdet 3.3.0)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 8. Build runner config — T-004 faithful
# ---------------------------------------------------------------------------
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


cfg_path = CODETR_PROJ / "configs" / "codino" / "co_dino_5scale_r50_8xb2_1x_coco.py"
cfg = Config.fromfile(str(cfg_path))

_set_num_classes(cfg.model, num_classes)
if "data_preprocessor" in cfg.model:
    cfg.model.data_preprocessor.batch_augments = None
    cfg.model.data_preprocessor.pad_mask = False
cfg.model.use_lsj = False

# T-004 exact pipeline: RandomFlip + fixed Resize(768, 432)
SCALE = (768, 432)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=SCALE, keep_ratio=True),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=SCALE, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

metainfo = dict(classes=tuple(class_names))

# Effective batch = samples_per_gpu(1) × num_gpus(2) × accumulative_counts(5) = 10
# Matches T-004: samples_per_gpu=5 × num_gpus=2 = 10 (no grad accum)
cfg.train_dataloader = dict(
    batch_size=1,
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

EPOCHS = 1 if DRY_RUN else 12
cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=EPOCHS, val_interval=1)
cfg.val_cfg = dict(type="ValLoop")
cfg.test_cfg = dict(type="TestLoop")

# T-004 exact optimizer: AdamW lr=1e-4, clip_grad, backbone lr_mult=0.1, AMP dynamic
# accumulative_counts=5 to match effective batch=10 on 2xT4 with batch_size=1/GPU
cfg.optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    accumulative_counts=5,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)
cfg.param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=EPOCHS, by_epoch=True,
         milestones=[max(1, EPOCHS - 1)], gamma=0.1),
]

cfg.default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3,
                    save_best="coco/bbox_mAP", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook", draw=False),
)

RUN_SUFFIX = "dry-run" if DRY_RUN else "pseudo-label"
RUN_NAME = f"codetr-r50-{RUN_SUFFIX}-{RUN_ID}"
vis_backends = [dict(type="LocalVisBackend")]
if USE_WANDB:
    vis_backends.append(dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="fathomnet2026", entity="little_farms", name=RUN_NAME),
    ))
cfg.visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends,
                      name="visualizer")

cfg.work_dir = str(WORK / "run")
cfg.load_from = "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth"
cfg.resume = False
cfg.randomness = dict(seed=42, deterministic=False)
cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

print(f"[cfg] num_classes={num_classes} scale={SCALE} epochs={EPOCHS} "
      f"dry_run={DRY_RUN} pseudo_labels={'yes' if PSEUDO_LABEL_JSON else 'no'}")
print(f"[cfg] effective_batch=1gpu×2×accum5=10  lr=1e-4  run={RUN_NAME}")

import torch
_n_gpu = torch.cuda.device_count()
_local_rank = os.environ.get("LOCAL_RANK")

# ---------------------------------------------------------------------------
# 9. Train
# ---------------------------------------------------------------------------
if _n_gpu >= 2 and _local_rank is None:
    cfg.launcher = "pytorch"
    print(f"[train] launching torchrun on {_n_gpu} GPUs")
    subprocess.check_call([
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(_n_gpu),
        "--master_port", "29500",
        sys.argv[0],
    ])
elif _local_rank is not None:
    cfg.launcher = "pytorch"
    runner = Runner.from_cfg(cfg)
    runner.train()
    sys.exit(0)
else:
    cfg.launcher = "none"
    runner = Runner.from_cfg(cfg)
    runner.train()

# ---------------------------------------------------------------------------
# 10. Inference → submission.csv  (skipped for dry run)
# ---------------------------------------------------------------------------
if DRY_RUN:
    print("[dry-run] skipping inference — check train.log for epoch timing")
    sys.exit(0)

import glob
import shutil

_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

ckpts = (sorted(glob.glob(str(Path(cfg.work_dir) / "best_*.pth")))
         or sorted(glob.glob(str(Path(cfg.work_dir) / "epoch_*.pth"))))
assert ckpts, f"No checkpoint found in {cfg.work_dir}"
ckpt = ckpts[-1]
print(f"[infer] using checkpoint: {ckpt}")

from mmdet.apis import init_detector, inference_detector
cfg.dump(str(Path(cfg.work_dir) / "config_used.py"))
detector = init_detector(str(Path(cfg.work_dir) / "config_used.py"), ckpt, device="cuda:0")
shutil.copy2(str(Path(cfg.work_dir) / "config_used.py"), WORK / "resolved-config.py")

_run_log_root = max(Path(cfg.work_dir).glob("*/"), default=None)
if _run_log_root:
    for _src, _dst in [
        (_run_log_root / f"{_run_log_root.name}.log", WORK / "framework.log"),
        (_run_log_root / "vis_data" / "scalars.json", WORK / "metrics.jsonl"),
    ]:
        if _src.exists():
            shutil.copy2(_src, _dst)

test_coco = json.loads(test_json.read_text())
rows: list[dict] = []
ann_id = 1
SCORE_THR = 0.001
print(f"[infer] running on {len(test_coco['images'])} test images")
for im in test_coco["images"]:
    img_path = TEST_FARM / im["file_name"]
    if not img_path.exists():
        continue
    result = inference_detector(detector, str(img_path))
    pred = result.pred_instances
    for (x1, y1, x2, y2), s, c in zip(
            pred.bboxes.cpu().numpy(),
            pred.scores.cpu().numpy(),
            pred.labels.cpu().numpy()):
        if s < SCORE_THR:
            continue
        rows.append({
            "annotation_id": ann_id,
            "image_id": int(im["id"]),
            "category_id": int(idx_to_cat_id[int(c)]),
            "bbox_x": float(x1), "bbox_y": float(y1),
            "bbox_width": float(x2 - x1), "bbox_height": float(y2 - y1),
            "score": float(s),
        })
        ann_id += 1

SUB = WORK / "submission.csv"
with SUB.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score"])
    w.writeheader()
    w.writerows(rows)
print(f"[infer] wrote {len(rows)} rows → {SUB}")

# ---------------------------------------------------------------------------
# 11. Cleanup mmdetection-src + zip run/
# ---------------------------------------------------------------------------
if MMDET_SRC.exists():
    shutil.rmtree(str(MMDET_SRC))
    print(f"[cleanup] removed {MMDET_SRC}")
shutil.make_archive("/kaggle/working/run_checkpoint", "zip", str(Path(cfg.work_dir)))
print("[ckpt] checkpoint zip ready: /kaggle/working/run_checkpoint.zip")
