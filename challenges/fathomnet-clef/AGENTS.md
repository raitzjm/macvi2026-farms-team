# FathomNet-CLEF 2026 — Challenge Instructions

## Challenge
Positive-Unlabeled (PU) object detection in marine underwater images.
31 categories, COCO bbox format, evaluated by mAP@[0.50:0.95].
Deadline: **2026-05-07**.

## Key Documentation (read before acting)

| Document | Location | Contents |
|---|---|---|
| Competition reference | `docs/challenge-overview.md` | Full verbatim competition text: overview, data, rules, evaluation, submission format, timeline |
| Kaggle CLI & Python SDK | `../../docs/agent-reference/kaggle-cli.md` | How to fetch pages, submit, manage kernels, access discussions via SDK |
| Gemini delegation | `../../docs/agent-reference/gemini-delegation.md` | How to delegate to Gemini Flash |
| OpenCode delegation | `../../docs/agent-reference/opencode-delegation.md` | How to delegate to MiniMax, Nemotron, GPT |

## Dataset (read-only)
```
/home/farms/data/datasets/fathomnet-2026/
  train_dataset.json   # 6,463 images, 22,225 annotations, 32 categories (COCO format)
  test_dataset.json    # 1,425 images, no annotations (held out)
  train_set/           # PNG images
  test_set/            # PNG images
```

## Challenge Directory Structure
```
challenges/fathomnet-clef/
  AGENTS.md                     # this file
  CLAUDE.md                     # Claude compatibility stub (`@AGENTS.md`)
  docs/
    challenge-overview.md       # complete competition reference (verbatim)
    discussions/                # saved discussion threads
  artifacts/                    # curated checkpoints and submission CSVs
  configs/                      # model configs (YOLO, MMDet)
  scripts/
    _paths.py                   # multi-environment data path resolution
    infer_codetr.py             # local Co-DETR inference → CSV pipeline
    infer_yolo.py               # local YOLO inference → CSV pipeline
    make_submission.py          # inference → CSV pipeline
  training-log.md               # training run tracker
  inference-log.md              # inference and Kaggle submission tracker
  outputs/                      # gitignored — training outputs, checkpoints
  envs/                         # gitignored — conda environments
```

## Multi-Environment Data Paths
`scripts/_paths.py` resolves the dataset root automatically:
- Kaggle kernel: `/kaggle/input/fathomnet-2026/`
- TensorDock: `$FATHOMNET_DATA_DIR`
- Local/container: `/home/farms/data/datasets/fathomnet-2026/`

## Submission Format
CSV with columns: `annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score`

Submit via: `kaggle competitions submit -c fathomnet-2026 -f submission.csv -m "description"`

Track training runs in `training-log.md` and every generated or official submission in `inference-log.md`.
There is intentionally no separate `submissions.md` for this challenge.

## Rules Summary (read full rules in docs/challenge-overview.md)
- ✅ Pretrained models (ImageNet, iNaturalist, Megalodon, Megafishdetector): allowed, must disclose
- ✅ Relabeling / re-annotating provided competition data: allowed
- ❌ External image datasets (including additional FathomNet Database images): not allowed
- ❌ Web crawling for additional data: not allowed
- Transparency is required — disclose all external models/resources in the Kaggle Discussion tab

*Source: Official organizer ruling in discussion #688810 (2026-04-08)*

## Environments
- `bootstrap-ultralytics.sh` → `./envs/ultralytics/` — YOLOv11, SAHI, pycocotools
- `bootstrap-mmdet.sh` → `./envs/mmdet/` — MMDetection, Co-DETR, SAHI for local tiled inference

## Documentation Maintenance
If you discover errors in the docs, new CLI commands, or new SDK capabilities while working on this challenge, update the relevant documentation immediately:
- Competition rule clarifications → `docs/challenge-overview.md`
- New Kaggle CLI/SDK patterns → `../../docs/agent-reference/kaggle-cli.md`
- New discussion findings → `docs/discussions/`

The `training-log.md` and `inference-log.md` formats may be adapted as the challenge evolves — if you find either format insufficient for a new type of experiment or dependency pattern, update the relevant file and note the change in its header.

Do not leave stale or incorrect documentation — future agents depend on it.

## Evaluation Notebook
`kaggle/code/lauravchrobak/map50-95` is a **reference implementation of the metric** — not a submission tool. It shows how mAP@[.50:.95] is computed via pycocotools. You can use its logic to evaluate your model on the micro-val split locally. To get official competition scores, you must submit a CSV to Kaggle — test annotations are withheld server-side.

## Agent Conventions
- All commits: `[agent]` prefix, body includes model name
- Do not run full training locally (RTX 3050 4GB) — use TensorDock or Kaggle
- Outputs go to `outputs/` (gitignored)
- Curated checkpoints and submission CSVs go to `artifacts/`; checkpoints are Git LFS tracked
- Training runs are tracked in `training-log.md`; inference and official Kaggle submissions are tracked in `inference-log.md`
- WandB credentials must come from the `WANDB_API_KEY` environment variable; do not add file-based secret fallbacks in tracked scripts
- This challenge intentionally does not maintain `STATE.md`; status lives in the repo-level challenge table and this AGENTS.md
