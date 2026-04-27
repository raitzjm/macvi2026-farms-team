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
  CLAUDE.md                     # this file
  STATE.md                      # current_year: 2026, status: active
  docs/
    challenge-overview.md       # complete competition reference (verbatim)
    discussions/                # saved discussion threads
  configs/                      # model configs (YOLO, MMDet)
  scripts/
    _paths.py                   # multi-environment data path resolution
    make_submission.py          # inference → CSV pipeline
  submissions.md                # leaderboard tracker
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

## Rules Summary (read full rules in docs/challenge-overview.md)
- ✅ Pretrained models (ImageNet, iNaturalist, Megalodon, Megafishdetector): allowed, must disclose
- ✅ Relabeling / re-annotating provided competition data: allowed
- ❌ External image datasets (including additional FathomNet Database images): not allowed
- ❌ Web crawling for additional data: not allowed
- Transparency is required — disclose all external models/resources in the Kaggle Discussion tab

*Source: Official organizer ruling in discussion #688810 (2026-04-08)*

## Environments
- `bootstrap-ultralytics.sh` → `./envs/ultralytics/` — YOLO11, SAHI, pycocotools
- `bootstrap-mmdet.sh` → `./envs/mmdet/` — MMDetection, Co-DETR

## Documentation Maintenance
If you discover errors in the docs, new CLI commands, or new SDK capabilities while working on this challenge, update the relevant documentation immediately:
- Competition rule clarifications → `docs/challenge-overview.md`
- New Kaggle CLI/SDK patterns → `../../docs/agent-reference/kaggle-cli.md`
- New discussion findings → `docs/discussions/`

The `experiment-log.md` format may be adapted as the challenge evolves — if you find the format insufficient for a new type of experiment or dependency pattern, update the format and note the change in the file header.

Do not leave stale or incorrect documentation — future agents depend on it.

## Evaluation Notebook
`kaggle/code/lauravchrobak/map50-95` is a **reference implementation of the metric** — not a submission tool. It shows how mAP@[.50:.95] is computed via pycocotools. You can use its logic to evaluate your model on the micro-val split locally. To get official competition scores, you must submit a CSV to Kaggle — test annotations are withheld server-side.

## Agent Conventions
- All commits: `[agent]` prefix, body includes model name
- Do not run full training locally (RTX 3050 4GB) — use TensorDock or Kaggle
- Outputs go to `outputs/` (gitignored)
- Submissions tracked in `submissions.md`
