# FathomNet-CLEF 2026 — Challenge Instructions

## Challenge
Positive-Unlabeled (PU) object detection in marine underwater images.
32 categories, COCO bbox format, evaluated by mAP@[0.50:0.95].
Deadline: **2026-05-07**.

## Key Documentation (read before acting)

| Document | Location | Contents |
|---|---|---|
| Competition reference | `docs/challenge-overview.md` | Full verbatim competition text: overview, data, rules, evaluation, submission format, timeline |
| Run tracking policy | `docs/run-tracking-policy.md` | Required policy for training, inference, Kaggle/TensorDock run logging, artifacts, configs, and historical backfills |
| TensorDock remote workflow | `docs/tensordock-remote-workflow.md` | Policy for SSH-based TensorDock orchestration, remote git sync, host setup, disk monitoring, and access handling |
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
    run-tracking-policy.md      # required run/artifact/config/logging policy
    tensordock-remote-workflow.md # SSH-based TensorDock orchestration policy
    discussions/                # saved discussion threads
  artifacts/
    training-runs/              # committed — metadata JSON, metrics artifact, resolved config, raw training stdout.log when available
    inference-runs/             # committed — metadata JSON per inference run; raw inference stdout.log is optional/local
    checkpoints/                # NOT committed — local only; pull from remote via scp/rsync
    submissions/                # NOT committed — local only; pull from Kaggle CLI or remote
    split-manifests/            # committed — shared train/val split JSON files
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

Submit via: `kaggle competitions submit -c fathomnet-2026 -f submission-I-009-from-T-004.csv -m "description"`

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
- Use judgment for local training: full local runs are allowed when the model is fast enough to fit the available GPU and the run is expected to complete safely; prefer local YOLO runs when they are materially faster than remote execution, and use TensorDock or Kaggle for larger or expensive runs that do not fit well on the RTX 3050 4GB
- Outputs go to `outputs/` (gitignored)
- Checkpoints (`.pt`, `.pth`) and submission CSVs are **not committed to git** — store locally under `artifacts/checkpoints/` and `artifacts/submissions/` (both gitignored). Pull from remotes via `kaggle kernels output` (Kaggle) or `scp`/`rsync` (TensorDock). Only metadata JSON files and small results CSVs under `artifacts/training-runs/` and `artifacts/inference-runs/` are committed. *(History: checkpoints and submission CSVs were previously LFS-tracked; this was stopped during the T-006/T-007 sprint, commit 96b2eda5, to reduce repo bloat. At end of challenge, selectively LFS-track curated artifacts as needed.)*
- Raw training logs under `artifacts/training-runs/` are durable committed artifacts and must not be blocked by gitignore. Raw inference launcher logs are optional debugging artifacts: keep them locally when useful, but they do not need to be committed by default.
- Pseudo-label JSON files (< 30 MB) are committed directly to git (no LFS needed). Generated merged annotation files (GT + pseudo-labels, produced by `train_mmdet.py --pseudo-label-json`) are NOT committed — regenerate locally or on fathom1 before each training run. Everything produced on remote machines (fathom1, Kaggle) must be copied back to this machine via scp/rsync so the local repo is the canonical artifact hub.
- After any YOLO local training run completes, **immediately copy** `best.pt`, `last.pt`, and all periodic epoch checkpoints (e.g. `epoch5.pt` from `--save_period`) to `artifacts/checkpoints/<run-id>/` before any cleanup or output directory removal. T-008 lost its periodic checkpoints because this step was skipped — periodic epochs cannot be inferred after deletion and the fitness-best pick cannot be audited.
- Agents working on training, inference, Kaggle/TensorDock run execution, or logging/backfilling training or inference results must read and follow `docs/run-tracking-policy.md`
- Agents working on TensorDock orchestration or remote training execution must read and follow `docs/tensordock-remote-workflow.md`
- For FathomNet sprint execution, the primary agent should act as an orchestrator and manager: delegate implementation, monitoring, review, and other sidecar work to sub-agents where practical, and keep the top-level thread focused on coordination, decisions, and information flow between lead, worker, and reviewer agents
- WandB credentials must come from the `WANDB_API_KEY` environment variable; do not add file-based secret fallbacks in tracked scripts
- FathomNet Kaggle GPU kernels should keep `kernel-metadata.json` set to `"machine_shape": "NvidiaTeslaT4"` as the standard metadata value unless the lead explicitly changes the accelerator strategy
- All FathomNet training runs must log to WandB entity `little_farms`, project `fathomnet2026`
- Tracked training run names must append the training run ID suffix (example: `codetr-r50-baseline-T-004`)
- Training run IDs are only burned by retained tracked runs; if a launch is scrapped before it yields a run worth logging, reuse that `T-NNN` on the corrected relaunch rather than incrementing
- Tracked inference CSV filenames must append the inference run ID, and the source training run ID when known (example: `submission-I-009-from-T-004.csv`) so Kaggle UI entries remain identifiable
- Closely related experiment families that are being judged together (for example model-size comparisons or k-fold campaigns) should stay on one branch and one tracking issue unless the workstreams are operationally distinct enough to justify separate review and merge paths
- If a training launch fails after creating a WandB run, delete that failed WandB run before retrying so the project history stays clean
- Keep the sprint issue journal terse: post only when a run starts and survives the babysitting period, when a meaningful event or issue occurs, and when the run finishes
- For experiment-specific detail that would otherwise bloat the sprint issue, open a linked follow-up issue, document the experiment there, and close that follow-up issue when appropriate
- This challenge intentionally does not maintain `STATE.md`; status lives in the repo-level challenge table and this AGENTS.md
