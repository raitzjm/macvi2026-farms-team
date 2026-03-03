# Archive Experiment Run

Use this skill after a training run completes to export, validate, and commit
the artifacts for a given phase and model.

## Logging issues and gotchas during training

Whenever you encounter an issue, unexpected behaviour, or a workaround during
training or config setup, append it to `challenges/embedded-semantic-segmentation/{phase}/notes.md`.
Create the file if it does not exist. Format each entry as:

```
### [{model_name}] Short description
What happened, how it was diagnosed, and how it was resolved or worked around.
```

This file is temporary — it is consumed when the phase report is written and
then deleted (see step 6 below).

## Steps

1. **Identify inputs** — confirm with the user:
   - Phase (e.g. `phase1`)
   - Model name (e.g. `stdc2`)
   - Best checkpoint path (e.g. `outputs/phase1/stdc2/best_mIoU_iter_10000.pth`)
   - Config path used for training

2. **Export ONNX** from the best checkpoint using the export script pattern in
   `fun_experiments/{model}/export_onnx.py`:
   - Output shape: `1×1×384×768`, dtype int64 (argmaxed class map)
   - ArgMax(keepdims=0) + Unsqueeze pattern (SNPE requirement)
   - Opset version: 13
   - Save as `outputs/{phase}/{model_name}/{model_name}.onnx`

3. **Validate ONNX precision** — run both eval scripts on the val set:
   ```
   python fun_experiments/scripts/eval_pytorch_val.py --checkpoint <pth> --config <config>
   python fun_experiments/scripts/eval_onnx_val.py --onnx <onnx_path>
   ```
   Compare mIoU values. Expected delta: <0.2%. A delta >0.5% is significant
   and warrants investigation before archiving.

4. **Prepare archive directory** at `challenges/embedded-semantic-segmentation/{phase}/{model_name}/`:
   - `config.py` — copy of the training config
   - `train.log` — training log from the work-dir
   - `{model_name}.onnx` — exported ONNX (committed via Git LFS)
   - `report.md` — brief summary: batch size, LR, final val mIoU, F1, pth vs ONNX delta
   - `{model_name}.pth` — ONLY if pth→ONNX delta exceeds 0.5% mIoU (committed via Git LFS)

5. **Update experiment-log.md** — add or complete the entry for this run:
   required fields: config path, dataset, resolution, iters, val mIoU, val F1, git commit hash.

6. **If this is the last model in the phase**, write a phase-level summary report at
   `challenges/embedded-semantic-segmentation/{phase}/{phase}_report.md` covering:
   - All models trained, with final val mIoU and F1
   - Per-model training metadata — verify each value against the training log before
     writing (do not rely solely on config comments). If any value in the log
     disagrees with the config comment, flag it explicitly in the report:
     - `samples_per_gpu` (batch size)
     - Effective LR used (check log for actual LR at iter 1 post-warmup)
     - Total iters run
     - Eval interval and checkpoint interval
     - Inherited schedule (e.g. schedule_80k, schedule_160k)
     - mmseg version and conda env
   - Methodology decisions and rationale (batch size strategy, LR scaling rules,
     preprocessing fixes applied, config philosophy)
   - Architecture rankings and key observations
   - Issues, gotchas, and workarounds — pull from `{phase}/notes.md` and include
     in full so the knowledge is not lost
   - What carries forward to the next phase
   This report is the permanent record of the phase — it replaces issue references
   as the authoritative source of methodology and rationale for the team.
   After writing the report, delete `{phase}/notes.md` and commit the deletion
   alongside the report.

7. **Commit** the archive directory with message:
   `[agent] archive {phase} {model_name}: mIoU=X.X, F1=X.X, ONNX delta=X.X%`

---

## Updating this skill

If the archival process needs to change, first ask:
> "Is this a one-time change for this specific run, or a permanent change to
> the archival process?"

- **One-time**: proceed with the variation without modifying this file.
- **Permanent**: update this skill file to reflect the new process, then proceed.
