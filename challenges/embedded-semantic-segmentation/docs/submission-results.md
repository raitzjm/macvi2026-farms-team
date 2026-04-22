# Submission Results Layout

How server result files are stored and how summary.json fields map to the
submissions tracker for the MaCVi embedded semantic segmentation challenge.

## File Location

After each submission, download result files from the challenge platform.
They land at:

```
~/data/submission_results/<submission_id>_results/
```

This path is a read-only mount inside the container (passed via `--data-dir`
to run.sh). Files: `frames.csv`, `segments.json`, `summary.json`.

When tracking a new submission, copy result files into:

```
challenges/embedded-semantic-segmentation/submissions/<id>_<model>_<exp>/results/
```

## summary.json → submissions.md Column Mapping

| summary.json field | submissions.md column |
|---|---|
| Q | Q score |
| WE_acc | μ (waterline estimation accuracy) |
| Pr | detection Precision |
| Re | detection Recall |
| F1 | detection F1 |
| mIoU | segmentation mIoU |
| FPS | **NOT in file** — leave as TBD unless provided separately |
