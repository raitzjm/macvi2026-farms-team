# Training Run Bundles

Each tracked training run gets `artifacts/training-runs/T-NNN/` with:

- `metadata.json`
- raw `stdout.log` when available
- resolved config snapshot (`resolved-config.*`)
- native metrics artifact (`results.csv`, `metrics.jsonl`, or equivalent)
- optional framework-native side logs (`framework.log`, `kaggle-console.json`)

Selected checkpoints stay under `artifacts/checkpoints/` and are referenced from `metadata.json`.
