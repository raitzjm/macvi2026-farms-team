# Inference Run Bundles

Each tracked inference run gets `artifacts/inference-runs/I-NNN/` with:

- `metadata.json`
- resolved inference config snapshot (`resolved-config.json`)
- raw `stdout.log` optionally, when preserved for debugging or provenance

Submission CSV payloads stay under `artifacts/submissions/` and are referenced from `metadata.json`.
