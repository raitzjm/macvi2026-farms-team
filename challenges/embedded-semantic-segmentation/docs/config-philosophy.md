# Config Philosophy

Design principles for mmseg training configs in this challenge.

## What is Architecture-Specific vs Shared

- **Architecture-specific** (optimizer, LR, losses, heads): left untouched per model
- **Shared** (resolution, pipeline, schedule, dataset paths): standardised across models

## Key Rules

- **LR scaling**: SGD uses the linear scaling rule; AdamW keeps LR fixed. See issue #9.
- **SyncBN**: required for BatchNorm models (STDC2, BiSeNetV2) when training with DDP.
- **Runner type**: `IterBasedRunner` (mmseg 0.30.0) / iter-based (mmseg 1.2.2).
- **Training resolution**: always 768×384 — this is the challenge server target resolution.

## Config API Migration (mmseg 0.30.0 → 1.x)

For differences between mmseg 0.x and 1.x config API, see
`phase1/phase1_report.md` section 13.

## Custom Imports

Phase 2+ configs include `custom_imports` to register the LaRSDataset and
ReflectPad transform. PYTHONPATH must include the repo root (handled by the
run scripts).
