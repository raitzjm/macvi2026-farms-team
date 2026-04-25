# Output Directory Convention and Artifact Archival

Where training outputs go and how curated artifacts are committed.

## Output Directory Convention

Training live outputs (checkpoints, logs) go to:
```
challenges/embedded-semantic-segmentation/outputs/{phase}/{model_name}/
```

Pass this as `--work-dir` to the train script. Examples:
```
outputs/phase1/stdc2/
outputs/phase2/stdc2/
```

This keeps phases isolated and prevents checkpoint collisions. The `outputs/`
tree is gitignored — only curated artifacts are committed (see Artifact Archival).

## Artifact Archival

After each training run completes, use the `/archive-experiment` skill.
It handles ONNX export, pth vs ONNX precision validation, artifact commit, and
experiment-log.md update. INT8 quantization evaluation is deferred to Phase 2.

## Phase 2 Committed Checkpoints

Best checkpoints for INT8-stable models are committed to Git LFS under
`phase2/artifacts/{model}/best_mIoU_iter_NNNN.pth`.

| Model | Checkpoint | Val mIoU | Val F1 | Q | INT8-stable? |
|-------|-----------|----------|--------|---|--------------|
| pidnets (EXP-008) | best_mIoU_iter_4750.pth | 93.25% | 76.17% | 71.03% | ✓ |
| pidnetm (EXP-016) | best_mIoU_iter_3250.pth | 92.33% | 72.82% | 67.23% | ✓ |
| stdc2 (EXP-007) | best_mIoU_iter_2750.pth | 90.69% | 68.92% | 62.50% | ✓ |
| bisenetv2 (EXP-009) | best_mIoU_iter_6000.pth | 91.09% | 69.06% | 62.91% | ✓ |
| segnext_t_uperhead (EXP-015) | best_mIoU_iter_4250.pth | 92.48% | 80.21% | 74.18% | ✗ (−78%) |
| segnext_s (EXP-010) | best_mIoU_iter_4750.pth | 92.40% | 72.33% | 66.83% | ✗ (−82%) |
| segnext_t (EXP-006) | best_mIoU_iter_4250.pth | 91.79% | 71.74% | 65.85% | ✗ (−67%) |

SegNeXt models (EXP-006/010/015) committed to LFS for reference only — INT8 catastrophic
failure (MSCAN large-kernel depthwise attention). See `eval_metrics.md` for full results.
