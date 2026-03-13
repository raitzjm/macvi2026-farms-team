# QAT_1 — One-off Quantization Experiment

This directory tracks the first Quantization-Aware Training (QAT) experiment batch.
These are isolated quantization experiments and use the **QEXP** namespace to avoid
consuming slots in the main EXP sequence.

## QEXP → EXP Mapping

Each QEXP entry applies QAT fine-tuning to the checkpoint produced by a prior
training experiment (EXP).

| QEXP ID   | Quantized EXP | Model                  | Config                                                                    | Result  |
|-----------|---------------|------------------------|---------------------------------------------------------------------------|---------|
| QEXP-005  | EXP-006       | SegNeXt-T (HAMHead)    | phase3/configs/segnext_t_768x384_lars_qat.py                             | SUCCESS (massive recovery — INT8 44.29% vs baseline ~0.86%) |
| QEXP-006  | EXP-015       | SegNeXt-T (UPerHead)   | phase3/configs/segnext_t_uperhead_768x384_lars_qat.py                    | SUCCESS (partial recovery — INT8 36.54%) |
| QEXP-007  | EXP-008       | PIDNet-S (Phase 2)     | phase3/configs/pidnets_768x384_lars_qat.py                               | SUCCESS (stable — INT8 69.28%) |
| QEXP-008  | EXP-018       | PIDNet-S (Phase 2.5)   | phase3/configs/pidnets_p2_5_768x384_lars_qat.py                          | SUCCESS (stable — INT8 67.43%) |

## Method

Quantization-Aware Training: 2,000 iterations, batch size 2, manual FakeQuant
wrapper. Fine-tuning on LaRS training set using Phase 2.5/3 pipeline logic.

**Conclusion:** QAT successfully recovered SegNeXt from INT8 failure (0.86% → 44.29%).
PIDNet-S remained stable. QAT is the recommended quantization technique for
MSCAN-based architectures in this project.

## Scripts

- `../phase3/scripts/prepare_qat_model.py` — wraps model in FakeQuant modules
- `../phase3/scripts/train_qat.py` — QAT fine-tuning loop
- `../phase3/scripts/run_qat.sh` — end-to-end QAT pipeline script
- `../phase3/scripts/unwrap_qat_ckpt.py` — removes FakeQuant wrappers for export

## Relationship to SmoothQuant_1

SmoothQuant (QEXP-001 through QEXP-004) was attempted first and failed for all models.
QAT was the follow-up and is the recommended path. See `../SmoothQuant_1/README.md`.

## Full Results

See experiment entries QEXP-005 through QEXP-008 in `../experiment-log.md`.
