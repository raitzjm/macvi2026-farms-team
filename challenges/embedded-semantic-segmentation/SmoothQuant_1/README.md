# SmoothQuant_1 — One-off Quantization Experiment

This directory tracks the first SmoothQuant (Post-Training Quantization) experiment batch.
These are isolated quantization experiments and use the **QEXP** namespace to avoid
consuming slots in the main EXP sequence.

## QEXP → EXP Mapping

Each QEXP entry applies a quantization technique to the checkpoint produced by a prior
training experiment (EXP).

| QEXP ID   | Quantized EXP | Model                  | Config                                                              | Result  |
|-----------|---------------|------------------------|---------------------------------------------------------------------|---------|
| QEXP-001  | EXP-006       | SegNeXt-T (HAMHead)    | phase2/configs/segnext_t_768x384_lars_phase2.py                    | FAILED (catastrophic — INT8 17.84% vs FP32 87.36%) |
| QEXP-002  | EXP-015       | SegNeXt-T (UPerHead)   | phase2/configs/segnext_t_uperhead_768x384_lars_phase2.py           | FAILED (catastrophic — INT8 9.11% vs FP32 85.28%) |
| QEXP-003  | EXP-008       | PIDNet-S (Phase 2)     | phase2/configs/pidnets_768x384_lars_phase2.py                      | FAILED (regression — INT8 8.84% vs baseline 93.43%) |
| QEXP-004  | EXP-018       | PIDNet-S (Phase 2.5)   | phase2_5/configs/pidnets_768x384_lars_phase2_5.py                  | FAILED (regression — INT8 8.84% vs baseline 93.43%) |

## Method

Manual SmoothQuant transformation (alpha=0.5). Applied to safe layer transitions
(backbone Mlp fc1, stage transitions, stem). Multiple alpha values tested (0.01–0.9);
0.5 was optimal but insufficient to recover SegNeXt outlier sensitivity.

**Conclusion:** SmoothQuant alone cannot rescue MSCAN outlier activations or stable
PIDNet-S models. Proceed to QAT (see QAT_1/).

## Scripts

- `../phase2_5/scripts/smoothquant_recovery.py` — manual SmoothQuant implementation
  with explicit layer mapping for MSCAN and PIDNet architectures.

## Full Results

See experiment entries QEXP-001 through QEXP-004 in `../experiment-log.md`.
