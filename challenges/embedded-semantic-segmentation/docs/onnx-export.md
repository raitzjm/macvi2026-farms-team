# ONNX Export Requirements

Requirements for exporting models to ONNX for challenge submission.

## Output Spec

- Output: argmaxed class map, shape `1×1×384×768`, dtype `int64`
- Opset version: **13** (challenge server uses SNPE which requires opset 13)

## Reference Implementations

See `fun_experiments/{model}/export_onnx.py` for per-model reference implementations.

For Phase 2+, use `phase2/scripts/export_and_eval.py` (mmseg 1.x API).
The same script handles PyTorch/ONNX precision validation.

## Downstream Pipeline

The challenge platform handles INT8 quantization (Qualcomm SNPE) and deployment
(Luxonis RVC4) — we do NOT do this. Our job ends at submitting the ONNX file.
