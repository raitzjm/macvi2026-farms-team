# Phase 1 — Training Notes and Gotchas

Collected during Phase 1 baseline runs (EXP-001 through EXP-004).
This file is consumed when writing `phase1_report.md` and then deleted.

---

### [Infrastructure] torchrun vs torch.distributed.launch — --local-rank argument mismatch

PyTorch 2.1.x `torch.distributed.launch` passes `--local-rank` (hyphen) as a CLI arg,
but mmseg 0.30.0 expects `--local_rank` (underscore). Using `dist_train.sh` (which calls
`torch.distributed.launch`) causes all workers to crash immediately with an argparse error.

Fix: use `torchrun --standalone --nproc_per_node=N tools/train.py --launcher pytorch`.
`torchrun` sets `LOCAL_RANK` as an environment variable — mmseg reads this correctly.
All Phase 1 run scripts use `torchrun`.

---

### [Infrastructure] MMDistributedDataParallel._use_replicated_tensor_module AttributeError during eval

Training runs fine (uses `train_step()` which bypasses `DDP.forward()`), but the first
eval at iter 250 crashes with `AttributeError: '_use_replicated_tensor_module'`.

Root cause: mmcv's `_run_ddp_forward()` references `self._use_replicated_tensor_module`
directly, but PyTorch 2.1.2 does not define this attribute on `DistributedDataParallel`.
The attribute was introduced in a later PyTorch version.

Fix: in `mmcv/parallel/distributed.py` (~line 160):
```python
# Before:
module_to_run = self._replicated_tensor_module if \
    self._use_replicated_tensor_module else self.module
# After:
module_to_run = self._replicated_tensor_module if \
    getattr(self, '_use_replicated_tensor_module', False) else self.module
```
Must reapply after any mmcv reinstall. Needs to be baked into the Dockerfile.

---

### [Infrastructure] WANDB_DIR must not be on PYTHONPATH

If `WANDB_DIR` is not set, wandb defaults to writing `wandb/` in the current directory.
If the current directory (repo root) is on `PYTHONPATH`, Python imports the `wandb/`
cache directory instead of the `wandb` package — all `import wandb` calls fail silently
or raise obscure errors.

Fix: always set `export WANDB_DIR=<path-not-on-PYTHONPATH>` before training.
Phase 1 scripts set `WANDB_DIR=$OUTPUTS/phase1`. The `wandb/` dir is gitignored.

---

### [Infrastructure] wandb==0.25.0 validation.py patch for new key format

New wandb API keys start with `wandb_` (e.g. `wandb_abc...`). The validation.py code
in the wandb SDK checks for exactly 40 characters, rejecting `wandb_` prefix keys.

Fix: in `wandb/sdk/lib/wbauth/validation.py`, add an early-return before the 40-char check:
```python
if api_key.startswith("wandb_"):
    return True
```
Must reapply after any wandb reinstall.

---

### [STDC2] OOM at calibrated batch size — full training step needs more memory than fwd+bwd

Batch size was calibrated by running fwd+bwd only. During actual training (loss computation,
optimizer step, metric buffers), memory usage is higher. STDC2 OOMed at the calibrated
bs=174; reduced to bs=160.

Lesson: always reduce calibrated batch size by ~10% before the first real run, or calibrate
with a full training step (train_step_only mode not available in mmseg 0.30.0).

---

### [BiSeNetV2] OOM at bs=152 — 5 auxiliary heads are memory-intensive

BiSeNetV2 has 5 auxiliary segmentation heads in addition to the main head. Memory usage
is significantly higher than STDC2 despite similar backbone complexity. OOMed at bs=152;
reduced to bs=128, with LR linearly scaled from 3.8→3.2 (bs 192→256 reference adjusted).

---

### [BiSeNetV2] Near-catastrophic obstacle collapse at iter 750 (IoU=0.12%)

With SGD LR=3.2 (aggressively linear-scaled for bs=256), BiSeNetV2 showed extreme training
instability: obstacle IoU collapsed from ~52% (iter 1000) to 0.12% at iter 750, then
partially recovered to 52% at iter 1000, then fully recovered to 86% at iter 1500.

The model was still climbing at iter 1500 (best checkpoint) — Phase 1's 1500 iter budget
was too short to confirm convergence. For Phase 2: investigate longer warmup (currently
150 iters) or a lower LR. The linear scaling rule may be too aggressive for BiSeNetV2's
5-head architecture.

---

### [SegNeXt-T] OOM at bs=75 — HAM decoder is memory-intensive

SegNeXt-T uses a Hamburger (HAM) decoder which consumes significantly more activation
memory than standard FPN-style decoders. Calibrated at bs=75 but OOMed during training;
reduced to bs=56. LR unchanged (AdamW — no linear scaling).

---

### [SegFormer] Pre-emptive batch size reduction to bs=36

After the OOM pattern seen in the first three models, SegFormer was pre-emptively reduced
from the calibrated bs=45 to bs=36 (~20% reduction). No OOM occurred. The pre-emptive
reduction saved a failed run cycle.

---

### [All Models] Full training step memory exceeds fwd+bwd calibration

Consistent pattern across all 4 models: calibrated batch size (fwd+bwd only) leads to OOM
during actual training. Recommend: use 75–80% of calibrated batch size as starting point
for Phase 2 configs, or calibrate with full training step if mmseg version supports it.

---

### [All Models] val F1 not reported — eval script only computes mIoU and aAcc

The mmseg 0.30.0 `LaRSDataset` evaluator reports mIoU and aAcc but not the challenge
metric (F1 / BF score). Phase 1 results are logged in mIoU only. For Phase 2, add an
F1 evaluation callback or run the standalone eval scripts post-training.

---
