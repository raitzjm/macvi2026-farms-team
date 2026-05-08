T-019 final retrain kernel for the T-014 YOLO26s k-fold pseudo-label family.

This kernel is training-only. It mounts:
- the Kaggle competition/images datasets
- the small uploaded metadata bundle with:
  - `t014-pseudo-labels.json`
  - `t014-expanded-train.json`
  - `t014-expanded-manifest.json`

It rebuilds YOLO labels at runtime and trains with the locked final T-014 policy.
