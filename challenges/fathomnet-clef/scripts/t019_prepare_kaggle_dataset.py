#!/usr/bin/env python3
"""Build the small Kaggle dataset bundle for the T-019 final retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_tracking import git_commit, write_json


DATASET_ID = "raitzjm/fathomnet-t019-final-retrain-inputs"
LICENSE_NAME = "CC0-1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_dataset_metadata() -> dict:
    return {
        "title": "FathomNet T-019 Final Retrain Inputs",
        "id": DATASET_ID,
        "licenses": [{"name": LICENSE_NAME}],
    }


def bundle_readme() -> str:
    return (
        "# T-019 Final Retrain Inputs\n\n"
        "Small metadata bundle for the Kaggle final retrain of the T-014 YOLO26s k-fold pseudo-label family.\n\n"
        "Contents:\n"
        "- `t014-pseudo-labels.json`\n"
        "- `t014-expanded-train.json`\n"
        "- `t014-expanded-manifest.json`\n\n"
        "Images are not duplicated here. The Kaggle kernel reads competition images directly and rebuilds YOLO labels at runtime.\n"
    )


def build_bundle(
    bundle_dir: Path,
    pseudo_label_json: Path,
    expanded_coco_json: Path,
    expanded_manifest_json: Path,
) -> dict:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "pseudo_label_json": {
            "src": Path(pseudo_label_json),
            "dst": "t014-pseudo-labels.json",
        },
        "expanded_coco_json": {
            "src": Path(expanded_coco_json),
            "dst": "t014-expanded-train.json",
        },
        "expanded_manifest_json": {
            "src": Path(expanded_manifest_json),
            "dst": "t014-expanded-manifest.json",
        },
    }
    allowed = {payload["dst"] for payload in files.values()} | {"dataset-metadata.json", "README.md", "bundle-summary.json"}
    extras = sorted(path.name for path in bundle_dir.iterdir() if path.name not in allowed)
    if extras:
        raise RuntimeError(f"Bundle dir contains unexpected files: {extras}")

    summary = {
        "dataset_id": DATASET_ID,
        "file_count": len(files),
        "files": {},
    }
    for key, payload in files.items():
        src = payload["src"]
        dst = bundle_dir / payload["dst"]
        shutil.copy2(src, dst)
        summary["files"][key] = {
            "src": str(src),
            "dst": payload["dst"],
            "size_bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }

    write_json(bundle_dir / "dataset-metadata.json", kaggle_dataset_metadata())
    (bundle_dir / "README.md").write_text(bundle_readme())
    write_json(bundle_dir / "bundle-summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    challenge_root = Path(__file__).resolve().parent.parent
    default_bundle_dir = challenge_root / "kaggle" / "t019-final-retrain-dataset"
    default_pseudo = challenge_root / "artifacts" / "pseudo-labels" / "t014-yolo26s-kfold-plateau354045-conf025-iou070.json"
    default_expanded = challenge_root / "outputs" / "t014-yolo26s-final-T-019" / "expanded-train.json"
    default_manifest = challenge_root / "outputs" / "t014-yolo26s-final-T-019" / "t014-expanded-manifest.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=str(default_bundle_dir))
    parser.add_argument("--pseudo-label-json", default=str(default_pseudo))
    parser.add_argument("--expanded-coco-json", default=str(default_expanded))
    parser.add_argument("--expanded-manifest-json", default=str(default_manifest))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    challenge_root = Path(__file__).resolve().parent.parent
    bundle_dir = Path(args.bundle_dir)
    summary = build_bundle(
        bundle_dir=bundle_dir,
        pseudo_label_json=Path(args.pseudo_label_json),
        expanded_coco_json=Path(args.expanded_coco_json),
        expanded_manifest_json=Path(args.expanded_manifest_json),
    )
    summary["git_commit"] = git_commit(challenge_root)
    write_json(bundle_dir / "bundle-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
