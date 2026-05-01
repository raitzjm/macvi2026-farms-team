#!/usr/bin/env python3
"""Helpers for durable training/inference run artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


class _TeeWriter:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def resolve_artifact_dir(challenge_root: Path, kind: str, run_id: str | None, artifact_dir: str | None) -> Path | None:
    if artifact_dir:
        return Path(artifact_dir)
    if run_id:
        return challenge_root / "artifacts" / f"{kind}-runs" / run_id
    return None


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def relpath_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def tracked_run_name(base_name: str, run_id: str | None) -> str:
    name = str(base_name).strip()
    if not run_id:
        return name
    suffix = str(run_id).strip().upper()
    if not suffix:
        return name
    if name.upper().endswith(f"-{suffix}") or name.upper() == suffix:
        return name
    return f"{name}-{suffix}"


def tracked_submission_path(output_csv: Path, run_id: str | None, source_train_id: str | None) -> Path:
    parts: list[str] = []
    if run_id:
        parts.append(str(run_id).strip().upper())
    if source_train_id:
        source = str(source_train_id).strip().upper()
        if source:
            parts.append(f"from-{source}")

    if not parts:
        return output_csv

    suffix = "-".join(parts)
    stem_upper = output_csv.stem.upper()
    if suffix.upper() in stem_upper:
        return output_csv
    return output_csv.with_name(f"{output_csv.stem}-{suffix}{output_csv.suffix}")


@contextmanager
def tee_output(log_path: Path | None):
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(orig_stdout, log_file)
        sys.stderr = _TeeWriter(orig_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
