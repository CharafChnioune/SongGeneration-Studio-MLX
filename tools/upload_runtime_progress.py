#!/usr/bin/env python3
"""Upload MLX runtime assets to Hugging Face with a simple % progress bar."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi
from tqdm import tqdm

DEFAULT_PATTERNS = (
    "ckpt/vae/**",
    "ckpt/model_1rvq/**",
    "ckpt/model_septoken/**",
    "third_party/demucs/**",
    "third_party/Qwen2-7B/**",
    "third_party/hub/**",
)
SKIP_SUFFIXES = {".ckpt"}


def _iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            files[path] = None
    return sorted(files.keys())


def _bytes_total(paths: Iterable[Path]) -> int:
    return sum(p.stat().st_size for p in paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="AITRADER/SongGeneration-Runtime-MLX",
        help="Hugging Face repo ID to upload to.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Local SongGeneration repo root.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=list(DEFAULT_PATTERNS),
        help="Glob pattern to include (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = _iter_files(root, args.pattern)
    if not files:
        print("No files matched. Check --root and --pattern.")
        return 1

    total_bytes = _bytes_total(files)
    print(f"Found {len(files)} files, {total_bytes / (1024**3):.2f} GB total.")
    if args.dry_run:
        for path in files:
            print(path.relative_to(root))
        return 0

    api = HfApi()
    bar = tqdm(total=total_bytes, unit="B", unit_scale=True)
    for path in files:
        rel = path.relative_to(root)
        bar.set_description(f"Uploading {rel}")
        api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(path),
            path_in_repo=str(rel),
            commit_message=f"Add runtime asset: {rel}",
        )
        bar.update(path.stat().st_size)
    bar.close()
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
