"""Download SongGeneration runtime assets (ckpt + third_party) from Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="AITRADER/SongGeneration-Runtime-MLX",
        help="Hugging Face repo ID for runtime assets.",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="Target directory to place downloaded files.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=["ckpt/**", "third_party/**"],
        help="Allow patterns to download (repeatable).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional repo revision (branch, tag, or commit).",
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"Downloading {args.repo} -> {args.local_dir}")
    print(f"Allow patterns: {args.allow}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
