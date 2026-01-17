"""Download SongGeneration runtime assets (ckpt + third_party) from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REQUIRED_FILES = (
    "ckpt/vae/autoencoder_music_1320k.npz",
    "ckpt/vae/stable_audio_1920_vae.json",
)


def _verify_assets(root: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def _cleanup_legacy_ckpt(root: Path) -> int:
    removed = 0
    for path in root.glob("ckpt/**/*.ckpt"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


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
    root = Path(args.local_dir)
    missing = _verify_assets(root)
    if missing:
        print("Missing MLX runtime assets:", ", ".join(missing))
        print("Update the runtime repo or re-run download.")
        return 1
    removed = _cleanup_legacy_ckpt(root)
    if removed:
        print(f"Removed {removed} legacy .ckpt file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
