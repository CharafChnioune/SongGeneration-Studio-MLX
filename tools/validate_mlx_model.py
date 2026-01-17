#!/usr/bin/env python3
"""Validate MLX model layout and runtime assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config():
    root = _resolve_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import BASE_DIR, MLX_WEIGHT_PREFERENCE

    return BASE_DIR, MLX_WEIGHT_PREFERENCE


def _check_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MLX model + runtime assets.")
    parser.add_argument("--model-dir", required=True, help="Path to the MLX model folder.")
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="Root folder that contains ckpt/ and third_party/ (defaults to repo root).",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Skip runtime asset checks (ckpt + third_party).",
    )
    args = parser.parse_args()

    base_dir, weight_pref = _load_config()
    model_dir = Path(args.model_dir).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else base_dir

    missing = []
    if not model_dir.exists():
        missing.append(model_dir / "<model-dir>")
    else:
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            missing.append(config_path)
        weight_path = None
        for name in weight_pref:
            candidate = model_dir / name
            if candidate.exists():
                weight_path = candidate
                break
        if weight_path is None:
            missing.append(model_dir / weight_pref[0])

    runtime_missing = []
    if not args.skip_runtime:
        runtime_required = [
            runtime_dir / "ckpt" / "vae" / "stable_audio_1920_vae.json",
            runtime_dir / "ckpt" / "vae" / "autoencoder_music_1320k.npz",
            runtime_dir / "third_party" / "demucs" / "ckpt" / "htdemucs.onnx",
        ]
        runtime_missing = _check_paths(runtime_required)

    if missing or runtime_missing:
        if missing:
            print("Missing model files:")
            for path in missing:
                print(f"  - {path}")
        if runtime_missing:
            print("Missing runtime assets:")
            for path in runtime_missing:
                print(f"  - {path}")
        return 1

    print("OK: MLX model layout + runtime assets look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
