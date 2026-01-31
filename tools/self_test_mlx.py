"""Self-test helper for MLX SongGeneration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_MODEL, MLX_WEIGHT_PREFERENCE  # noqa: E402


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
        import importlib.metadata as metadata

        try:
            version = metadata.version("mlx")
        except Exception:
            version = getattr(mx, "__version__", "unknown")
        _log(f"[MLX] version={version}")
        try:
            _log(f"[MLX] default_device={mx.default_device()}")
            _log(f"[MLX] device_info={mx.device_info()}")
        except Exception as exc:
            _log(f"[MLX] device query failed: {exc}")
        return True
    except Exception as exc:
        _log(f"[MLX] import failed: {exc}")
        return False


def _find_local_models(base_dir: Path, preferred: str) -> tuple[str, Path] | tuple[None, None]:
    candidates: list[tuple[str, Path]] = []
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if not (entry / "config.yaml").exists():
            continue
        for weight_name in MLX_WEIGHT_PREFERENCE:
            weight_path = entry / weight_name
            if weight_path.exists():
                candidates.append((entry.name, weight_path))
                break
    if not candidates:
        return None, None
    if preferred:
        for model_id, weight_path in candidates:
            if model_id == preferred:
                return model_id, weight_path
    return candidates[0]


def _check_runtime_assets(base_dir: Path) -> list[Path]:
    required = [
        base_dir / "ckpt" / "vae" / "stable_audio_1920_vae.json",
        base_dir / "ckpt" / "vae" / "autoencoder_music_1320k.npz",
    ]
    missing = [path for path in required if not path.exists()]
    return missing


def _make_test_prompt() -> dict:
    return {
        "idx": "self_test",
        "gt_lyric": "[intro-short] ; [verse] Night drive. neon sky. slow heart. steady sigh. Wheels hum. city glow. we go. we go. ; [chorus] Hold on. stay close. feel the light. In the dark we rise. move in time. ; [outro-short]",
        "descriptions": "female, warm, pop, uplifting, piano and synthesizer",
    }


def _run_generate(cmd: list[str], dry_run: bool) -> int:
    _log("[SELF-TEST] Running: " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="MLX self-test for SongGeneration")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to test")
    parser.add_argument("--duration", type=float, default=12.0, help="Duration in seconds")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--tokens_only", action="store_true", help="Only generate tokens")
    parser.add_argument("--debug", action="store_true", help="Enable MLX debug logs")
    parser.add_argument("--fetch_runtime", action="store_true", help="Download runtime assets if missing")
    parser.add_argument("--dry_run", action="store_true", help="Show what would run without executing")
    args = parser.parse_args()

    if not _check_mlx():
        _log("[SELF-TEST] MLX is not available. Install MLX and retry.")
        return 1

    model_id, weight_path = _find_local_models(BASE_DIR, args.model)
    if model_id is None or weight_path is None:
        _log("[SELF-TEST] No local MLX models found. Download a model in the UI first.")
        return 2

    missing_runtime = _check_runtime_assets(BASE_DIR)
    if missing_runtime and not args.tokens_only:
        if args.fetch_runtime:
            _log("[SELF-TEST] Fetching runtime assets...")
            cmd = [sys.executable, str(BASE_DIR / "tools" / "fetch_runtime.py"), "--local-dir", str(BASE_DIR)]
            if _run_generate(cmd, args.dry_run) != 0:
                _log("[SELF-TEST] Runtime asset download failed.")
                return 3
        else:
            missing_list = ", ".join(str(p) for p in missing_runtime)
            _log(f"[SELF-TEST] Missing runtime assets: {missing_list}")
            _log("[SELF-TEST] Run: python tools/fetch_runtime.py --local-dir .")
            return 3

    output_dir = Path(args.output_dir) if args.output_dir else (BASE_DIR / "output" / f"self_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "self_test.jsonl"
        payload = _make_test_prompt()
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        cmd = [
            sys.executable,
            str(BASE_DIR / "generate_mlx.py"),
            "--ckpt_path",
            str(BASE_DIR / model_id),
            "--weights",
            str(weight_path),
            "--input_jsonl",
            str(input_path),
            "--save_dir",
            str(output_dir),
            "--duration",
            str(args.duration),
        ]
        if args.tokens_only:
            cmd.append("--tokens_only")
        if args.debug or _truthy_env("SONGGEN_MLX_DEBUG"):
            cmd.append("--debug_mlx")

        code = _run_generate(cmd, args.dry_run)
        if code != 0:
            _log(f"[SELF-TEST] Generation failed with code {code}")
            return code

    _log(f"[SELF-TEST] Done. Outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
