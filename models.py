"""
SongGeneration Studio - MLX Model Registry & Download Manager.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List

import requests
from huggingface_hub import HfApi, HfFolder, hf_hub_url

from config import (
    BASE_DIR,
    HF_MODEL_REPO,
    HF_MODEL_LAYOUT,
    HF_MODEL_PREFIX,
    MLX_WEIGHT_PREFERENCE,
)
from gpu import gpu_info, refresh_gpu_info

MODEL_REGISTRY: Dict[str, dict] = {
    "songgeneration_base": {
        "name": "SongGeneration - Base (2m30s)",
        "description": "Chinese + English, MLX weights, max 2m30s",
        "vram_required": 10,
        "hf_repo": "AITRADER/SongGeneration-Base-MLX",
        "size_gb": 11.3,
        "priority": 1,
    },
    "songgeneration_base_new": {
        "name": "SongGeneration - Base New (2m30s)",
        "description": "Updated base model, MLX weights",
        "vram_required": 10,
        "hf_repo": "AITRADER/SongGeneration-Base-New-MLX",
        "size_gb": 11.3,
        "priority": 2,
    },
    "songgeneration_base_full": {
        "name": "SongGeneration - Base Full (4m30s)",
        "description": "Full duration up to 4m30s, MLX weights",
        "vram_required": 12,
        "hf_repo": "AITRADER/SongGeneration-Base-Full-MLX",
        "size_gb": 11.3,
        "priority": 3,
    },
    "songgeneration_large": {
        "name": "SongGeneration - Large (4m30s)",
        "description": "Best quality, MLX weights",
        "vram_required": 22,
        "hf_repo": "AITRADER/SongGeneration-Large-MLX",
        "size_gb": 20.5,
        "priority": 4,
    },
}

# Download state tracking (lightweight)
download_states: Dict[str, dict] = {}
_download_threads: Dict[str, threading.Thread] = {}
_download_cancel_flags: Dict[str, threading.Event] = {}
_download_lock = threading.Lock()
_runtime_lock = threading.Lock()

RUNTIME_REQUIRED = (
    BASE_DIR / "ckpt" / "vae" / "stable_audio_1920_vae.json",
    BASE_DIR / "ckpt" / "vae" / "autoencoder_music_1320k.npz",
)


def _ensure_runtime_assets() -> None:
    missing = [path for path in RUNTIME_REQUIRED if not path.exists()]
    if not missing:
        return
    script = BASE_DIR / "tools" / "fetch_runtime.py"
    if not script.exists():
        missing_list = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Missing runtime assets: {missing_list}")
    print("[RUNTIME] Downloading MLX runtime assets (requested by model download)...", flush=True)
    subprocess.check_call([sys.executable, str(script), "--local-dir", str(BASE_DIR)])


def _model_dir(model_id: str) -> Path:
    return BASE_DIR / model_id


def _find_weights(model_dir: Path) -> Optional[Path]:
    for name in MLX_WEIGHT_PREFERENCE:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def _resolve_hf_repo(model_id: str) -> Optional[str]:
    info = MODEL_REGISTRY.get(model_id)
    if info and info.get("hf_repo"):
        return info["hf_repo"]
    if HF_MODEL_LAYOUT == "single":
        return HF_MODEL_REPO or None
    if HF_MODEL_PREFIX:
        return f"{HF_MODEL_PREFIX}/{model_id}"
    return HF_MODEL_REPO or None


def get_model_status(model_id: str) -> str:
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"

    model_dir = _model_dir(model_id)
    if not model_dir.exists():
        return "not_downloaded"

    weights = _find_weights(model_dir)
    if weights and (model_dir / "config.yaml").exists():
        return "ready"

    return "not_downloaded"


def get_model_status_quick(model_id: str) -> str:
    return get_model_status(model_id)


def is_model_ready_quick(model_id: str) -> bool:
    return get_model_status_quick(model_id) == "ready"


def get_download_progress(model_id: str) -> dict:
    return download_states.get(model_id, {"progress": 0})


def get_available_models_sync() -> List[dict]:
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        if get_model_status_quick(model_id) == "ready":
            models.append({"id": model_id, **info})
    return models


def _available_memory_gb() -> Optional[float]:
    g = gpu_info.get("gpu") if isinstance(gpu_info, dict) else None
    if g is None:
        return None
    return g.get("free_gb") or g.get("total_gb")


def get_recommended_model(refresh: bool = False) -> str:
    if refresh:
        refresh_gpu_info()
    available_gb = _available_memory_gb()
    candidates = sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["priority"], reverse=True)
    if available_gb is None:
        return candidates[0][0] if candidates else "songgeneration_base"
    for model_id, info in candidates:
        if info.get("vram_required", 0) <= available_gb:
            return model_id
    return candidates[-1][0] if candidates else "songgeneration_base"


def get_best_ready_model(refresh: bool = False) -> Optional[str]:
    if refresh:
        refresh_gpu_info()
    available_gb = _available_memory_gb()
    ready = [mid for mid in MODEL_REGISTRY if get_model_status_quick(mid) == "ready"]
    if not ready:
        return None
    ready_sorted = sorted(ready, key=lambda mid: MODEL_REGISTRY[mid]["priority"], reverse=True)
    if available_gb is None:
        return ready_sorted[0]
    for mid in ready_sorted:
        if MODEL_REGISTRY[mid].get("vram_required", 0) <= available_gb:
            return mid
    return ready_sorted[-1]


def _download_worker(model_id: str) -> None:
    repo_id = _resolve_hf_repo(model_id)
    if not repo_id:
        download_states[model_id] = {
            "status": "error",
            "error": "No Hugging Face repo configured for downloads.",
        }
        return

    try:
        with _runtime_lock:
            _ensure_runtime_assets()
        api = HfApi()
        info = api.model_info(repo_id=repo_id)
        prefix = f"{model_id}/" if HF_MODEL_LAYOUT == "single" else ""
        wanted = set(MLX_WEIGHT_PREFERENCE + ("config.yaml", "README.md"))
        files = []
        total_bytes = 0

        for sibling in info.siblings:
            name = sibling.rfilename
            if prefix and not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if rel not in wanted:
                continue
            size = sibling.size or 0
            files.append({"remote": name, "rel": rel, "size": size})
            total_bytes += size

        if not files:
            raise RuntimeError(f"No model files found in {repo_id}")

        model_dir = _model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        downloaded_bytes = 0
        for item in files:
            dest = model_dir / item["rel"]
            if dest.exists() and item["size"] and dest.stat().st_size == item["size"]:
                downloaded_bytes += item["size"]

        last_time = time.time()
        last_bytes = downloaded_bytes

        def update_progress(current_bytes: int) -> None:
            nonlocal last_time, last_bytes
            now = time.time()
            if now - last_time < 1.0:
                return
            speed = (current_bytes - last_bytes) / max(1e-6, now - last_time)
            last_time = now
            last_bytes = current_bytes
            if total_bytes > 0:
                progress = int(min(100, (current_bytes / total_bytes) * 100))
                eta = int(max(0, (total_bytes - current_bytes) / speed)) if speed > 0 else 0
            else:
                progress = 0
                eta = 0
            download_states[model_id] = {
                "status": "downloading",
                "progress": progress,
                "downloaded_gb": round(current_bytes / (1024 ** 3), 2),
                "speed_mbps": round(speed / (1024 ** 2), 2),
                "eta_seconds": eta,
            }

        token = HfFolder.get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        for item in files:
            dest = model_dir / item["rel"]
            expected = item["size"]
            if dest.exists() and expected and dest.stat().st_size == expected:
                update_progress(downloaded_bytes)
                continue

            tmp_path = dest.with_suffix(dest.suffix + ".part")
            existing = tmp_path.stat().st_size if tmp_path.exists() else 0
            if dest.exists() and expected and dest.stat().st_size != expected:
                dest.unlink()

            url = hf_hub_url(repo_id=repo_id, filename=item["remote"], repo_type="model")
            req_headers = dict(headers)
            if existing > 0:
                req_headers["Range"] = f"bytes={existing}-"

            with requests.get(url, headers=req_headers, stream=True, timeout=30) as resp:
                if resp.status_code == 416:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    existing = 0
                    req_headers.pop("Range", None)
                    resp = requests.get(url, headers=req_headers, stream=True, timeout=30)
                resp.raise_for_status()
                mode = "ab" if existing > 0 else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        existing += len(chunk)
                        current_total = downloaded_bytes + existing
                        update_progress(current_total)

            tmp_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            if expected and tmp_size != expected:
                raise RuntimeError(f"Incomplete download for {item['rel']}: {tmp_size} / {expected}")
            tmp_path.replace(dest)
            downloaded_bytes += tmp_size
            update_progress(downloaded_bytes)

        download_states[model_id] = {
            "status": "completed",
            "progress": 100,
            "downloaded_gb": round(total_bytes / (1024 ** 3), 2),
            "speed_mbps": 0,
            "eta_seconds": 0,
        }
    except Exception as exc:
        download_states[model_id] = {"status": "error", "error": str(exc)}


def start_model_download(model_id: str, notify_cb=None) -> dict:
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}

    with _download_lock:
        status = download_states.get(model_id, {})
        if status.get("status") == "downloading":
            return status

        download_states[model_id] = {
            "status": "downloading",
            "progress": 0,
            "downloaded_gb": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
        }
        thread = threading.Thread(target=_download_worker, args=(model_id,), daemon=True)
        _download_threads[model_id] = thread
        thread.start()

    if notify_cb:
        notify_cb()

    return {"status": "downloading"}


def cancel_model_download(model_id: str) -> dict:
    status = download_states.get(model_id, {})
    if status.get("status") != "downloading":
        return {"error": "No active download"}
    return {"error": "Download cancellation is not supported."}


def delete_model(model_id: str) -> dict:
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}
    model_dir = _model_dir(model_id)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    return {"status": "deleted"}


def cleanup_download_states() -> None:
    stale = [k for k, v in download_states.items() if v.get("status") in ("completed", "error")]
    for k in stale:
        download_states.pop(k, None)
