"""
SongGeneration Studio - MLX Model Registry & Download Manager.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, List

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
        "hf_repo": None,
        "size_gb": 11.3,
        "priority": 1,
    },
    "songgeneration_base_new": {
        "name": "SongGeneration - Base New (2m30s)",
        "description": "Updated base model, MLX weights",
        "vram_required": 10,
        "hf_repo": None,
        "size_gb": 11.3,
        "priority": 2,
    },
    "songgeneration_base_full": {
        "name": "SongGeneration - Base Full (4m30s)",
        "description": "Full duration up to 4m30s, MLX weights",
        "vram_required": 12,
        "hf_repo": None,
        "size_gb": 11.3,
        "priority": 3,
    },
    "songgeneration_large": {
        "name": "SongGeneration - Large (4m30s)",
        "description": "Best quality, MLX weights",
        "vram_required": 22,
        "hf_repo": None,
        "size_gb": 20.5,
        "priority": 4,
    },
}

# Download state tracking (lightweight)
download_states: Dict[str, dict] = {}
_download_threads: Dict[str, threading.Thread] = {}
_download_cancel_flags: Dict[str, threading.Event] = {}
_download_lock = threading.Lock()


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
    from huggingface_hub import snapshot_download

    repo_id = _resolve_hf_repo(model_id)
    if not repo_id:
        download_states[model_id] = {
            "status": "error",
            "error": "No Hugging Face repo configured for downloads.",
        }
        return

    download_states[model_id] = {"status": "downloading", "progress": 0}
    allow_patterns = None
    local_dir = None

    if HF_MODEL_LAYOUT == "single":
        local_dir = str(BASE_DIR)
        allow_patterns = [f"{model_id}/*"]
    else:
        local_dir = str(_model_dir(model_id))

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        download_states[model_id] = {"status": "completed", "progress": 100}
    except Exception as exc:
        download_states[model_id] = {"status": "error", "error": str(exc)}


def start_model_download(model_id: str, notify_cb=None) -> dict:
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}

    with _download_lock:
        status = download_states.get(model_id, {})
        if status.get("status") == "downloading":
            return status

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
