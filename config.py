"""
SongGeneration Studio - Configuration
Directories, constants, and shared state initialization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# ============================================================================
# Directory Configuration
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR
DEFAULT_MODEL = "songgeneration_base"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "web" / "static"  # Static files in app/web/static/
QUEUE_FILE = BASE_DIR / "queue.json"
VERIFIED_MODELS_FILE = BASE_DIR / "verified_models.json"
TIMING_FILE = BASE_DIR / "timing_history.json"
AUTO_PROMPT_PATH = BASE_DIR / "tools" / "new_prompt.npz"
SEPARATOR_MODEL_PATH = BASE_DIR / "third_party" / "demucs" / "ckpt" / "htdemucs.onnx"
MLX_WEIGHT_PREFERENCE = ("model_fp16.npz", "model_int8.npz")

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Model Server Configuration (not used in MLX version)
MODEL_SERVER_PORT = 42100
MODEL_SERVER_URL = f"http://127.0.0.1:{MODEL_SERVER_PORT}"
USE_MODEL_SERVER = False

# Hugging Face download settings (optional)
# Layout: "multi" -> one repo per model (prefix + model_id),
#         "single" -> one repo with subfolders per model_id.
HF_MODEL_REPO = os.environ.get("SONGGEN_MLX_HF_REPO", "").strip()
HF_MODEL_LAYOUT = os.environ.get("SONGGEN_MLX_HF_LAYOUT", "multi").strip().lower()
HF_MODEL_PREFIX = os.environ.get("SONGGEN_MLX_HF_PREFIX", "").strip()

# Timing History
MAX_TIMING_RECORDS = 1000  # Keep last 1000 successful generations

# ============================================================================
# Verified Models Cache
# ============================================================================

verified_models_cache: Dict[str, dict] = {}

def load_verified_models() -> dict:
    """Load verified models cache from disk"""
    try:
        if VERIFIED_MODELS_FILE.exists():
            with open(VERIFIED_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_verified_models(cache: dict):
    """Save verified models cache to disk"""
    try:
        with open(VERIFIED_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def mark_model_verified(model_id: str, model_weights_size: int):
    """Mark a model as verified (size checked against HuggingFace)"""
    global verified_models_cache
    verified_models_cache[model_id] = {
        "verified": True,
        "model_weights_size": model_weights_size,
        "verified_at": datetime.now().isoformat()
    }
    save_verified_models(verified_models_cache)

def is_model_verified(model_id: str) -> bool:
    """Check if model has been verified"""
    return model_id in verified_models_cache and verified_models_cache[model_id].get("verified")

def get_verified_model_size(model_id: str) -> int:
    """Get verified model weights size if available"""
    if model_id in verified_models_cache:
        return verified_models_cache[model_id].get("model_weights_size", 0)
    return 0

# Load verified models cache on startup
verified_models_cache = load_verified_models()

# ============================================================================
# Queue Storage
# ============================================================================

def load_queue() -> list:
    """Load queue from file"""
    try:
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[QUEUE] Error loading queue: {e}")
    return []

def save_queue(queue: list):
    """Save queue to file"""
    try:
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        print(f"[QUEUE] Error saving queue: {e}")

# ============================================================================
# Logging Helpers
# ============================================================================

def log_startup_info():
    """Log startup configuration info"""
    print(f"[CONFIG] Base dir: {BASE_DIR}")
    print(f"[CONFIG] Output dir: {OUTPUT_DIR}")
    print(f"[CONFIG] Python: {sys.executable}")
