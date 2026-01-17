"""
SongGeneration Studio - GPU Detection
Apple Silicon aware GPU/VRAM detection utilities.
"""

import os
import subprocess
import sys
import shutil
from typing import Optional
from pathlib import Path

# ============================================================================
# GPU/VRAM Detection
# ============================================================================

def _darwin_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    total_bytes = None
    free_bytes = None
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    except Exception:
        total_bytes = None

    try:
        vm_stat = subprocess.check_output(["vm_stat"]).decode("utf-8", errors="ignore")
        page_size = 4096
        free_pages = 0
        for line in vm_stat.splitlines():
            if "page size of" in line:
                digits = "".join(ch for ch in line if ch.isdigit())
                if digits:
                    page_size = int(digits)
            if line.startswith("Pages free:") or line.startswith("Pages inactive:") or line.startswith("Pages speculative:"):
                digits = "".join(ch for ch in line if ch.isdigit())
                if digits:
                    free_pages += int(digits)
        free_bytes = free_pages * page_size if free_pages else None
    except Exception:
        free_bytes = None

    return total_bytes, free_bytes


def get_gpu_info() -> dict:
    """Detect GPU / unified memory (Apple Silicon) or NVIDIA VRAM."""
    if sys.platform == "darwin":
        total_bytes, free_bytes = _darwin_memory_bytes()
        total_gb = round(total_bytes / (1024 ** 3), 1) if total_bytes else None
        free_gb = round(free_bytes / (1024 ** 3), 1) if free_bytes else None
        recommended = 'full' if (total_gb or 0) >= 24 else 'low'
        return {
            'available': True,
            'gpu': {
                'name': 'Apple Silicon (Unified Memory)',
                'total_gb': total_gb,
                'free_gb': free_gb,
            },
            'recommended_mode': recommended,
            'can_run_full': (total_gb or 0) >= 24,
            'can_run_low': (total_gb or 0) >= 10,
        }

    if shutil.which("nvidia-smi") is None:
        return {
            'available': False,
            'gpu': None,
            'recommended_mode': 'low',
            'can_run_full': False,
            'can_run_low': False,
        }

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'name': parts[0],
                        'total_mb': int(parts[1]),
                        'free_mb': int(parts[2]),
                        'used_mb': int(parts[3]),
                        'total_gb': round(int(parts[1]) / 1024, 1),
                        'free_gb': round(int(parts[2]) / 1024, 1),
                    })
            if gpus:
                gpu = gpus[0]  # Primary GPU
                if gpu['free_gb'] >= 24:
                    recommended = 'full'
                else:
                    recommended = 'low'
                return {
                    'available': True,
                    'gpu': gpu,
                    'recommended_mode': recommended,
                    'can_run_full': gpu['free_gb'] >= 24,
                    'can_run_low': gpu['free_gb'] >= 10,
                }
    except Exception as e:
        print(f"[GPU] Detection error: {e}")

    return {
        'available': False,
        'gpu': None,
        'recommended_mode': 'low',
        'can_run_full': False,
        'can_run_low': False,
    }

# ============================================================================
# Audio Duration Helper
# ============================================================================

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"[AUDIO] Failed to get duration for {audio_path}: {e}")
    return None

# Global GPU info - initialized on import
gpu_info = get_gpu_info()

def log_gpu_info():
    """Log GPU detection results"""
    if gpu_info['available']:
        print(f"[GPU] Detected: {gpu_info['gpu']['name']}")
        print(f"[GPU] VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total")
        print(f"[GPU] Recommended mode: {gpu_info['recommended_mode']}")
    else:
        print("[GPU] No NVIDIA GPU detected or nvidia-smi not available")

def refresh_gpu_info() -> dict:
    """Refresh GPU info and return updated data"""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info
