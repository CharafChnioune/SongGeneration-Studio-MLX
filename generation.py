"""
SongGeneration Studio - Generation Logic
Song generation, lyrics building, and style control.
"""

import re
import json
import asyncio
import time
import threading
import sys
import random
import shutil
import os

import numpy as np
import librosa
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from config import (
    BASE_DIR,
    DEFAULT_MODEL,
    OUTPUT_DIR,
    UPLOADS_DIR,
    AUTO_PROMPT_PATH,
    SEPARATOR_MODEL_PATH,
    MLX_WEIGHT_PREFERENCE,
)
from gpu import get_audio_duration
from schemas import Section, SongRequest
from timing import save_timing_record, get_timing_stats

# ============================================================================
# State
# ============================================================================

generations: Dict[str, dict] = {}
generation_lock = threading.Lock()

# ============================================================================
# Genre to Auto-Prompt Mapping
# ============================================================================

GENRE_TO_AUTO_PROMPT = {
    "pop": "Pop",
    "r&b": "R&B",
    "rnb": "R&B",
    "dance": "Dance",
    "electronic": "Dance",
    "edm": "Dance",
    "jazz": "Jazz",
    "folk": "Folk",
    "acoustic": "Folk",
    "rock": "Rock",
    "metal": "Metal",
    "heavy metal": "Metal",
    "reggae": "Reggae",
    "chinese": "Chinese Style",
    "chinese style": "Chinese Style",
    "chinese tradition": "Chinese Tradition",
    "chinese opera": "Chinese Opera",
}

# Stronger conditioning defaults to make outputs follow prompts more closely.
_STYLE_FOLLOW_MIN_CFG = 2.2
_STYLE_FOLLOW_MAX_TEMP = 0.7
_MAX_CANDIDATES = 5
_SCORING_SECONDS = 30.0

_GENRE_PRESETS = {
    "hip-hop": {"cfg_coef": 2.6, "temperature": 0.65, "top_k": 60, "top_p": 0.9, "extend_stride": 6},
    "trap": {"cfg_coef": 2.7, "temperature": 0.62, "top_k": 64, "top_p": 0.92, "extend_stride": 6},
    "pop": {"cfg_coef": 2.4, "temperature": 0.75, "top_k": 70, "top_p": 0.95, "extend_stride": 5},
    "r&b": {"cfg_coef": 2.5, "temperature": 0.7, "top_k": 68, "top_p": 0.92, "extend_stride": 5},
    "rnb": {"cfg_coef": 2.5, "temperature": 0.7, "top_k": 68, "top_p": 0.92, "extend_stride": 5},
    "rock": {"cfg_coef": 2.3, "temperature": 0.8, "top_k": 72, "top_p": 0.9, "extend_stride": 6},
    "metal": {"cfg_coef": 2.4, "temperature": 0.78, "top_k": 72, "top_p": 0.9, "extend_stride": 6},
    "edm": {"cfg_coef": 2.2, "temperature": 0.85, "top_k": 80, "top_p": 0.95, "extend_stride": 7},
    "dance": {"cfg_coef": 2.2, "temperature": 0.85, "top_k": 80, "top_p": 0.95, "extend_stride": 7},
}

_GENRE_PRESET_ALIASES = {
    "hiphop": "hip-hop",
    "hip hop": "hip-hop",
    "rap": "hip-hop",
    "gangster rap": "hip-hop",
    "trap rap": "trap",
    "electronic": "edm",
    "house": "edm",
    "techno": "edm",
}

_TAG_DIR = BASE_DIR / "sample" / "description"


def _load_tag_list(filename: str) -> set[str]:
    path = _TAG_DIR / filename
    if not path.exists():
        return set()
    tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.append(tag.lower())
    return set(tags)


_TAG_GENRES = _load_tag_list("genre.txt")
_TAG_EMOTIONS = _load_tag_list("emotion.txt")
_TAG_TIMBRES = _load_tag_list("timbre.txt")
_TAG_INSTRUMENTS = _load_tag_list("instrument.txt")

_GENRE_ALIASES = {
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "hip hop": "hip hop",
    "rnb": "r&b",
    "r&b": "r&b",
    "kpop": "k-pop",
    "k-pop": "k-pop",
}

_EMOTION_ALIASES = {
    "energetic": "intense",
    "motivational": "uplifting",
    "uplift": "uplifting",
    "sadness": "sad",
    "anger": "angry",
    "melancholy": "melancholic",
}

_TIMBRE_ALIASES = {
    "gritty": "dark",
    "harsh": "dark",
    "raspy": "dark",
    "smooth": "soft",
    "airy": "vocal",
}

_INSTRUMENT_ALIASES = {
    "drums": "drums",
    "kick": "drums",
    "snare": "drums",
    "clap": "drums",
    "hi-hats": "drums",
    "hihats": "drums",
    "hi hat": "drums",
    "percussion": "drums",
    "cymbals": "drums",
    "toms": "drums",
    "shakers": "drums",
    "tambourine": "drums",
    "drum machine": "beats",
    "808": "beats",
    "beats": "beats",
    "turntables": "beats",
    "scratch": "beats",
    "piano": "piano",
    "electric piano": "piano",
    "rhodes": "piano",
    "wurlitzer": "piano",
    "keys": "piano",
    "guitar": "guitar",
    "electric guitar": "electric guitar",
    "acoustic guitar": "acoustic guitar",
    "bass": "bass",
    "synth bass": "bass",
    "sub bass": "bass",
    "synthesizer": "synthesizer",
    "synth": "synthesizer",
    "pad": "synthesizer",
    "lead synth": "synthesizer",
    "arpeggiator": "synthesizer",
    "violin": "violin",
    "cello": "cello",
    "saxophone": "saxophone",
    "trumpet": "trumpet",
    "banjo": "banjo",
    "harmonica": "harmonica",
    "fiddle": "fiddle",
}


def _split_tags(value: str) -> List[str]:
    if not value:
        return []
    parts = []
    for part in value.split(","):
        cleaned = part.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def _normalize_tags(
    raw: str,
    allowed: set[str],
    aliases: dict[str, str],
    max_items: int,
) -> tuple[List[str], List[str]]:
    tags: List[str] = []
    extras: List[str] = []
    for item in _split_tags(raw):
        key = aliases.get(item.lower(), item.lower())
        if key in allowed:
            if key not in tags:
                tags.append(key)
        else:
            extras.append(item)
    if max_items:
        tags = tags[:max_items]
    return tags, extras


def _normalize_genre(raw: str) -> tuple[str, List[str]]:
    tags, extras = _normalize_tags(raw, _TAG_GENRES, _GENRE_ALIASES, max_items=1)
    return (tags[0] if tags else ""), extras


def _extract_instrument_bases(raw: str) -> tuple[set[str], List[str]]:
    bases: set[str] = set()
    extras: List[str] = []
    for item in _split_tags(raw):
        lower = item.lower()
        mapped = _INSTRUMENT_ALIASES.get(lower)
        if mapped:
            bases.add(mapped)
            continue
        matched = False
        for key, base in _INSTRUMENT_ALIASES.items():
            if key in lower:
                bases.add(base)
                matched = True
                break
        if not matched:
            extras.append(item)
    return bases, extras


def _pick_instrument_tag(bases: set[str], genre_tag: str) -> str:
    if not bases and genre_tag in ("hip hop", "rap"):
        return "beats"
    if "beats" in bases and "piano" in bases:
        return "beats and piano"
    if "beats" in bases:
        return "beats"
    if "drums" in bases and "piano" in bases:
        return "piano and drums"
    if "drums" in bases and "electric guitar" in bases:
        return "electric guitar and drums"
    if "drums" in bases and "acoustic guitar" in bases:
        return "acoustic guitar and drums"
    if "drums" in bases and "guitar" in bases:
        return "guitar and drums"
    if "drums" in bases and "bass" in bases:
        return "bass and drums"
    if "drums" in bases and "synthesizer" in bases:
        return "synthesizer and drums"
    if "piano" in bases and "guitar" in bases:
        return "piano and guitar"
    if "piano" in bases and "synthesizer" in bases:
        return "piano and synthesizer"
    if "synthesizer" in bases and "electric guitar" in bases:
        return "synthesizer and electric guitar"
    if "synthesizer" in bases and "guitar" in bases:
        return "synthesizer and guitar"
    if "synthesizer" in bases and "bass" in bases:
        return "synthesizer and bass"
    if "piano" in bases and "violin" in bases:
        return "piano and violin"
    if "violin" in bases and "piano" in bases:
        return "violin and piano"
    if "piano" in bases:
        return "piano"
    if "synthesizer" in bases:
        return "synthesizer"
    if "guitar" in bases or "electric guitar" in bases or "acoustic guitar" in bases:
        return "guitar"
    if "violin" in bases:
        return "violin"
    if "saxophone" in bases:
        return "saxophone"
    return ""

_CANDIDATE_VARIATIONS = [
    {"cfg": 0.0, "temp": 0.0, "top_k": 0, "top_p": 0.0},
    {"cfg": 0.2, "temp": -0.05, "top_k": 10, "top_p": 0.05},
    {"cfg": -0.1, "temp": 0.05, "top_k": -10, "top_p": 0.05},
    {"cfg": 0.3, "temp": -0.1, "top_k": 20, "top_p": 0.1},
    {"cfg": -0.2, "temp": 0.1, "top_k": -20, "top_p": 0.1},
]


def _normalize_genre_key(genre: Optional[str]) -> str:
    if not genre:
        return ""
    key = genre.split(",")[0].strip().lower()
    return _GENRE_PRESET_ALIASES.get(key, key)


def _get_genre_preset(genre: Optional[str]) -> Optional[dict]:
    key = _normalize_genre_key(genre)
    return _GENRE_PRESETS.get(key)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_generation_params(request: SongRequest) -> dict:
    params = {
        "cfg_coef": float(request.cfg_coef),
        "temperature": float(request.temperature),
        "top_k": int(request.top_k),
        "top_p": float(request.top_p),
        "extend_stride": int(request.extend_stride),
    }
    preset = _get_genre_preset(request.genre) if request.use_genre_presets else None
    preset_name = _normalize_genre_key(request.genre) if preset else ""
    if preset:
        params.update(preset)
    params["cfg_coef"], params["temperature"] = _apply_style_follow(params["cfg_coef"], params["temperature"])
    params["top_k"] = int(_clamp(params["top_k"], 1, 100))
    params["top_p"] = float(_clamp(params["top_p"], 0.0, 0.98))
    params["extend_stride"] = int(_clamp(params["extend_stride"], 1, 12))
    return {"params": params, "preset_name": preset_name}


def _candidate_params_list(base_params: dict, count: int) -> List[dict]:
    variations = []
    for idx in range(count):
        delta = _CANDIDATE_VARIATIONS[idx % len(_CANDIDATE_VARIATIONS)]
        params = dict(base_params)
        params["cfg_coef"] = _clamp(params["cfg_coef"] + delta["cfg"], 0.1, 5.0)
        params["temperature"] = _clamp(params["temperature"] + delta["temp"], 0.2, 1.2)
        params["top_k"] = int(_clamp(params["top_k"] + delta["top_k"], 1, 100))
        params["top_p"] = float(_clamp(params["top_p"] + delta["top_p"], 0.0, 0.98))
        variations.append(params)
    return variations


def _score_range(value: float, target: float, tol: float) -> float:
    if tol <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tol)


def _score_audio_file(path: Path, genre: Optional[str], bpm: Optional[int]) -> dict:
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=_SCORING_SECONDS)
    except Exception as exc:
        return {"score": -1.0, "error": f"load_failed: {exc}"}
    if y.size == 0:
        return {"score": -1.0, "error": "empty_audio"}

    abs_y = np.abs(y)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    clip_ratio = float(np.mean(abs_y >= 0.98))
    silence_ratio = float(np.mean(abs_y < 1e-4))
    dyn = float(np.percentile(abs_y, 95) - np.percentile(abs_y, 5))
    tempo = None
    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
    except Exception:
        tempo = None

    low_ratio = None
    try:
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = freqs <= 200
        low_energy = float(stft[low_mask].sum())
        total_energy = float(stft.sum()) + 1e-9
        low_ratio = low_energy / total_energy
    except Exception:
        low_ratio = None

    score = 0.0
    score += _score_range(rms, 0.12, 0.08) * 2.0
    score += _score_range(dyn, 0.2, 0.15) * 1.0
    score -= clip_ratio * 5.0
    score -= silence_ratio * 3.0

    if bpm and tempo:
        score += _score_range(tempo, float(bpm), max(8.0, 0.2 * float(bpm))) * 1.5

    genre_key = _normalize_genre_key(genre)
    if low_ratio is not None:
        if genre_key in ("hip-hop", "trap", "r&b"):
            score += low_ratio * 1.2
        elif genre_key in ("pop", "edm", "dance"):
            score += _score_range(low_ratio, 0.28, 0.18) * 0.6
        elif genre_key in ("rock", "metal"):
            score += _score_range(low_ratio, 0.2, 0.15) * 0.4

    return {
        "score": float(score),
        "rms": rms,
        "clip_ratio": clip_ratio,
        "silence_ratio": silence_ratio,
        "dynamic_range": dyn,
        "tempo": tempo,
        "low_ratio": low_ratio,
    }


def _materialize_candidate_output(candidate_files: List[Path], output_dir: Path) -> List[Path]:
    audio_dir = output_dir / "audios"
    audio_dir.mkdir(exist_ok=True)
    copied = []
    for path in candidate_files:
        target = audio_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _apply_style_follow(cfg_coef: float, temperature: float) -> tuple[float, float]:
    effective_cfg = max(float(cfg_coef), _STYLE_FOLLOW_MIN_CFG)
    effective_temp = min(float(temperature), _STYLE_FOLLOW_MAX_TEMP)
    return effective_cfg, effective_temp

# ============================================================================
# Lyrics Normalization (matches official HuggingFace Gradio app)
# ============================================================================

# Regex to filter lyrics - keeps only:
# - Word chars (\w), whitespace (\s), brackets [], hyphen -
# - CJK Chinese (u4e00-u9fff), Japanese Hiragana (u3040-u309f),
#   Japanese Katakana (u30a0-u30ff), Korean (uac00-ud7af)
# - Extended Latin (u00c0-u017f)
LYRICS_FILTER_REGEX = re.compile(
    r"[^\w\s\[\]\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]"
)

# Section types that have vocals (lyrics should be cleaned)
VOCAL_SECTION_TYPES = {"verse", "chorus", "bridge", "prechorus"}


def clean_lyrics_line(line: str) -> str:
    """Clean a single line of lyrics by removing unwanted punctuation."""
    cleaned = LYRICS_FILTER_REGEX.sub("", line)
    return cleaned.strip()


AUDIO_EXT_PRIORITY = {
    ".flac": 0,
    ".wav": 1,
    ".mp3": 2,
}


def _select_weights(model_path: Path) -> Path:
    for name in MLX_WEIGHT_PREFERENCE:
        candidate = model_path / name
        if candidate.exists():
            return candidate
    expected = ", ".join(MLX_WEIGHT_PREFERENCE)
    raise FileNotFoundError(f"Model weights not found in {model_path} (expected {expected})")


def _pick_preferred_audio_files(audio_files: List[Path]) -> List[Path]:
    best: Dict[str, tuple[int, Path]] = {}
    for path in audio_files:
        ext = path.suffix.lower()
        priority = AUDIO_EXT_PRIORITY.get(ext, 99)
        stem = path.stem
        current = best.get(stem)
        if current is None or priority < current[0]:
            best[stem] = (priority, path)
    return [item[1] for item in sorted(best.values(), key=lambda item: item[1].name)]


def _collect_audio_files(search_dirs: List[Path]) -> List[Path]:
    candidates: List[Path] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in (".flac", ".wav", ".mp3"):
            candidates.extend(search_dir.glob(f"*{ext}"))
    if not candidates:
        return []
    return _pick_preferred_audio_files(candidates)


def _separator_backend(path: Path) -> str:
    if path.suffix.lower() == ".mlpackage":
        return "coreml"
    return "onnx"


def _resolve_separator_model() -> Optional[Path]:
    if SEPARATOR_MODEL_PATH.exists():
        return SEPARATOR_MODEL_PATH
    candidates: List[Path] = []
    if SEPARATOR_MODEL_PATH.suffix.lower() != ".mlpackage":
        candidates.append(SEPARATOR_MODEL_PATH.with_suffix(".mlpackage"))
    if SEPARATOR_MODEL_PATH.suffix.lower() != ".onnx":
        candidates.append(SEPARATOR_MODEL_PATH.with_suffix(".onnx"))
    for path in candidates:
        if path.exists():
            return path
    return None


def _append_generation_log(
    output_dir: Path,
    gen_id: str,
    message: str,
    payload: Optional[dict] = None,
) -> None:
    log_path = output_dir / "generation.log"
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().isoformat()}] [{gen_id}] {message}\n")
            if payload is not None:
                try:
                    handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
                except TypeError:
                    handle.write(str(payload))
                handle.write("\n")
    except Exception as exc:
        print(f"[GEN {gen_id}] Warning: failed to write log: {exc}")


def _normalize_auto_prompt_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() in ("none", "off", "disabled"):
        return None
    return cleaned


_DURATION_TAG_SECONDS = {
    "short": 6.0,
    "medium": 12.0,
    "long": 12.0,
}

_MLX_SAFE_MAX_DURATION = float(os.environ.get("SONGGEN_MLX_MAX_DURATION", "180"))

_BAR_TARGETS = {
    "verse": 2.0,
    "chorus": 2.0,
    "prechorus": 2.0,
    "bridge": 2.0,
    "rap": 2.0,
    "hook": 2.0,
}
_WORDS_PER_SEC = {
    "verse": 2.6,
    "chorus": 2.4,
    "prechorus": 2.5,
    "bridge": 2.5,
    "rap": 3.2,
    "hook": 2.5,
}
_DEFAULT_BARS_PER_LINE = 2.0
_DEFAULT_WORDS_PER_SEC = 2.6
_LINE_GAP_SECONDS = 0.15


def _normalize_bpm(bpm: Optional[float]) -> float:
    if not bpm or bpm <= 0:
        return 120.0
    return max(60.0, min(200.0, float(bpm)))


def _count_words(text: str) -> int:
    cleaned = clean_lyrics_line(text or "")
    return len(cleaned.split()) if cleaned else 0


def _bars_to_seconds(bars: float, bpm: float) -> float:
    bar_seconds = 240.0 / bpm
    return bars * bar_seconds


def _estimate_vocal_duration(lines: List[str], bpm: float, base_type: str) -> float:
    bpm = _normalize_bpm(bpm)
    bars_per_line = _BAR_TARGETS.get(base_type, _DEFAULT_BARS_PER_LINE)
    words_per_sec = _WORDS_PER_SEC.get(base_type, _DEFAULT_WORDS_PER_SEC)
    if not lines:
        return max(4.0, _bars_to_seconds(bars_per_line, bpm))

    total = 0.0
    for idx, line in enumerate(lines):
        words = _count_words(line)
        by_words = (words / words_per_sec) if words else 0.0
        by_bars = _bars_to_seconds(bars_per_line, bpm)
        line_seconds = max(1.0, by_bars, by_words)
        total += line_seconds
        if idx < len(lines) - 1:
            total += _LINE_GAP_SECONDS
    return total


def _estimate_duration(
    sections: List[Section],
    model_path: Path,
    allow_intro_outro: bool,
    bpm: Optional[float],
) -> float:
    cfg_path = model_path / "config.yaml"
    min_dur = 30.0
    max_dur = 150.0
    try:
        cfg = OmegaConf.load(cfg_path)
        min_dur = float(getattr(cfg, "min_dur", min_dur))
        max_dur = float(getattr(cfg, "max_dur", max_dur))
    except Exception as exc:
        print(f"[GEN] Warning: failed to load {cfg_path}: {exc}")

    total = 0.0
    has_lyrics = False
    for section in sections:
        base, *rest = section.type.split("-", 1)
        base = base.lower()
        duration_tag = rest[0].lower() if rest else ""
        if duration_tag == "long":
            duration_tag = "medium"
        if base == "pre" and duration_tag == "chorus":
            base = "prechorus"
            duration_tag = ""
        lyrics = (section.lyrics or "").strip()
        use_lyrics = bool(lyrics) and (base not in ("intro", "outro", "inst") or (base in ("intro", "outro") and allow_intro_outro))
        if use_lyrics:
            lines = [line for line in lyrics.splitlines() if line.strip()]
            total += _estimate_vocal_duration(lines, bpm, base)
            if lines:
                has_lyrics = True
            continue
        if base in ("intro", "outro", "inst") and duration_tag in _DURATION_TAG_SECONDS:
            total += _DURATION_TAG_SECONDS[duration_tag]
            continue
        if base in ("intro", "outro") and allow_intro_outro:
            total += _estimate_vocal_duration([], bpm, base)
            continue
        if base in ("intro", "outro"):
            total += 4.0
        elif base == "inst":
            total += 8.0
        else:
            total += 6.0

    if has_lyrics:
        total += 0.5
        total = max(6.0, min(max_dur, total))
    else:
        total = max(min_dur, min(max_dur, total))
    return total


# ============================================================================
# Helper Functions
# ============================================================================

def is_generation_active() -> bool:
    """Check if there's currently an active generation running."""
    # Check if any generation is in active state
    for gen in generations.values():
        if gen.get("status") in ("pending", "processing"):
            return True
    return False


def get_active_generation_id() -> Optional[str]:
    """Get the ID of the currently active generation, if any."""
    for gen_id, gen in generations.items():
        if gen.get("status") in ("pending", "processing"):
            return gen_id
    return None


def build_lyrics_string(sections: List[Section], allow_intro_outro: bool = False) -> str:
    """Build the lyrics string in SongGeneration format.

    Normalizes lyrics to match the official HuggingFace Gradio app:
    - Filters out special punctuation (keeps letters, numbers, CJK, hyphens)
    - Joins lines with '.' for vocal sections
    - Joins all sections with ' ; '
    """
    parts = []
    vocal_types = set(VOCAL_SECTION_TYPES)
    if allow_intro_outro:
        vocal_types.update({"intro", "outro"})
    for section in sections:
        # Extract base type (e.g., "verse" from "verse" or "intro" from "intro-short")
        base_type = section.type.split('-')[0].lower()
        tag_type = section.type
        if base_type in ("intro", "outro", "inst") and section.type.endswith("-long"):
            tag_type = f"{base_type}-medium"
        elif base_type in ("intro", "outro", "inst") and "-" not in section.type:
            tag_type = f"{base_type}-short"
        tag = f"[{tag_type}]"

        if section.lyrics and base_type in vocal_types:
            # Vocal section with lyrics - clean each line
            lines = section.lyrics.strip().split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned = clean_lyrics_line(line)
                if cleaned:
                    cleaned_lines.append(cleaned)

            if cleaned_lines:
                # Join cleaned lines with '.' as per official app
                lyrics_str = '.'.join(cleaned_lines)
                parts.append(f"{tag} {lyrics_str}")
            else:
                parts.append(tag)
        else:
            # Instrumental section or no lyrics - just the tag
            parts.append(tag)

    return " ; ".join(parts)


def build_description(request: SongRequest, exclude_genre: bool = False) -> str:
    """Build the description string for style control."""
    parts = []

    custom_extras: List[str] = []

    if request.gender and request.gender != "auto":
        parts.append(request.gender.lower())

    timbres, timbre_extras = _normalize_tags(request.timbre or "", _TAG_TIMBRES, _TIMBRE_ALIASES, max_items=2)
    parts.extend(timbres)
    custom_extras.extend(timbre_extras)

    genre_tag = ""
    if not exclude_genre and request.genre and request.genre != "Auto":
        genre_tag, genre_extras = _normalize_genre(request.genre)
        if genre_tag:
            parts.append(genre_tag)
        custom_extras.extend(genre_extras)

    emotions, emotion_extras = _normalize_tags(request.emotion or "", _TAG_EMOTIONS, _EMOTION_ALIASES, max_items=2)
    parts.extend(emotions)
    custom_extras.extend(emotion_extras)

    bases, instrument_extras = _extract_instrument_bases(request.instruments or "")
    instrument_tag = _pick_instrument_tag(bases, genre_tag)
    if instrument_tag and instrument_tag in _TAG_INSTRUMENTS:
        parts.append(instrument_tag)
    custom_extras.extend(instrument_extras)

    if request.custom_style:
        custom_extras.append(request.custom_style)

    if custom_extras:
        unique = []
        seen = set()
        for item in custom_extras:
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(cleaned)
        if unique:
            parts.append(", ".join(unique[:4]))

    if request.bpm:
        parts.append(f"the bpm is {request.bpm}")

    return ", ".join(parts) + "." if parts else ""


def restore_library():
    """Restore completed generations from output directory on startup."""
    global generations
    restored = 0

    print(f"[LIBRARY] Scanning output directory: {OUTPUT_DIR}")

    if not OUTPUT_DIR.exists():
        return

    subdirs = list(OUTPUT_DIR.iterdir())
    for subdir in subdirs:
        if not subdir.is_dir():
            continue

        gen_id = subdir.name

        audio_files = _collect_audio_files([subdir, subdir / "audios"])
        if not audio_files:
            continue

        metadata_path = subdir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[LIBRARY] Error loading metadata for {gen_id}: {e}")

        try:
            file_mtime = datetime.fromtimestamp(audio_files[0].stat().st_mtime).isoformat()
        except:
            file_mtime = datetime.now().isoformat()

        if not metadata.get("cover"):
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                cover_path = subdir / f"cover{ext}"
                if cover_path.exists():
                    metadata["cover"] = f"cover{ext}"
                    try:
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[LIBRARY] Warning: Could not update metadata for {gen_id}: {e}")
                    break

        duration = metadata.get("duration")
        if duration is None and audio_files:
            duration = get_audio_duration(audio_files[0])
            if duration is not None:
                metadata["duration"] = duration
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[LIBRARY] Warning: Could not save duration for {gen_id}: {e}")

        # Determine output_mode and track_labels from files
        output_mode = metadata.get("output_mode", "mixed")
        track_labels = []
        ordered_files = list(audio_files)

        # Check if this is a separate mode generation (has vocal and bgm files)
        file_names = [f.stem for f in audio_files]
        has_vocal = any(n.endswith('_vocal') for n in file_names)
        has_bgm = any(n.endswith('_bgm') for n in file_names)

        if has_vocal and has_bgm:
            # This is a separate mode generation - order the files
            main_file = None
            vocal_file = None
            bgm_file = None
            for f in audio_files:
                if f.stem.endswith('_vocal'):
                    vocal_file = f
                elif f.stem.endswith('_bgm'):
                    bgm_file = f
                else:
                    main_file = f
            ordered_files = [f for f in [main_file, vocal_file, bgm_file] if f]
            track_labels = ["Full Song", "Vocals", "Instrumental"]
            output_mode = "separate"
        else:
            track_labels = ["Full Song"] if output_mode == 'mixed' else ["Vocals" if output_mode == 'vocal' else "Instrumental"]

        generations[gen_id] = {
            "id": gen_id,
            "status": "completed",
            "progress": 100,
            "message": "Complete",
            "title": metadata.get("title", "Untitled"),
            "model": metadata.get("model", "unknown"),
            "created_at": metadata.get("created_at", file_mtime),
            "completed_at": metadata.get("completed_at", file_mtime),
            "duration": duration,
            "output_files": [str(f) for f in ordered_files],
            "audio_files": [f.name for f in ordered_files],
            "output_dir": str(subdir),
            "track_labels": track_labels,
            "output_mode": output_mode,
            "metadata": metadata if metadata else {
                "title": "Untitled",
                "model": "unknown",
                "created_at": file_mtime,
            }
        }
        restored += 1

    print(f"[LIBRARY] Restored {restored} generation(s)")


async def run_generation(
    gen_id: str,
    request: SongRequest,
    reference_path: Optional[str],
    notify_generation_update,
    notify_library_update,
    notify_models_update
):
    """Run the MLX SongGeneration inference."""
    global generations

    try:
        print(f"[GEN {gen_id}] Starting generation...")
        generations[gen_id]["status"] = "processing"
        generations[gen_id]["started_at"] = datetime.now().isoformat()
        generations[gen_id]["message"] = "Initializing..."
        generations[gen_id]["progress"] = 0

        await notify_models_update()

        model_id = request.model or DEFAULT_MODEL
        num_sections = len(request.sections) if request.sections else 5
        timing_stats = get_timing_stats()
        estimated_seconds = 180

        if timing_stats.get("has_history") and model_id in timing_stats.get("models", {}):
            model_timing = timing_stats["models"][model_id]
            by_sections = model_timing.get("by_sections", {})
            if str(num_sections) in by_sections:
                estimated_seconds = by_sections[str(num_sections)]
            else:
                estimated_seconds = model_timing.get("avg_time", 180)

        generations[gen_id]["estimated_seconds"] = estimated_seconds
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update(generations)

        model_path = BASE_DIR / model_id
        if not model_path.exists():
            raise Exception(f"Model not found: {model_id}")
        weights_path = _select_weights(model_path)

        print(f"[GEN {gen_id}] Using model: {model_id}")
        generations[gen_id]["model"] = model_id

        input_file = UPLOADS_DIR / f"{gen_id}_input.jsonl"
        output_subdir = OUTPUT_DIR / gen_id
        output_subdir.mkdir(exist_ok=True)

        _append_generation_log(
            output_subdir,
            gen_id,
            "Request received",
            {
                "model": model_id,
                "title": request.title,
                "genre": request.genre,
                "emotion": request.emotion,
                "timbre": request.timbre,
                "instruments": request.instruments,
                "custom_style": request.custom_style,
                "bpm": request.bpm,
                "output_mode": request.output_mode,
                "auto_prompt_type": request.auto_prompt_type,
                "reference_audio_id": request.reference_audio_id,
                "use_genre_presets": request.use_genre_presets,
                "num_candidates": request.num_candidates,
                "auto_select_best": request.auto_select_best,
                "arrangement_template": request.arrangement_template,
                "sections": [
                    {"type": s.type, "lyrics_len": len(s.lyrics or "")} for s in request.sections
                ],
            },
        )

        lyrics = build_lyrics_string(
            request.sections,
            allow_intro_outro=request.allow_intro_outro_lyrics,
        )

        if reference_path and request.auto_prompt_type:
            raise Exception("auto_prompt_type cannot be used with reference audio")

        input_data = {
            "idx": gen_id,
            "gt_lyric": lyrics,
        }

        description = ""
        resolved = _resolve_generation_params(request)
        gen_params = resolved["params"]
        preset_name = resolved["preset_name"]

        if reference_path:
            input_data["prompt_audio_path"] = reference_path
        else:
            auto_type = _normalize_auto_prompt_type(request.auto_prompt_type)
            auto_type_provided = auto_type is not None and request.auto_prompt_type is not None
            genre_for_auto_prompt = None
            auto_prompt_source = "none"

            genre_value = (request.genre or "").strip()
            has_genre = bool(genre_value) and genre_value.lower() != "auto"
            first_genre = ""
            if has_genre:
                first_genre = genre_value.split(',')[0].strip().lower()
                if not auto_type_provided:
                    if first_genre in GENRE_TO_AUTO_PROMPT:
                        auto_type = GENRE_TO_AUTO_PROMPT[first_genre]
                        genre_for_auto_prompt = first_genre
                        auto_prompt_source = "genre_map"
                    else:
                        auto_type = "Auto"
                        auto_prompt_source = "default"
                else:
                    auto_prompt_source = "explicit"
            else:
                if auto_type is None:
                    auto_type = "Auto"
                    auto_prompt_source = "default"
                else:
                    auto_prompt_source = "explicit"

            if auto_type:
                input_data["auto_prompt_audio_type"] = auto_type

            exclude_genre = genre_for_auto_prompt is not None
            description = build_description(request, exclude_genre=exclude_genre)

            if description:
                input_data["descriptions"] = description

            _append_generation_log(
                output_subdir,
                gen_id,
                "Auto prompt selection",
                {
                    "auto_prompt_type": auto_type,
                    "auto_prompt_source": auto_prompt_source,
                    "genre_value": genre_value,
                    "first_genre": first_genre,
                    "exclude_genre": exclude_genre,
                    "description": description,
                },
            )

        input_data["cfg_coef"] = gen_params["cfg_coef"]
        input_data["temperature"] = gen_params["temperature"]
        input_data["top_k"] = gen_params["top_k"]
        input_data["top_p"] = gen_params["top_p"]
        input_data["extend_stride"] = gen_params["extend_stride"]

        print(f"[GEN {gen_id}] Lyrics: {lyrics[:200]}...")
        print(f"[GEN {gen_id}] Input data: {json.dumps(input_data, indent=2)}")

        _append_generation_log(
            output_subdir,
            gen_id,
            "Input data",
            {"lyrics_preview": lyrics[:400], "input_data": input_data},
        )

        if (
            gen_params["cfg_coef"] != request.cfg_coef
            or gen_params["temperature"] != request.temperature
            or gen_params["top_k"] != request.top_k
            or gen_params["top_p"] != request.top_p
            or gen_params["extend_stride"] != request.extend_stride
            or preset_name
        ):
            _append_generation_log(
                output_subdir,
                gen_id,
                "Parameter adjustment",
                {
                    "requested_cfg": request.cfg_coef,
                    "requested_temperature": request.temperature,
                    "requested_top_k": request.top_k,
                    "requested_top_p": request.top_p,
                    "requested_extend_stride": request.extend_stride,
                    "effective_cfg": gen_params["cfg_coef"],
                    "effective_temperature": gen_params["temperature"],
                    "effective_top_k": gen_params["top_k"],
                    "effective_top_p": gen_params["top_p"],
                    "effective_extend_stride": gen_params["extend_stride"],
                    "min_cfg": _STYLE_FOLLOW_MIN_CFG,
                    "max_temperature": _STYLE_FOLLOW_MAX_TEMP,
                    "genre_preset": preset_name,
                },
            )

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False)
            f.write('\n')

        separator_model = None
        if reference_path:
            separator_model = _resolve_separator_model()
            if not separator_model:
                raise Exception(
                    f"Separator model not found (expected {SEPARATOR_MODEL_PATH} or {SEPARATOR_MODEL_PATH.with_suffix('.mlpackage')})"
                )

        if not reference_path and not AUTO_PROMPT_PATH.exists():
            raise Exception(f"Auto prompt file not found: {AUTO_PROMPT_PATH}")

        generations[gen_id]["message"] = "Loading Model..."
        generations[gen_id]["progress"] = 10
        notify_generation_update(gen_id, generations[gen_id])

        gen_type = request.output_mode or "mixed"
        if gen_type not in ("mixed", "vocal", "bgm", "separate"):
            gen_type = "mixed"

        duration = _estimate_duration(
            request.sections,
            model_path,
            allow_intro_outro=request.allow_intro_outro_lyrics,
            bpm=request.bpm,
        )
        if _MLX_SAFE_MAX_DURATION > 0 and duration > _MLX_SAFE_MAX_DURATION:
            _append_generation_log(
                output_subdir,
                gen_id,
                "Duration capped",
                {"estimated_duration": duration, "cap_seconds": _MLX_SAFE_MAX_DURATION},
            )
            duration = _MLX_SAFE_MAX_DURATION

        def build_cmd(save_dir: Path, params: dict, seed: Optional[int]) -> List[str]:
            cmd = [
                sys.executable,
                str(BASE_DIR / "generate_mlx.py"),
                "--ckpt_path",
                str(model_path),
                "--weights",
                str(weights_path),
                "--input_jsonl",
                str(input_file),
                "--save_dir",
                str(save_dir),
                "--generate_type",
                gen_type,
                "--auto_prompt_path",
                str(AUTO_PROMPT_PATH),
                "--duration",
                str(duration),
                "--cfg_coef",
                str(params["cfg_coef"]),
                "--temperature",
                str(params["temperature"]),
                "--top_k",
                str(params["top_k"]),
                "--top_p",
                str(params["top_p"]),
                "--extend_stride",
                str(params["extend_stride"]),
            ]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if reference_path and separator_model:
                cmd += [
                    "--separator_backend",
                    _separator_backend(separator_model),
                    "--separator_model",
                    str(separator_model),
                ]
            return cmd

        candidate_count = max(1, min(int(request.num_candidates or 1), _MAX_CANDIDATES))
        candidate_params = _candidate_params_list(gen_params, candidate_count)
        candidates_dir = output_subdir / "candidates"
        if candidate_count > 1:
            candidates_dir.mkdir(exist_ok=True)

        candidate_results = []
        total_gen_time = 0.0

        for idx in range(candidate_count):
            candidate_id = f"cand_{idx + 1:02d}"
            candidate_dir = candidates_dir / candidate_id if candidate_count > 1 else output_subdir
            candidate_dir.mkdir(exist_ok=True)
            seed = random.randint(0, 2**31 - 1)
            params = candidate_params[idx]
            cmd = build_cmd(candidate_dir, params, seed)

            _append_generation_log(
                output_subdir,
                gen_id,
                "Candidate command",
                {"candidate": candidate_id, "seed": seed, "params": params, "cmd": cmd},
            )

            generations[gen_id]["message"] = f"Generating music (take {idx + 1}/{candidate_count})..."
            generations[gen_id]["progress"] = 35 + int((idx / max(1, candidate_count)) * 40)
            generations[gen_id]["stage"] = "generating"
            notify_generation_update(gen_id, generations[gen_id])

            start_time = time.time()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(BASE_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            candidate_time = time.time() - start_time
            total_gen_time += candidate_time

            stdout_text = stdout.decode("utf-8", errors="ignore").strip() if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="ignore").strip() if stderr else ""
            if stdout_text:
                print(f"[GEN {gen_id}] MLX output ({candidate_id}):\n{stdout_text}")
            if stderr_text:
                print(f"[GEN {gen_id}] MLX error output ({candidate_id}):\n{stderr_text}")

            _append_generation_log(
                output_subdir,
                gen_id,
                "Candidate output",
                {
                    "candidate": candidate_id,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": proc.returncode,
                    "elapsed_seconds": candidate_time,
                },
            )

            if proc.returncode != 0:
                error_detail = stderr_text or stdout_text or "unknown error"
                if proc.returncode < 0:
                    signal_id = -proc.returncode
                    error_detail = f"process_killed_signal_{signal_id} (possible OOM)."
                    if stderr_text:
                        error_detail = f"{error_detail} {stderr_text}"
                candidate_results.append({
                    "id": candidate_id,
                    "status": "failed",
                    "error": error_detail,
                })
                continue

            output_files = _collect_audio_files([candidate_dir / "audios", candidate_dir])
            if not output_files:
                candidate_results.append({
                    "id": candidate_id,
                    "status": "failed",
                    "error": "no_output_files",
                })
                continue

            score_info = _score_audio_file(output_files[0], request.genre, request.bpm)
            _append_generation_log(
                output_subdir,
                gen_id,
                "Candidate score",
                {"candidate": candidate_id, "score_details": score_info},
            )
            candidate_results.append({
                "id": candidate_id,
                "status": "ok",
                "seed": seed,
                "params": params,
                "score": score_info.get("score", -1.0),
                "score_details": score_info,
                "output_files": output_files,
            })

        successful = [c for c in candidate_results if c.get("status") == "ok"]
        if not successful:
            details = candidate_results[-1].get("error") if candidate_results else "unknown error"
            raise Exception(f"MLX generation failed: {details}")

        chosen = successful[0]
        if candidate_count > 1 and request.auto_select_best:
            chosen = max(successful, key=lambda c: c.get("score", -1.0))

        _append_generation_log(
            output_subdir,
            gen_id,
            "Candidate selection",
            {
                "auto_select_best": request.auto_select_best,
                "chosen": chosen.get("id"),
                "scores": [
                    {"id": c.get("id"), "score": c.get("score"), "status": c.get("status")}
                    for c in candidate_results
                ],
            },
        )

        if candidate_count > 1:
            output_files = _materialize_candidate_output(chosen["output_files"], output_subdir)
        else:
            output_files = chosen["output_files"]
        if not output_files:
            raise Exception("No output files generated")

        gen_time = total_gen_time

        track_labels = []
        if gen_type == 'separate' and len(output_files) >= 3:
            main_file = None
            vocal_file = None
            bgm_file = None
            for f in output_files:
                name = f.stem
                if name.endswith('_vocal'):
                    vocal_file = f
                elif name.endswith('_bgm'):
                    bgm_file = f
                else:
                    main_file = f
            output_files = [f for f in [main_file, vocal_file, bgm_file] if f]
            track_labels = ["Full Song", "Vocals", "Instrumental"]
        else:
            track_labels = ["Full Song"] if gen_type == 'mixed' else ["Vocals" if gen_type == 'vocal' else "Instrumental"]

        audio_duration = get_audio_duration(output_files[0]) if output_files else None

        generations[gen_id]["status"] = "completed"
        generations[gen_id]["progress"] = 100
        generations[gen_id]["message"] = "Song generated successfully!"
        generations[gen_id]["output_files"] = [str(f) for f in output_files]
        generations[gen_id]["output_file"] = str(output_files[0])
        generations[gen_id]["track_labels"] = track_labels
        generations[gen_id]["output_mode"] = gen_type
        generations[gen_id]["completed_at"] = datetime.now().isoformat()
        generations[gen_id]["duration"] = audio_duration
        generations[gen_id]["candidates"] = [
            {
                "id": c.get("id"),
                "status": c.get("status"),
                "score": c.get("score"),
                "seed": c.get("seed"),
                "params": c.get("params"),
            }
            for c in candidate_results
        ]

        await notify_models_update()

        generation_time_seconds = int(gen_time)
        total_lyrics_length = sum(len(s.lyrics or '') for s in request.sections)
        num_sections = len(request.sections)
        has_lyrics = total_lyrics_length > 0

        try:
            metadata_path = output_subdir / "metadata.json"
            metadata = {
                "id": gen_id,
                "title": request.title,
                "model": model_id,
                "created_at": generations[gen_id].get("created_at", datetime.now().isoformat()),
                "completed_at": generations[gen_id]["completed_at"],
                "generation_time_seconds": generation_time_seconds,
                "duration": audio_duration,
                "sections": [s.model_dump() for s in request.sections],
                "genre": request.genre,
                "gender": request.gender,
                "timbre": request.timbre,
                "emotion": request.emotion,
                "bpm": request.bpm,
                "instruments": request.instruments,
                "output_mode": gen_type,
                "num_sections": num_sections,
                "has_lyrics": has_lyrics,
                "total_lyrics_length": total_lyrics_length,
                "used_model_server": False,
                "reference_audio_id": request.reference_audio_id,
                "cfg_coef": gen_params["cfg_coef"],
                "temperature": gen_params["temperature"],
                "top_k": gen_params["top_k"],
                "top_p": gen_params["top_p"],
                "extend_stride": gen_params["extend_stride"],
                "genre_preset": preset_name,
                "use_genre_presets": request.use_genre_presets,
                "arrangement_template": request.arrangement_template,
                "num_candidates": candidate_count,
                "auto_select_best": request.auto_select_best,
                "candidates": [
                    {
                        "id": c.get("id"),
                        "status": c.get("status"),
                        "score": c.get("score"),
                        "seed": c.get("seed"),
                        "params": c.get("params"),
                    }
                    for c in candidate_results
                ],
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            generations[gen_id]["metadata"] = metadata
            save_timing_record(metadata)
        except Exception as meta_err:
            print(f"[GEN {gen_id}] Warning: Could not save metadata: {meta_err}")

        input_file.unlink(missing_ok=True)
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update(generations)
        return

    except Exception as e:
        print(f"[GEN {gen_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            if "output_subdir" in locals() and output_subdir.exists():
                _append_generation_log(
                    output_subdir,
                    gen_id,
                    "Generation failed",
                    {"error": str(e)},
                )
        except Exception:
            pass
        if gen_id in generations:
            generations[gen_id]["status"] = "failed"
            generations[gen_id]["message"] = str(e)

        notify_generation_update(gen_id, generations.get(gen_id, {}))
        notify_library_update(generations)
        await notify_models_update()
