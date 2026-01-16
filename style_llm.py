"""Style parameter generation via local LLMs (LM Studio / Ollama)."""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

import requests

from schemas import StyleRequest


DEFAULT_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
}


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise ValueError("No JSON object found in LLM response")


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_genre(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    return str(value).strip()


def _normalize_gender(value: Any) -> str:
    if value is None:
        return ""
    gender = str(value).strip().lower()
    if gender in ("male", "female", "auto"):
        return gender
    return ""


def _normalize_bpm(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        bpm = int(float(str(value).strip()))
    except ValueError:
        return None
    bpm = max(60, min(180, bpm))
    return bpm


def _build_messages(req: StyleRequest) -> List[Dict[str, str]]:
    instructions = [
        "You are an expert music producer and A&R.",
        "Return ONLY valid JSON.",
        "Schema: {\"genre\":\"<single>\",\"moods\":[...],\"timbres\":[...],\"instruments\":[...],\"bpm\":120,\"custom_style\":\"...\",\"gender\":\"auto\"}",
        "genre must be a single primary genre string (not a list).",
        "moods/timbres/instruments should be 2-6 concise items.",
        "bpm must be an integer between 60 and 180.",
        "custom_style should be a short, comma-separated descriptor string.",
        "Use ASCII punctuation only.",
    ]

    user_parts = []
    if req.title:
        user_parts.append(f"TITLE: {req.title}")
    if req.language:
        user_parts.append(f"LANGUAGE: {req.language}")
    if req.target_genre:
        user_parts.append(f"TARGET_GENRE: {req.target_genre}")
    if req.seed_words:
        user_parts.append(f"SEED_WORDS: {req.seed_words}")
    if req.lyrics:
        preview = req.lyrics.strip()
        if len(preview) > 800:
            preview = preview[:800] + "..."
        user_parts.append(f"LYRICS_PREVIEW: {preview}")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts) if user_parts else "Generate a chart-ready style."},
    ]


def _call_lmstudio(req: StyleRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["lmstudio"]).rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    payload = {
        "model": req.model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 600,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(req: StyleRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["ollama"]).rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": req.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9,
            "num_predict": 600,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def generate_style(req: StyleRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    messages = _build_messages(req)
    raw_text = _call_ollama(req, messages) if provider == "ollama" else _call_lmstudio(req, messages)
    parsed = _extract_json(raw_text)

    genre = _normalize_genre(parsed.get("genre"))
    moods = _normalize_list(parsed.get("moods"))
    timbres = _normalize_list(parsed.get("timbres"))
    instruments = _normalize_list(parsed.get("instruments"))
    bpm = _normalize_bpm(parsed.get("bpm"))
    custom_style = str(parsed.get("custom_style", "")).strip()
    gender = _normalize_gender(parsed.get("gender"))

    if req.target_genre and not genre:
        genre = req.target_genre.strip()

    return {
        "genre": genre,
        "moods": moods,
        "timbres": timbres,
        "instruments": instruments,
        "bpm": bpm,
        "custom_style": custom_style,
        "gender": gender,
        "raw_text": raw_text,
    }
