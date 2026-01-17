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


def _extract_json_safe(text: str) -> Dict[str, Any] | None:
    try:
        return _extract_json(text)
    except Exception:
        return None


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


def _build_step_messages(step: str, req: StyleRequest, context: Dict[str, Any]) -> List[Dict[str, str]]:
    instructions: List[str] = [
        "You are an expert music producer and A&R.",
        "Return ONLY valid JSON.",
        "Use ASCII punctuation only.",
    ]
    schema = ""
    if step == "genre":
        schema = "{\"genre\":\"<single>\"}"
        instructions += [
            f"Schema: {schema}",
            "genre must be a single primary genre string (not a list).",
        ]
    elif step == "moods":
        schema = "{\"moods\":[...]}"
        instructions += [
            f"Schema: {schema}",
            "moods should be 3-6 concise items.",
        ]
    elif step == "timbres":
        schema = "{\"timbres\":[...],\"instruments\":[...]}"
        instructions += [
            f"Schema: {schema}",
            "timbres should be 3-6 concise items.",
            "instruments should be 4-10 concise items.",
        ]
    elif step == "details":
        schema = "{\"bpm\":120,\"custom_style\":\"...\",\"gender\":\"auto\"}"
        instructions += [
            f"Schema: {schema}",
            "bpm must be an integer between 60 and 180.",
            "custom_style should be a short, comma-separated descriptor string.",
            "gender must be male, female, or auto.",
        ]
    else:
        schema = "{\"genre\":\"<single>\",\"moods\":[...],\"timbres\":[...],\"instruments\":[...],\"bpm\":120,\"custom_style\":\"...\",\"gender\":\"auto\"}"
        instructions.append(f"Schema: {schema}")

    user_parts: List[str] = []
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

    if context.get("genre"):
        user_parts.append(f"GENRE: {context['genre']}")
    if context.get("moods"):
        user_parts.append(f"MOODS: {', '.join(context['moods'])}")
    if context.get("timbres"):
        user_parts.append(f"TIMBRES: {', '.join(context['timbres'])}")
    if context.get("instruments"):
        user_parts.append(f"INSTRUMENTS: {', '.join(context['instruments'])}")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts) if user_parts else "Generate style details."},
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


def _call_provider(provider: str, req: StyleRequest, messages: List[Dict[str, str]]) -> str:
    if provider == "ollama":
        return _call_ollama(req, messages)
    return _call_lmstudio(req, messages)


def _call_step(provider: str, req: StyleRequest, messages: List[Dict[str, str]]) -> tuple[str, str, Dict[str, Any]]:
    raw = _call_provider(provider, req, messages)
    parsed = _extract_json_safe(raw)
    retry_raw = ""
    if parsed is None:
        retry_messages = messages + [{"role": "user", "content": "Return ONLY JSON for the schema."}]
        retry_raw = _call_provider(provider, req, retry_messages)
        parsed = _extract_json_safe(retry_raw)
    return raw, retry_raw, parsed or {}


def generate_style(req: StyleRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    context: Dict[str, Any] = {}
    raw_steps: Dict[str, Any] = {}

    genre_messages = _build_step_messages("genre", req, context)
    genre_raw, genre_retry, genre_parsed = _call_step(provider, req, genre_messages)
    genre = _normalize_genre(genre_parsed.get("genre")) or ""
    if req.target_genre and not genre:
        genre = req.target_genre.strip()
    context["genre"] = genre
    raw_steps["genre"] = {"raw": genre_raw, "raw_retry": genre_retry}

    mood_messages = _build_step_messages("moods", req, context)
    mood_raw, mood_retry, mood_parsed = _call_step(provider, req, mood_messages)
    moods = _normalize_list(mood_parsed.get("moods"))
    context["moods"] = moods
    raw_steps["moods"] = {"raw": mood_raw, "raw_retry": mood_retry}

    timbre_messages = _build_step_messages("timbres", req, context)
    timbre_raw, timbre_retry, timbre_parsed = _call_step(provider, req, timbre_messages)
    timbres = _normalize_list(timbre_parsed.get("timbres"))
    instruments = _normalize_list(timbre_parsed.get("instruments"))
    context["timbres"] = timbres
    context["instruments"] = instruments
    raw_steps["timbres"] = {"raw": timbre_raw, "raw_retry": timbre_retry}

    details_messages = _build_step_messages("details", req, context)
    details_raw, details_retry, details_parsed = _call_step(provider, req, details_messages)
    bpm = _normalize_bpm(details_parsed.get("bpm"))
    custom_style = str(details_parsed.get("custom_style", "")).strip()
    gender = _normalize_gender(details_parsed.get("gender"))
    raw_steps["details"] = {"raw": details_raw, "raw_retry": details_retry}

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
        "raw_text": json.dumps(raw_steps, ensure_ascii=False),
    }
