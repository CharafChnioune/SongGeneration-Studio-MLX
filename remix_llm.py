"""Remix generation via local LLMs (LM Studio / Ollama)."""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

import requests

from schemas import RemixRequest, LyricsRequest, LyricsSection
from lyrics_llm import generate_lyrics


DEFAULT_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
}

OUTPUT_MODES = {"mixed", "vocal", "bgm", "separate"}
VOCAL_SECTION_TYPES = {"verse", "chorus", "bridge", "prechorus"}


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


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_gender(value: Any) -> str:
    if value is None:
        return ""
    gender = str(value).strip().lower()
    if gender in ("male", "female", "auto"):
        return gender
    return ""


def _normalize_bpm(value: Any, fallback: Optional[int]) -> Optional[int]:
    if value is None:
        return fallback
    try:
        bpm = int(float(str(value).strip()))
    except ValueError:
        return fallback
    bpm = max(60, min(180, bpm))
    return bpm


def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, num))


def _clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, num))


def _normalize_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return fallback


def _normalize_output_mode(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    mode = str(value).strip().lower()
    if mode in OUTPUT_MODES:
        return mode
    return fallback


def _build_messages(req: RemixRequest) -> List[Dict[str, str]]:
    instructions = [
        "You are an expert music producer and A&R.",
        "Return ONLY valid JSON.",
        "Schema: {\"title\":\"...\",\"genre\":\"...\",\"moods\":[...],\"timbres\":[...],\"instruments\":[...],\"bpm\":120,\"custom_style\":\"...\",\"gender\":\"auto\",\"advanced\":{\"cfg_coef\":1.6,\"temperature\":0.8,\"top_k\":50,\"top_p\":0.9,\"extend_stride\":5,\"use_genre_presets\":true,\"num_candidates\":2,\"auto_select_best\":true,\"output_mode\":\"mixed\"}}",
        "genre must be a single primary genre string (not a list).",
        "moods/timbres/instruments should be 2-6 concise items.",
        "bpm must be an integer between 60 and 180.",
        "gender must be male, female, or auto.",
        "output_mode must be mixed, vocal, bgm, or separate.",
        "Use ASCII punctuation only.",
    ]

    user_parts = [
        f"PROMPT: {req.prompt}",
    ]
    if req.title:
        user_parts.append(f"CURRENT_TITLE: {req.title}")
    if req.language:
        user_parts.append(f"LANGUAGE: {req.language}")
    if req.genre:
        user_parts.append(f"CURRENT_GENRE: {req.genre}")
    if req.moods:
        user_parts.append(f"CURRENT_MOODS: {', '.join(req.moods)}")
    if req.timbres:
        user_parts.append(f"CURRENT_TIMBRES: {', '.join(req.timbres)}")
    if req.instruments:
        user_parts.append(f"CURRENT_INSTRUMENTS: {', '.join(req.instruments)}")
    if req.bpm:
        user_parts.append(f"CURRENT_BPM: {req.bpm}")
    if req.gender:
        user_parts.append(f"CURRENT_GENDER: {req.gender}")
    if req.custom_style:
        user_parts.append(f"CURRENT_CUSTOM_STYLE: {req.custom_style}")
    if req.output_mode:
        user_parts.append(f"CURRENT_OUTPUT_MODE: {req.output_mode}")
    if req.advanced:
        user_parts.append(f"CURRENT_ADVANCED: {json.dumps(req.advanced)}")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _call_lmstudio(req: RemixRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["lmstudio"]).rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
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


def _call_ollama(req: RemixRequest, messages: List[Dict[str, str]]) -> str:
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


def _call_provider(provider: str, req: RemixRequest, messages: List[Dict[str, str]]) -> str:
    if provider == "ollama":
        return _call_ollama(req, messages)
    return _call_lmstudio(req, messages)


def _build_style_description(result: Dict[str, Any]) -> str:
    parts = []
    gender = result.get("gender")
    if gender and gender != "auto":
        parts.append(gender)
    for key in ("timbres", "genre", "moods", "instruments", "custom_style"):
        value = result.get(key)
        if isinstance(value, list) and value:
            parts.append(", ".join(value))
        elif isinstance(value, str) and value:
            parts.append(value)
    bpm = result.get("bpm")
    if bpm:
        parts.append(f"the bpm is {bpm}")
    return ", ".join([p for p in parts if p])


def _section_has_lyrics(section_type: str) -> bool:
    base = (section_type or "").split("-")[0].lower()
    return base in VOCAL_SECTION_TYPES


def generate_remix(req: RemixRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    if not req.song_model:
        raise ValueError("song_model is required")
    if not req.prompt or not req.prompt.strip():
        raise ValueError("prompt is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    messages = _build_messages(req)
    raw_text = _call_provider(provider, req, messages)
    parsed = _extract_json_safe(raw_text)
    if parsed is None:
        retry_messages = messages + [{"role": "user", "content": "Return ONLY JSON for the schema."}]
        raw_text = _call_provider(provider, req, retry_messages)
        parsed = _extract_json_safe(raw_text)
    parsed = parsed or {}

    genre = _normalize_str(parsed.get("genre")) or _normalize_str(req.genre)
    moods = _normalize_list(parsed.get("moods")) or list(req.moods or [])
    timbres = _normalize_list(parsed.get("timbres")) or list(req.timbres or [])
    instruments = _normalize_list(parsed.get("instruments")) or list(req.instruments or [])
    custom_style = _normalize_str(parsed.get("custom_style")) or _normalize_str(req.custom_style)
    gender = _normalize_gender(parsed.get("gender")) or _normalize_gender(req.gender)
    bpm = _normalize_bpm(parsed.get("bpm"), req.bpm)

    advanced = parsed.get("advanced") or {}
    if not isinstance(advanced, dict):
        advanced = {}
    base_adv = req.advanced or {}
    cfg_coef = _clamp_float(advanced.get("cfg_coef"), 0.1, 5.0, float(base_adv.get("cfg_coef", 1.5)))
    temperature = _clamp_float(advanced.get("temperature"), 0.1, 2.0, float(base_adv.get("temperature", 0.8)))
    top_k = _clamp_int(advanced.get("top_k"), 1, 250, int(base_adv.get("top_k", 50)))
    top_p = _clamp_float(advanced.get("top_p"), 0.0, 1.0, float(base_adv.get("top_p", 0.0)))
    extend_stride = _clamp_int(advanced.get("extend_stride"), 1, 10, int(base_adv.get("extend_stride", 5)))
    use_genre_presets = _normalize_bool(advanced.get("use_genre_presets"), bool(base_adv.get("use_genre_presets", True)))
    num_candidates = _clamp_int(advanced.get("num_candidates"), 1, 5, int(base_adv.get("num_candidates", 1)))
    auto_select_best = _normalize_bool(advanced.get("auto_select_best"), bool(base_adv.get("auto_select_best", True)))
    output_mode = _normalize_output_mode(advanced.get("output_mode"), req.output_mode or "mixed")

    result = {
        "title": _normalize_str(parsed.get("title")) or req.title or "Untitled",
        "genre": genre,
        "moods": moods,
        "timbres": timbres,
        "instruments": instruments,
        "custom_style": custom_style,
        "gender": gender,
        "bpm": bpm,
        "advanced": {
            "cfg_coef": cfg_coef,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "extend_stride": extend_stride,
            "use_genre_presets": use_genre_presets,
            "num_candidates": num_candidates,
            "auto_select_best": auto_select_best,
            "output_mode": output_mode,
        },
    }

    style_description = _build_style_description(result)

    lyrics_sections = [
        LyricsSection(
            type=sec.type,
            has_lyrics=_section_has_lyrics(sec.type),
            lyrics=sec.lyrics or "",
        )
        for sec in req.sections
    ]

    max_tokens_by_length = {"short": 800, "medium": 1400, "full": 2000}
    max_tokens = max_tokens_by_length.get(req.length or "full", 1400)

    lyrics_req = LyricsRequest(
        provider=req.provider,
        model=req.model,
        base_url=req.base_url,
        seed_words=req.prompt,
        sections=lyrics_sections,
        mode="refine",
        style=style_description,
        language=req.language,
        length=req.length or "full",
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    lyrics_result = generate_lyrics(lyrics_req)
    updated_sections = lyrics_result.get("sections") or [s.model_dump() for s in lyrics_sections]

    payload = {
        "title": result["title"],
        "sections": [
            {"type": s.get("type"), "lyrics": s.get("lyrics")}
            for s in updated_sections
        ],
        "gender": gender or "female",
        "genre": genre,
        "emotion": ", ".join(moods),
        "timbre": ", ".join(timbres),
        "instruments": ", ".join(instruments),
        "custom_style": custom_style,
        "bpm": bpm or 120,
        "output_mode": output_mode,
        "model": req.song_model,
        "reference_audio_id": req.reference_audio_id,
        "cfg_coef": cfg_coef,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "extend_stride": extend_stride,
        "use_genre_presets": use_genre_presets,
        "num_candidates": num_candidates,
        "auto_select_best": auto_select_best,
        "arrangement_template": req.arrangement_template,
    }

    return {
        "payload": payload,
        "style": result,
        "raw_text": raw_text,
        "lyrics_raw_text": lyrics_result.get("raw_text"),
    }
