"""Structure generation via local LLMs (LM Studio / Ollama)."""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any

import requests

from schemas import StructureRequest


DEFAULT_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
}

ALLOWED_SECTION_TYPES = {
    "intro-short",
    "intro-medium",
    "outro-short",
    "outro-medium",
    "inst-short",
    "inst-medium",
    "verse",
    "chorus",
    "bridge",
    "prechorus",
}

LENGTH_RANGES = {
    "short": (4, 7),
    "medium": (6, 9),
    "full": (8, 12),
}

DEFAULT_STRUCTURES = {
    "short": ["intro-short", "verse", "chorus", "outro-short"],
    "medium": ["intro-short", "verse", "chorus", "verse", "chorus", "outro-short"],
    "full": ["intro-short", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro-medium"],
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


def _normalize_label(value: str, length_key: str) -> str:
    raw = value.strip().lower()
    raw = raw.strip("[]")
    raw = re.sub(r"^[\d\.\)\-_\s]+", "", raw)
    raw = raw.replace("pre-chorus", "prechorus").replace("pre chorus", "prechorus")
    raw = raw.replace("instrumental", "inst")
    raw = raw.replace(" ", "-")

    if raw.endswith("-long"):
        raw = raw[:-5] + "medium"

    if raw in {"intro", "outro", "inst"}:
        duration = "medium" if length_key == "full" else "short"
        raw = f"{raw}-{duration}"

    if raw.startswith("prechorus-"):
        raw = "prechorus"

    if raw in {"verse", "chorus", "bridge", "prechorus"}:
        return raw

    if raw in ALLOWED_SECTION_TYPES:
        return raw

    return ""


def _normalize_sections(values: Any, length_key: str) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = re.split(r"[,|\n]", values)
    elif isinstance(values, list):
        parts = values
    else:
        parts = [values]

    sections: List[str] = []
    for item in parts:
        label = _normalize_label(str(item), length_key)
        if label:
            sections.append(label)
    return sections


def _trim_sections(sections: List[str], max_count: int) -> List[str]:
    if len(sections) <= max_count:
        return sections
    outro = sections[-1] if sections[-1].startswith("outro") else ""
    trimmed = sections[:max_count]
    if outro and trimmed[-1] != outro:
        trimmed = trimmed[:-1] + [outro]
    return trimmed


def _ensure_min_sections(sections: List[str], min_count: int) -> List[str]:
    if not sections:
        return sections
    insert_idx = len(sections)
    if sections and sections[-1].startswith("outro"):
        insert_idx = len(sections) - 1
    toggle = 0
    while len(sections) < min_count:
        sections.insert(insert_idx, "verse" if toggle % 2 == 0 else "chorus")
        insert_idx += 1
        toggle += 1
    return sections


def _ensure_core_sections(sections: List[str]) -> bool:
    has_verse = any(item == "verse" for item in sections)
    has_chorus = any(item == "chorus" for item in sections)
    return has_verse and has_chorus


def _normalize_length_key(value: str | None) -> str:
    key = (value or "full").strip().lower()
    if key not in LENGTH_RANGES:
        return "full"
    return key


def _build_messages(req: StructureRequest) -> List[Dict[str, str]]:
    length_key = _normalize_length_key(req.length)
    min_count, max_count = LENGTH_RANGES[length_key]
    allowed = ", ".join(sorted(ALLOWED_SECTION_TYPES))

    instructions = [
        "You are an expert music producer and songwriter.",
        "Return ONLY valid JSON.",
        "Schema: {\"sections\":[\"intro-short\",\"verse\",\"chorus\",...]}",
        f"Use ONLY these section labels: {allowed}.",
        "Use durations only for intro/outro/inst (short or medium).",
        "Avoid inst sections unless absolutely necessary.",
        "Always include at least one verse and one chorus.",
        f"Target {min_count}-{max_count} sections for length={length_key}.",
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
    if req.style:
        user_parts.append(f"STYLE: {req.style}")
    user_parts.append(f"LENGTH: {length_key}")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts) if user_parts else "Generate a hit-ready song structure."},
    ]


def _call_lmstudio(req: StructureRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["lmstudio"]).rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    payload = {
        "model": req.model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 400,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(req: StructureRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["ollama"]).rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": req.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9,
            "num_predict": 400,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def _call_provider(provider: str, req: StructureRequest, messages: List[Dict[str, str]]) -> str:
    if provider == "ollama":
        return _call_ollama(req, messages)
    return _call_lmstudio(req, messages)


def generate_structure(req: StructureRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    length_key = _normalize_length_key(req.length)
    messages = _build_messages(req)
    raw_text = _call_provider(provider, req, messages)
    parsed = _extract_json_safe(raw_text)
    if parsed is None:
        retry_messages = messages + [{"role": "user", "content": "Return ONLY JSON for the schema."}]
        raw_text = _call_provider(provider, req, retry_messages)
        parsed = _extract_json_safe(raw_text)

    sections_raw = []
    if isinstance(parsed, dict):
        sections_raw = parsed.get("sections") or parsed.get("structure") or []
    elif isinstance(parsed, list):
        sections_raw = parsed

    sections = _normalize_sections(sections_raw, length_key)
    min_count, max_count = LENGTH_RANGES[length_key]
    sections = _trim_sections(sections, max_count)
    sections = _ensure_min_sections(sections, min_count)

    if not sections or not _ensure_core_sections(sections):
        sections = list(DEFAULT_STRUCTURES[length_key])

    return {
        "sections": sections,
        "raw_text": raw_text,
    }
