"""
LLM-assisted composition helpers (LM Studio / Ollama).
Generates genre, mood, structure, and lyrics in multiple steps for stability.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

import requests

from generation import _TAG_GENRES, _TAG_EMOTIONS, _TAG_TIMBRES, _TAG_INSTRUMENTS


_ALLOWED_SECTIONS = {
    "intro-short",
    "intro-medium",
    "verse",
    "chorus",
    "bridge",
    "outro-short",
    "outro-medium",
    "inst-short",
    "inst-medium",
}

_DEFAULT_STRUCTURES = {
    "short": ["intro-short", "verse", "chorus", "verse", "chorus", "outro-short"],
    "medium": ["intro-medium", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro-medium"],
    "full": ["intro-medium", "verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus", "outro-medium"],
}


def _normalize_base_url(provider: str, base_url: Optional[str]) -> str:
    if provider == "lmstudio":
        url = (base_url or "http://localhost:1234/v1").rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url
    if provider == "ollama":
        return (base_url or "http://localhost:11434").rstrip("/")
    raise ValueError(f"Unknown provider: {provider}")


def _call_chat(provider: str, base_url: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    if provider == "lmstudio":
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    if provider == "ollama":
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response.")
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start:end + 1])


def _coerce_tag(value: Any, allowed: set[str], fallback: Optional[str] = None) -> str:
    if isinstance(value, list):
        for item in value:
            candidate = str(item).strip().lower()
            if candidate in allowed:
                return candidate
    elif value is not None:
        candidate = str(value).strip().lower()
        if candidate in allowed:
            return candidate
        for tag in allowed:
            if tag in candidate:
                return tag
    if fallback:
        return fallback
    if allowed:
        return random.choice(sorted(allowed))
    return ""


def _coerce_tags(value: Any, allowed: set[str], max_items: int = 2) -> List[str]:
    if isinstance(value, list):
        items = [str(v).strip().lower() for v in value]
    elif value is not None:
        items = [v.strip().lower() for v in re.split(r"[;,]", str(value)) if v.strip()]
    else:
        items = []
    picks = []
    for item in items:
        if item in allowed and item not in picks:
            picks.append(item)
        if len(picks) >= max_items:
            break
    if not picks and allowed:
        picks.append(random.choice(sorted(allowed)))
    return picks


def _coerce_bpm(value: Any, fallback: int = 94) -> int:
    if isinstance(value, (int, float)):
        return max(60, min(180, int(value)))
    if isinstance(value, str):
        match = re.search(r"\d{2,3}", value)
        if match:
            return max(60, min(180, int(match.group(0))))
    return fallback


def _coerce_gender(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in ("male", "female"):
        return candidate
    return "female"


def _build_section_prompt(length: str) -> str:
    if length == "short":
        return "Write 2-3 sentences."
    if length == "full":
        return "Write 6-8 sentences."
    return "Write 4-6 sentences."


def _structure_fallback(length: str) -> List[str]:
    return _DEFAULT_STRUCTURES.get(length, _DEFAULT_STRUCTURES["medium"])


def generate_ai_assist(request: Dict[str, Any]) -> Dict[str, Any]:
    provider = (request.get("provider") or "lmstudio").strip().lower()
    base_url = _normalize_base_url(provider, request.get("base_url"))
    model = (request.get("model") or "").strip()
    if not model:
        raise ValueError("Model is required for AI assist.")
    prompt = (request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required for AI assist.")
    language = (request.get("language") or "English").strip()
    length = (request.get("length") or "medium").strip().lower()
    if length not in _DEFAULT_STRUCTURES:
        length = "medium"

    allowed_genres = sorted(_TAG_GENRES)
    allowed_emotions = sorted(_TAG_EMOTIONS)
    allowed_timbres = sorted(_TAG_TIMBRES)
    allowed_instruments = sorted(_TAG_INSTRUMENTS)

    base_context = (
        f"Theme: {prompt}\n"
        f"Language: {language}\n"
        f"Length: {length}\n"
        "Use only the provided tags when possible."
    )

    def ask_json(task: str, guidance: str, temperature: float = 0.7, max_tokens: int = 180) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Return ONLY a JSON object. No extra text."},
            {"role": "user", "content": f"{base_context}\nTask: {task}\n{guidance}"},
        ]
        content = _call_chat(provider, base_url, model, messages, temperature, max_tokens)
        return _extract_json(content)

    # Step 1: genre
    genre_json = ask_json(
        "Pick one genre tag.",
        f"Allowed genre tags: {', '.join(allowed_genres)}\nJSON: {{\"genre\": \"<tag>\"}}",
    )
    genre = _coerce_tag(genre_json.get("genre"), _TAG_GENRES)

    # Step 2: mood / emotion
    mood_json = ask_json(
        "Pick 1-2 mood/emotion tags that fit the genre and theme.",
        f"Allowed mood tags: {', '.join(allowed_emotions)}\nJSON: {{\"emotion\": [\"<tag>\"]}}",
    )
    emotion = _coerce_tags(mood_json.get("emotion"), _TAG_EMOTIONS, max_items=2)

    # Step 3: timbre
    timbre_json = ask_json(
        "Pick 1-2 timbre tags that match the mood and genre.",
        f"Allowed timbre tags: {', '.join(allowed_timbres)}\nJSON: {{\"timbre\": [\"<tag>\"]}}",
    )
    timbre = _coerce_tags(timbre_json.get("timbre"), _TAG_TIMBRES, max_items=2)

    # Step 4: instruments
    inst_json = ask_json(
        "Pick 1-2 instrument tags that match the genre.",
        f"Allowed instrument tags: {', '.join(allowed_instruments)}\nJSON: {{\"instruments\": [\"<tag>\"]}}",
    )
    instruments = _coerce_tags(inst_json.get("instruments"), _TAG_INSTRUMENTS, max_items=2)

    # Step 5: bpm
    bpm_json = ask_json(
        "Pick a BPM (60-180) that fits the genre and mood.",
        "JSON: {\"bpm\": 94}",
    )
    bpm = _coerce_bpm(bpm_json.get("bpm"), fallback=94)

    # Step 6: gender
    gender_json = ask_json(
        "Pick a vocal gender (male or female) that fits the theme.",
        "JSON: {\"gender\": \"female\"}",
    )
    gender = _coerce_gender(gender_json.get("gender"))

    # Step 7: structure
    structure_guidance = (
        "Return a list of section types using only these labels: "
        "intro-short, intro-medium, verse, chorus, bridge, outro-short, outro-medium. "
        "Avoid inst labels unless absolutely needed. Include at least 2 verses and 2 choruses.\n"
        "JSON: {\"structure\": [\"intro-medium\", \"verse\", \"chorus\", \"verse\", \"chorus\", \"outro-medium\"]}"
    )
    try:
        structure_json = ask_json("Create a song structure.", structure_guidance, temperature=0.5, max_tokens=200)
        structure_raw = structure_json.get("structure", [])
        if not isinstance(structure_raw, list):
            structure_raw = []
        structure = [s.strip().lower() for s in structure_raw if isinstance(s, str)]
        structure = [s for s in structure if s in _ALLOWED_SECTIONS]
        if not structure:
            structure = _structure_fallback(length)
    except Exception:
        structure = _structure_fallback(length)

    # Step 8: lyrics per section (incremental)
    sections: List[Dict[str, Any]] = []
    lyric_context = []
    style_line = ", ".join(filter(None, [
        gender,
        ", ".join(timbre) if timbre else "",
        genre,
        ", ".join(emotion) if emotion else "",
        ", ".join(instruments) if instruments else "",
        f"the bpm is {bpm}",
    ]))

    for idx, section_type in enumerate(structure, start=1):
        base = section_type.split("-", 1)[0]
        if base not in ("verse", "chorus", "bridge"):
            sections.append({"type": section_type, "lyrics": ""})
            continue

        section_prompt = (
            f"Write lyrics for {section_type} {idx}.\n"
            f"Style tags: {style_line}\n"
            f"Structure so far: {' | '.join(structure)}\n"
            f"Previous lyrics: {' '.join(lyric_context[-3:]) if lyric_context else 'None'}\n"
            f"{_build_section_prompt(length)}\n"
            "Use complete sentences separated by periods. Do not add section labels.\n"
            "JSON: {\"lyrics\": \"Sentence one. Sentence two.\"}"
        )
        lyric_json = ask_json("Write section lyrics.", section_prompt, temperature=0.8, max_tokens=320)
        lyrics_text = str(lyric_json.get("lyrics", "")).strip()
        lyric_context.append(lyrics_text)
        sections.append({"type": section_type, "lyrics": lyrics_text})

    return {
        "genre": genre,
        "emotion": emotion,
        "timbre": timbre,
        "instruments": instruments,
        "bpm": bpm,
        "gender": gender,
        "custom_style": "",
        "structure": structure,
        "sections": sections,
    }
