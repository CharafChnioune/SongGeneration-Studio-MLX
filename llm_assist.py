"""
LLM-assisted composition helpers (LM Studio / Ollama).
Generates genre, mood, structure, and lyrics in multiple steps for stability.
"""

from __future__ import annotations

import json
import random
import re
import ast
from typing import Any, Dict, List, Optional

import requests

from generation import _TAG_GENRES, _TAG_EMOTIONS, _TAG_TIMBRES, _TAG_INSTRUMENTS


_ALLOWED_SECTIONS = {
    "verse",
    "chorus",
    "bridge",
}

_DEFAULT_STRUCTURES = {
    "short": ["verse", "chorus", "verse", "chorus", "bridge"],
    "medium": ["verse", "chorus", "verse", "chorus", "bridge", "chorus", "verse"],
    "full": ["verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus", "verse", "chorus"],
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
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    candidate = cleaned
    if not (candidate.startswith("{") and candidate.endswith("}")):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        value = ast.literal_eval(candidate)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    raise ValueError("No JSON object found in LLM response.")


def _ask_json(
    provider: str,
    base_url: str,
    model: str,
    context: str,
    task: str,
    guidance: str,
    temperature: float = 0.7,
    max_tokens: int = 180,
) -> Dict[str, Any]:
    strict = "Return ONLY a JSON object. No extra text, no markdown, no code fences."
    system_prompt = f"{strict}\nContext:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task}\n{guidance}"},
    ]
    for attempt in range(2):
        content = _call_chat(provider, base_url, model, messages, temperature, max_tokens)
        try:
            return _extract_json(content)
        except Exception:
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task}\n{guidance}\nReturn ONLY JSON. If unsure, return an empty JSON object."},
                ]
                continue
            return {}
    return {}


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


def _sentence_target(section_type: str, length: str) -> tuple[int, int]:
    base = section_type.split("-", 1)[0]
    if length == "short":
        if base == "chorus":
            return 4, 5
        if base == "bridge":
            return 3, 4
        return 5, 6
    if length == "full":
        if base == "chorus":
            return 6, 8
        if base == "bridge":
            return 5, 6
        return 8, 10
    # medium
    if base == "chorus":
        return 5, 6
    if base == "bridge":
        return 4, 5
    return 6, 8


def _build_section_prompt(section_type: str, length: str) -> str:
    low, high = _sentence_target(section_type, length)
    base = section_type.split("-", 1)[0]
    if base == "chorus":
        extra = "Make it catchy with a short hook phrase that repeats once."
    elif base == "bridge":
        extra = "Shift the imagery or perspective, but keep it cohesive."
    else:
        extra = "Tell the story with vivid imagery and forward momentum."
    return f"Write {low}-{high} sentences. {extra}"


def _count_sentences(text: str) -> int:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return 0
    parts = re.split(r"[.!?]+", cleaned)
    return len([p for p in parts if p.strip()])


def _selection_context(
    prompt: str,
    language: str,
    length: str,
    genre: str = "",
    emotion: Optional[List[str]] = None,
    timbre: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    bpm: Optional[int] = None,
    gender: str = "",
    structure: Optional[List[str]] = None,
    lyrics: Optional[List[str]] = None,
) -> str:
    lines = [
        f"Theme: {prompt}",
        f"Language: {language}",
        f"Length: {length}",
        "Goal: chart-ready hit song with rich imagery and a memorable hook.",
        "Only use verse/chorus/bridge sections. Skip intro/outro/inst sections (they are instrumental).",
    ]
    if genre:
        lines.append(f"Selected genre: {genre}")
    if emotion:
        lines.append(f"Selected mood: {', '.join(emotion)}")
    if timbre:
        lines.append(f"Selected timbre: {', '.join(timbre)}")
    if instruments:
        lines.append(f"Selected instruments: {', '.join(instruments)}")
    if bpm:
        lines.append(f"Selected BPM: {bpm}")
    if gender:
        lines.append(f"Selected gender: {gender}")
    if structure:
        lines.append(f"Structure: {' | '.join(structure)}")
    if lyrics:
        joined = " ".join([l.strip() for l in lyrics if l.strip()])
        if joined:
            lines.append(f"Lyrics so far: {joined}")
    lines.append("Use only the provided tags when possible.")
    return "\n".join(lines)


def _normalize_structure(structure: List[str], length: str) -> List[str]:
    cleaned = [s for s in structure if s in _ALLOWED_SECTIONS]
    if not cleaned:
        cleaned = _structure_fallback(length)
    # Ensure starts with verse
    if cleaned and cleaned[0] != "verse":
        cleaned.insert(0, "verse")
    # Ensure at least 2 verses and 2 choruses
    while cleaned.count("chorus") < 2:
        cleaned.append("chorus")
    while cleaned.count("verse") < 2:
        cleaned.append("verse")
    # Enforce length ranges
    min_len, max_len = (5, 6) if length == "short" else (7, 8) if length == "medium" else (9, 10)
    while len(cleaned) < min_len:
        cleaned.append("chorus" if cleaned[-1] != "chorus" else "verse")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def _structure_fallback(length: str) -> List[str]:
    return _DEFAULT_STRUCTURES.get(length, _DEFAULT_STRUCTURES["medium"])


def _generate_section_lyrics(
    provider: str,
    base_url: str,
    model: str,
    prompt: str,
    language: str,
    length: str,
    genre: str,
    emotion: List[str],
    timbre: List[str],
    instruments: List[str],
    bpm: int,
    gender: str,
    structure: List[str],
    lyric_context: List[str],
    section_type: str,
    idx: int,
) -> str:
    style_line = ", ".join(filter(None, [
        gender,
        ", ".join(timbre) if timbre else "",
        genre,
        ", ".join(emotion) if emotion else "",
        ", ".join(instruments) if instruments else "",
        f"the bpm is {bpm}",
    ]))
    section_prompt = (
        f"Write lyrics for {section_type} {idx}.\n"
        f"Style tags: {style_line}\n"
        f"Structure: {' | '.join(structure)}\n"
        f"Previous lyrics: {' '.join(lyric_context[-3:]) if lyric_context else 'None'}\n"
        f"{_build_section_prompt(section_type, length)}\n"
        "Avoid mentioning instruments or production. Avoid repeating the prompt verbatim.\n"
        "Continue the song naturally after the lyrics so far. Do not repeat previous lines.\n"
        "Use complete sentences separated by periods. Do not add section labels.\n"
        "JSON: {\"lyrics\": \"Sentence one. Sentence two.\"}"
    )
    lyric_context_text = _selection_context(
        prompt,
        language,
        length,
        genre=genre,
        emotion=emotion,
        timbre=timbre,
        instruments=instruments,
        bpm=bpm,
        gender=gender,
        structure=structure,
        lyrics=lyric_context,
    ) + f"\nSection: {section_type}"
    lyric_json = _ask_json(provider, base_url, model, lyric_context_text, "Write section lyrics.", section_prompt, temperature=0.8, max_tokens=450)
    lyrics_text = str(lyric_json.get("lyrics", "")).strip()
    min_sentences, _ = _sentence_target(section_type, length)
    if _count_sentences(lyrics_text) < min_sentences or len(lyrics_text) < 120:
        expand_prompt = (
            f"Expand these lyrics by adding {max(2, min_sentences)} more sentences.\n"
            f"Current lyrics: {lyrics_text}\n"
            "Append new sentences at the end. Keep the same theme and voice.\n"
            "JSON: {\"lyrics\": \"<expanded full lyrics>\"}"
        )
        lyric_json = _ask_json(provider, base_url, model, lyric_context_text, "Expand section lyrics.", expand_prompt, temperature=0.85, max_tokens=600)
        lyrics_text = str(lyric_json.get("lyrics", "")).strip()
    if _count_sentences(lyrics_text) < min_sentences or len(lyrics_text) < 120:
        expand_prompt = (
            f"Make the lyrics longer and richer. Ensure at least {min_sentences} sentences.\n"
            f"Current lyrics: {lyrics_text}\n"
            "Return the full expanded lyrics.\n"
            "JSON: {\"lyrics\": \"<expanded full lyrics>\"}"
        )
        lyric_json = _ask_json(provider, base_url, model, lyric_context_text, "Expand section lyrics.", expand_prompt, temperature=0.9, max_tokens=700)
        lyrics_text = str(lyric_json.get("lyrics", "")).strip()
    if _count_sentences(lyrics_text) < min_sentences:
        seed = prompt or "We rise tonight"
        filler = "We chase the night. We hold the line. We turn the pain to gold."
        lyrics_text = f"{seed}. {filler}"
    return lyrics_text


def _edit_section_lyrics(
    provider: str,
    base_url: str,
    model: str,
    prompt: str,
    language: str,
    length: str,
    genre: str,
    emotion: List[str],
    timbre: List[str],
    instruments: List[str],
    bpm: int,
    gender: str,
    structure: List[str],
    lyric_context: List[str],
    section_type: str,
    idx: int,
    instruction: str,
    current_lyrics: str,
) -> str:
    min_sentences, _ = _sentence_target(section_type, length)
    base_prompt = (
        f"Rewrite the lyrics for {section_type} {idx}.\n"
        f"Instruction: {instruction}\n"
        f"Current lyrics: {current_lyrics}\n"
        f"Structure: {' | '.join(structure)}\n"
        f"Previous lyrics: {' '.join(lyric_context[-3:]) if lyric_context else 'None'}\n"
        f"{_build_section_prompt(section_type, length)}\n"
        "Avoid mentioning instruments or production. Keep it cohesive with the song so far.\n"
        "Use complete sentences separated by periods. Do not add section labels.\n"
        "JSON: {\"lyrics\": \"<rewritten lyrics>\"}"
    )
    context = _selection_context(
        prompt,
        language,
        length,
        genre=genre,
        emotion=emotion,
        timbre=timbre,
        instruments=instruments,
        bpm=bpm,
        gender=gender,
        structure=structure,
        lyrics=lyric_context,
    ) + f"\nSection: {section_type}"
    lyric_json = _ask_json(provider, base_url, model, context, "Rewrite section lyrics.", base_prompt, temperature=0.8, max_tokens=500)
    lyrics_text = str(lyric_json.get("lyrics", "")).strip()
    if _count_sentences(lyrics_text) < min_sentences or len(lyrics_text) < 120:
        expand_prompt = (
            f"Make the rewrite longer and richer. Ensure at least {min_sentences} sentences.\n"
            f"Current rewrite: {lyrics_text}\n"
            "Return the full expanded lyrics.\n"
            "JSON: {\"lyrics\": \"<expanded full lyrics>\"}"
        )
        lyric_json = _ask_json(provider, base_url, model, context, "Expand section rewrite.", expand_prompt, temperature=0.85, max_tokens=650)
        lyrics_text = str(lyric_json.get("lyrics", "")).strip()
    if _count_sentences(lyrics_text) < min_sentences:
        seed = prompt or "We rise tonight"
        filler = "We chase the night. We hold the line. We turn the pain to gold."
        lyrics_text = f"{seed}. {filler}"
    return lyrics_text


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

    base_context = _selection_context(prompt, language, length)

    # Step 1: genre
    genre_json = _ask_json(
        provider,
        base_url,
        model,
        base_context,
        "Pick one genre tag.",
        f"Allowed genre tags: {', '.join(allowed_genres)}\nJSON: {{\"genre\": \"<tag>\"}}",
    )
    genre = _coerce_tag(genre_json.get("genre"), _TAG_GENRES)

    # Step 2: mood / emotion
    mood_context = _selection_context(prompt, language, length, genre=genre)
    mood_json = _ask_json(
        provider,
        base_url,
        model,
        mood_context,
        "Pick 1-2 mood/emotion tags that fit the genre and theme.",
        f"Allowed mood tags: {', '.join(allowed_emotions)}\nJSON: {{\"emotion\": [\"<tag>\"]}}",
    )
    emotion = _coerce_tags(mood_json.get("emotion"), _TAG_EMOTIONS, max_items=2)

    # Step 3: timbre
    timbre_context = _selection_context(prompt, language, length, genre=genre, emotion=emotion)
    timbre_json = _ask_json(
        provider,
        base_url,
        model,
        timbre_context,
        "Pick 1-2 timbre tags that match the mood and genre.",
        f"Allowed timbre tags: {', '.join(allowed_timbres)}\nJSON: {{\"timbre\": [\"<tag>\"]}}",
    )
    timbre = _coerce_tags(timbre_json.get("timbre"), _TAG_TIMBRES, max_items=2)

    # Step 4: instruments
    inst_context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre)
    inst_json = _ask_json(
        provider,
        base_url,
        model,
        inst_context,
        "Pick 1-2 instrument tags that match the genre.",
        f"Allowed instrument tags: {', '.join(allowed_instruments)}\nJSON: {{\"instruments\": [\"<tag>\"]}}",
    )
    instruments = _coerce_tags(inst_json.get("instruments"), _TAG_INSTRUMENTS, max_items=2)

    # Step 5: bpm
    bpm_context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments)
    bpm_json = _ask_json(
        provider,
        base_url,
        model,
        bpm_context,
        "Pick a BPM (60-180) that fits the genre and mood.",
        "JSON: {\"bpm\": 94}",
    )
    bpm = _coerce_bpm(bpm_json.get("bpm"), fallback=94)

    # Step 6: gender
    gender_context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments, bpm=bpm)
    gender_json = _ask_json(
        provider,
        base_url,
        model,
        gender_context,
        "Pick a vocal gender (male or female) that fits the theme.",
        "JSON: {\"gender\": \"female\"}",
    )
    gender = _coerce_gender(gender_json.get("gender"))

    # Step 7: structure
    structure_guidance = (
        "Return a list of section types using only: verse, chorus, bridge. "
        "Include at least 2 verses and 2 choruses. "
        "Length rules: short=5-6 sections, medium=7-8 sections, full=9-10 sections.\n"
        "JSON: {\"structure\": [\"verse\", \"chorus\", \"verse\", \"chorus\", \"bridge\", \"chorus\", \"verse\"]}"
    )
    try:
        structure_context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments, bpm=bpm, gender=gender)
        structure_json = _ask_json(provider, base_url, model, structure_context, "Create a song structure.", structure_guidance, temperature=0.5, max_tokens=200)
        structure_raw = structure_json.get("structure", [])
        if not isinstance(structure_raw, list):
            structure_raw = []
        structure = [s.strip().lower() for s in structure_raw if isinstance(s, str)]
        structure = _normalize_structure(structure, length)
    except Exception:
        structure = _structure_fallback(length)

    # Step 8: lyrics per section (incremental)
    sections: List[Dict[str, Any]] = []
    lyric_context: List[str] = []
    for idx, section_type in enumerate(structure, start=1):
        base = section_type.split("-", 1)[0]
        if base not in ("verse", "chorus", "bridge"):
            continue
        lyrics_text = _generate_section_lyrics(
            provider,
            base_url,
            model,
            prompt,
            language,
            length,
            genre,
            emotion,
            timbre,
            instruments,
            bpm,
            gender,
            structure,
            lyric_context,
            section_type,
            idx,
        )
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


def generate_ai_assist_step(request: Dict[str, Any]) -> Dict[str, Any]:
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

    state = request.get("state") or {}
    step = (request.get("step") or "").strip().lower()
    section_index = request.get("section_index")
    instruction = (request.get("instruction") or "").strip()

    genre = (state.get("genre") or "").strip().lower()
    emotion = state.get("emotion") or []
    timbre = state.get("timbre") or []
    instruments = state.get("instruments") or []
    bpm = state.get("bpm") or 94
    gender = state.get("gender") or "female"
    structure = state.get("structure") or []
    sections = state.get("sections") or []
    lyric_context = [s.get("lyrics", "") for s in sections if s.get("lyrics")]

    allowed_genres = sorted(_TAG_GENRES)
    allowed_emotions = sorted(_TAG_EMOTIONS)
    allowed_timbres = sorted(_TAG_TIMBRES)
    allowed_instruments = sorted(_TAG_INSTRUMENTS)

    if step == "genre":
        context = _selection_context(prompt, language, length)
        genre_json = _ask_json(provider, base_url, model, context, "Pick one genre tag.",
                              f"Allowed genre tags: {', '.join(allowed_genres)}\nJSON: {{\"genre\": \"<tag>\"}}")
        genre = _coerce_tag(genre_json.get("genre"), _TAG_GENRES)
    elif step == "emotion":
        context = _selection_context(prompt, language, length, genre=genre)
        mood_json = _ask_json(provider, base_url, model, context, "Pick 1-2 mood/emotion tags that fit the genre and theme.",
                              f"Allowed mood tags: {', '.join(allowed_emotions)}\nJSON: {{\"emotion\": [\"<tag>\"]}}")
        emotion = _coerce_tags(mood_json.get("emotion"), _TAG_EMOTIONS, max_items=2)
    elif step == "timbre":
        context = _selection_context(prompt, language, length, genre=genre, emotion=emotion)
        timbre_json = _ask_json(provider, base_url, model, context, "Pick 1-2 timbre tags that match the mood and genre.",
                                f"Allowed timbre tags: {', '.join(allowed_timbres)}\nJSON: {{\"timbre\": [\"<tag>\"]}}")
        timbre = _coerce_tags(timbre_json.get("timbre"), _TAG_TIMBRES, max_items=2)
    elif step == "instruments":
        context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre)
        inst_json = _ask_json(provider, base_url, model, context, "Pick 1-2 instrument tags that match the genre.",
                              f"Allowed instrument tags: {', '.join(allowed_instruments)}\nJSON: {{\"instruments\": [\"<tag>\"]}}")
        instruments = _coerce_tags(inst_json.get("instruments"), _TAG_INSTRUMENTS, max_items=2)
    elif step == "bpm":
        context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments)
        bpm_json = _ask_json(provider, base_url, model, context, "Pick a BPM (60-180) that fits the genre and mood.",
                             "JSON: {\"bpm\": 94}")
        bpm = _coerce_bpm(bpm_json.get("bpm"), fallback=94)
    elif step == "gender":
        context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments, bpm=bpm)
        gender_json = _ask_json(provider, base_url, model, context, "Pick a vocal gender (male or female) that fits the theme.",
                                "JSON: {\"gender\": \"female\"}")
        gender = _coerce_gender(gender_json.get("gender"))
    elif step == "structure":
        context = _selection_context(prompt, language, length, genre=genre, emotion=emotion, timbre=timbre, instruments=instruments, bpm=bpm, gender=gender)
        structure_guidance = (
            "Return a list of section types using only: verse, chorus, bridge. "
            "Include at least 2 verses and 2 choruses. "
            "Length rules: short=5-6 sections, medium=7-8 sections, full=9-10 sections.\n"
            "JSON: {\"structure\": [\"verse\", \"chorus\", \"verse\", \"chorus\", \"bridge\", \"chorus\", \"verse\"]}"
        )
        structure_json = _ask_json(provider, base_url, model, context, "Create a song structure.", structure_guidance, temperature=0.5, max_tokens=200)
        structure_raw = structure_json.get("structure", [])
        if not isinstance(structure_raw, list):
            structure_raw = []
        structure = [s.strip().lower() for s in structure_raw if isinstance(s, str)]
        structure = _normalize_structure(structure, length)
        if not sections:
            sections = [{"type": s, "lyrics": ""} for s in structure]
    elif step == "lyrics":
        if not structure:
            structure = _structure_fallback(length)
        structure = _normalize_structure(structure, length)
        if not sections:
            sections = [{"type": s, "lyrics": ""} for s in structure]
        if section_index is None:
            raise ValueError("section_index is required for lyrics step.")
        if section_index < 0 or section_index >= len(structure):
            raise ValueError("section_index out of range.")
        section_type = structure[section_index]
        lyrics_text = _generate_section_lyrics(
            provider,
            base_url,
            model,
            prompt,
            language,
            length,
            genre,
            emotion,
            timbre,
            instruments,
            bpm,
            gender,
            structure,
            lyric_context,
            section_type,
            section_index + 1,
        )
        sections[section_index]["type"] = section_type
        sections[section_index]["lyrics"] = lyrics_text
    elif step == "edit":
        if not structure:
            structure = _structure_fallback(length)
        structure = _normalize_structure(structure, length)
        if not sections:
            sections = [{"type": s, "lyrics": ""} for s in structure]
        if section_index is None:
            raise ValueError("section_index is required for edit step.")
        if section_index < 0 or section_index >= len(structure):
            raise ValueError("section_index out of range.")
        if not instruction:
            raise ValueError("instruction is required for edit step.")
        section_type = structure[section_index]
        current_lyrics = sections[section_index].get("lyrics", "")
        lyrics_text = _edit_section_lyrics(
            provider,
            base_url,
            model,
            prompt,
            language,
            length,
            genre,
            emotion,
            timbre,
            instruments,
            bpm,
            gender,
            structure,
            lyric_context,
            section_type,
            section_index + 1,
            instruction,
            current_lyrics,
        )
        sections[section_index]["type"] = section_type
        sections[section_index]["lyrics"] = lyrics_text
    else:
        raise ValueError(f"Unknown step: {step}")

    return {
        "state": {
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
    }
