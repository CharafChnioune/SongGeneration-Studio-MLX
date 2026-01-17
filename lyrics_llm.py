"""Lyrics generation via local LLMs (LM Studio / Ollama)."""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any

import requests

from schemas import LyricsRequest, LyricsSection, LyricsModelsRequest


DEFAULT_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
}

LINE_TARGETS = {
    "short": {
        "intro": (1, 2),
        "outro": (1, 2),
        "verse": (2, 4),
        "chorus": (2, 4),
        "prechorus": (2, 4),
        "bridge": (2, 4),
        "default": (2, 4),
    },
    "medium": {
        "intro": (2, 4),
        "outro": (2, 4),
        "verse": (6, 8),
        "chorus": (4, 6),
        "prechorus": (4, 6),
        "bridge": (4, 6),
        "default": (4, 6),
    },
    "full": {
        "intro": (4, 6),
        "outro": (4, 6),
        "verse": (10, 14),
        "chorus": (8, 12),
        "prechorus": (6, 8),
        "bridge": (6, 8),
        "default": (6, 10),
    },
}


def _line_targets_for(req: LyricsRequest) -> Dict[str, tuple[int, int]]:
    key = (req.length or "full").strip().lower()
    return LINE_TARGETS.get(key, LINE_TARGETS["full"])


def _build_sections_payload(sections: List[LyricsSection]) -> List[Dict[str, Any]]:
    payload = []
    for sec in sections:
        payload.append({
            "type": sec.type,
            "has_lyrics": bool(sec.has_lyrics),
            "lyrics": sec.lyrics or "",
        })
    return payload


def _build_messages(req: LyricsRequest) -> List[Dict[str, str]]:
    section_lines = []
    for idx, sec in enumerate(req.sections, start=1):
        section_lines.append(f"{idx}. {sec.type} (lyrics: {str(bool(sec.has_lyrics)).lower()})")

    targets = _line_targets_for(req)
    target_lines = [
        f"- intro: {targets['intro'][0]}-{targets['intro'][1]} lines",
        f"- outro: {targets['outro'][0]}-{targets['outro'][1]} lines",
        f"- verse: {targets['verse'][0]}-{targets['verse'][1]} lines",
        f"- chorus: {targets['chorus'][0]}-{targets['chorus'][1]} lines",
        f"- prechorus: {targets['prechorus'][0]}-{targets['prechorus'][1]} lines",
        f"- bridge: {targets['bridge'][0]}-{targets['bridge'][1]} lines",
        f"- default: {targets['default'][0]}-{targets['default'][1]} lines",
    ]

    instructions = [
        "You are a professional songwriter.",
        "Return ONLY valid JSON.",
        "Schema: {\"sections\":[{\"type\":\"<type>\",\"lines\":[\"line1\",\"line2\"]}]}",
        "Keep the same number and order of sections as provided.",
        "For instrumental sections (lyrics: false), return an empty lines array.",
        "For sections with lyrics: true, ALWAYS return at least one line (never empty).",
        "Intro/outro are vocal sections when lyrics: true; always fill them.",
        "Do not add extra keys, notes, or code fences.",
        "Make it a complete, full song; no placeholders.",
        "Prefer the upper end of the target line counts to make lyrics longer.",
        "Keep lines short and natural. Follow the target line counts.",
        "Make choruses hooky and repeat a key phrase within each chorus.",
        "Use ASCII punctuation only.",
    ]

    user_parts = [
        "SECTIONS:",
        "\n".join(section_lines),
        "LINE_TARGETS:",
        "\n".join(target_lines),
    ]

    if req.language:
        user_parts.append(f"LANGUAGE: {req.language}")
    if req.style:
        user_parts.append(f"STYLE: {req.style}")
    if req.seed_words:
        user_parts.append(f"SEED_WORDS: {req.seed_words}")

    if req.mode == "refine":
        user_parts.append("MODE: refine existing lyrics while keeping structure, expanding to meet line targets if short.")
        user_parts.append("EXISTING_LYRICS_JSON:")
        user_parts.append(json.dumps({"sections": _build_sections_payload(req.sections)}, ensure_ascii=False))
    else:
        user_parts.append("MODE: generate new lyrics from scratch.")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _clone_request(req: LyricsRequest, **updates: Any) -> LyricsRequest:
    if hasattr(req, "model_copy"):
        return req.model_copy(update=updates)
    return req.copy(update=updates)


def _find_missing_sections(sections: List[LyricsSection]) -> List[int]:
    missing = []
    for idx, sec in enumerate(sections):
        if sec.has_lyrics and not (sec.lyrics or "").strip():
            missing.append(idx)
    return missing


def _fill_missing_sections(
    base_sections: List[LyricsSection],
    missing_indices: List[int],
    fill_sections: List[LyricsSection],
) -> List[LyricsSection]:
    if not missing_indices or not fill_sections:
        return base_sections
    fill_iter = iter(fill_sections)
    for idx in missing_indices:
        try:
            candidate = next(fill_iter)
        except StopIteration:
            break
        if candidate.lyrics and candidate.lyrics.strip():
            base_sections[idx] = LyricsSection(
                type=base_sections[idx].type,
                has_lyrics=base_sections[idx].has_lyrics,
                lyrics=candidate.lyrics,
            )
    return base_sections


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


def _normalize_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _section_base(section_type: str) -> str:
    label = (section_type or "").strip().lower()
    if label.startswith("intro"):
        return "intro"
    if label.startswith("outro"):
        return "outro"
    if label.startswith("prechorus") or label.startswith("pre-chorus"):
        return "prechorus"
    if label.startswith("chorus"):
        return "chorus"
    if label.startswith("verse"):
        return "verse"
    if label.startswith("bridge"):
        return "bridge"
    return "default"


def _target_range_for_section(req: LyricsRequest, section_type: str) -> tuple[int, int]:
    targets = _line_targets_for(req)
    base = _section_base(section_type)
    return targets.get(base, targets["default"])


def _format_previous_sections(prev: List[Dict[str, Any]], limit: int = 1600) -> str:
    if not prev:
        return ""
    payload = list(prev)
    text = json.dumps(payload, ensure_ascii=False)
    while len(text) > limit and len(payload) > 1:
        payload.pop(0)
        text = json.dumps(payload, ensure_ascii=False)
    return text[:limit]


def _build_section_messages(
    req: LyricsRequest,
    section: LyricsSection,
    min_lines: int,
    max_lines: int,
    previous_sections: List[Dict[str, Any]],
    existing_lines: List[str],
) -> List[Dict[str, str]]:
    instructions = [
        "You are a professional songwriter.",
        "Return ONLY valid JSON.",
        "Schema: {\"lines\":[\"line1\",\"line2\"]}.",
        "Write lines ONLY for the CURRENT_SECTION.",
        "Keep lines short and natural.",
        "Prefer the upper end of the target line count.",
        "If this section type repeats (e.g., chorus), keep the hook consistent.",
        "Use ASCII punctuation only.",
    ]
    user_parts = [
        f"CURRENT_SECTION: {section.type}",
        f"LINE_TARGET: {min_lines}-{max_lines} lines",
    ]
    if req.language:
        user_parts.append(f"LANGUAGE: {req.language}")
    if req.style:
        user_parts.append(f"STYLE: {req.style}")
    if req.seed_words:
        user_parts.append(f"SEED_WORDS: {req.seed_words}")
    prev_text = _format_previous_sections(previous_sections)
    if prev_text:
        user_parts.append("PREVIOUS_SECTIONS_JSON:")
        user_parts.append(prev_text)
    if existing_lines:
        user_parts.append("EXISTING_LINES:")
        user_parts.append(json.dumps(existing_lines, ensure_ascii=False))
        user_parts.append("MODE: refine and expand existing lines to reach target length.")
    else:
        user_parts.append("MODE: generate new lines.")

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _build_expand_messages(
    req: LyricsRequest,
    section: LyricsSection,
    min_lines: int,
    max_lines: int,
    previous_sections: List[Dict[str, Any]],
    existing_lines: List[str],
) -> List[Dict[str, str]]:
    instructions = [
        "You are a professional songwriter.",
        "Return ONLY valid JSON.",
        "Schema: {\"lines\":[\"line1\",\"line2\"]}.",
        "Expand the EXISTING_LINES to reach the target line count.",
        "Keep the same tone and story.",
        "Use ASCII punctuation only.",
    ]
    user_parts = [
        f"CURRENT_SECTION: {section.type}",
        f"LINE_TARGET: {min_lines}-{max_lines} lines",
        "EXISTING_LINES:",
        json.dumps(existing_lines, ensure_ascii=False),
    ]
    if req.language:
        user_parts.append(f"LANGUAGE: {req.language}")
    if req.style:
        user_parts.append(f"STYLE: {req.style}")
    if req.seed_words:
        user_parts.append(f"SEED_WORDS: {req.seed_words}")
    prev_text = _format_previous_sections(previous_sections)
    if prev_text:
        user_parts.append("PREVIOUS_SECTIONS_JSON:")
        user_parts.append(prev_text)

    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _parse_section_lines(raw_text: str) -> List[str]:
    parsed = _extract_json_safe(raw_text)
    if parsed:
        return _normalize_lines(parsed.get("lines") or parsed.get("lyrics") or parsed.get("text"))
    cleaned = re.sub(r"^```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return _normalize_lines(cleaned)


def _pad_lines(lines: List[str], min_lines: int) -> List[str]:
    if not lines:
        return lines
    while len(lines) < min_lines:
        lines.append(lines[-1])
    return lines

def _parse_response(req: LyricsRequest, raw_text: str) -> List[LyricsSection]:
    parsed = _extract_json(raw_text)
    resp_sections = parsed.get("sections", [])
    output: List[LyricsSection] = []

    for idx, req_sec in enumerate(req.sections):
        lines: List[str] = []
        if idx < len(resp_sections) and isinstance(resp_sections[idx], dict):
            resp_sec = resp_sections[idx]
            if "lines" in resp_sec:
                lines = _normalize_lines(resp_sec.get("lines"))
            elif "lyrics" in resp_sec:
                lines = _normalize_lines(resp_sec.get("lyrics"))
        if not req_sec.has_lyrics:
            lines = []
        lyrics = "\n".join(lines)
        output.append(LyricsSection(type=req_sec.type, has_lyrics=req_sec.has_lyrics, lyrics=lyrics))

    return output


def _call_lmstudio(req: LyricsRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["lmstudio"]).rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    payload = {
        "model": req.model,
        "messages": messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(req: LyricsRequest, messages: List[Dict[str, str]]) -> str:
    base = (req.base_url or DEFAULT_BASE_URLS["ollama"]).rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": req.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "num_predict": req.max_tokens,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def _call_provider(provider: str, req: LyricsRequest, messages: List[Dict[str, str]]) -> str:
    if provider == "ollama":
        return _call_ollama(req, messages)
    return _call_lmstudio(req, messages)


def generate_lyrics(req: LyricsRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    output_sections: List[LyricsSection] = []
    previous_sections: List[Dict[str, Any]] = []
    raw_chunks: List[Dict[str, Any]] = []

    for sec in req.sections:
        if not sec.has_lyrics:
            output_sections.append(LyricsSection(type=sec.type, has_lyrics=False, lyrics=""))
            previous_sections.append({"type": sec.type, "lines": []})
            continue

        min_lines, max_lines = _target_range_for_section(req, sec.type)
        existing_lines = _normalize_lines(sec.lyrics)
        messages = _build_section_messages(
            req, sec, min_lines, max_lines, previous_sections, existing_lines
        )
        raw_text = _call_provider(provider, req, messages)
        lines = _parse_section_lines(raw_text)

        expanded_text = ""
        if len(lines) < min_lines:
            expand_messages = _build_expand_messages(
                req, sec, min_lines, max_lines, previous_sections, lines or existing_lines
            )
            expanded_text = _call_provider(provider, req, expand_messages)
            expanded_lines = _parse_section_lines(expanded_text)
            if expanded_lines:
                lines = expanded_lines

        if not lines and existing_lines:
            lines = existing_lines

        if len(lines) > max_lines:
            lines = lines[:max_lines]
        if len(lines) < min_lines:
            lines = _pad_lines(lines, min_lines)

        lyrics = "\n".join(lines)
        output_sections.append(LyricsSection(type=sec.type, has_lyrics=True, lyrics=lyrics))
        previous_sections.append({"type": sec.type, "lines": lines})
        raw_chunks.append({
            "type": sec.type,
            "raw": raw_text,
            "raw_expand": expanded_text,
        })

    return {
        "sections": [s.model_dump() for s in output_sections],
        "raw_text": json.dumps(raw_chunks, ensure_ascii=False),
        "raw_text_retry": "",
    }


def list_models(req: LyricsModelsRequest) -> Dict[str, Any]:
    provider = (req.provider or "lmstudio").strip().lower()
    base_url = (req.base_url or DEFAULT_BASE_URLS.get(provider, "")).strip()
    if not base_url:
        raise ValueError("base_url is required")

    if provider == "ollama":
        url = base_url.rstrip("/") + "/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models.append(name)
        return {"models": sorted(set(models))}

    if provider == "lmstudio":
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/models"
        else:
            url = f"{base}/v1/models"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("data", []):
            model_id = item.get("id") or item.get("name")
            if model_id:
                models.append(model_id)
        return {"models": sorted(set(models))}

    raise ValueError("provider must be 'lmstudio' or 'ollama'")
