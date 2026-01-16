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


def _normalize_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(value).strip()] if str(value).strip() else []


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


def generate_lyrics(req: LyricsRequest) -> Dict[str, Any]:
    if not req.model:
        raise ValueError("model is required")
    provider = (req.provider or "lmstudio").strip().lower()
    if provider not in ("lmstudio", "ollama"):
        raise ValueError("provider must be 'lmstudio' or 'ollama'")

    messages = _build_messages(req)
    if provider == "ollama":
        raw_text = _call_ollama(req, messages)
    else:
        raw_text = _call_lmstudio(req, messages)

    sections = _parse_response(req, raw_text)
    missing = _find_missing_sections(sections)
    retry_text = ""
    if missing:
        missing_sections = [req.sections[idx] for idx in missing]
        retry_req = _clone_request(req, sections=missing_sections, mode="generate")
        retry_messages = _build_messages(retry_req)
        if provider == "ollama":
            retry_text = _call_ollama(retry_req, retry_messages)
        else:
            retry_text = _call_lmstudio(retry_req, retry_messages)
        fill_sections = _parse_response(retry_req, retry_text)
        sections = _fill_missing_sections(sections, missing, fill_sections)

    return {
        "sections": [s.model_dump() for s in sections],
        "raw_text": raw_text,
        "raw_text_retry": retry_text,
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
