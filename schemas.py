"""
SongGeneration Studio - Pydantic Schemas
Data models for API requests and responses.
"""

from typing import Optional, List
from pydantic import BaseModel

# ============================================================================
# Request/Response Models
# ============================================================================

class Section(BaseModel):
    type: str
    lyrics: Optional[str] = None

class LyricsSection(BaseModel):
    type: str
    has_lyrics: bool = True
    lyrics: Optional[str] = None


class LyricsRequest(BaseModel):
    provider: str = "lmstudio"
    model: str
    base_url: Optional[str] = None
    seed_words: str = ""
    sections: List[LyricsSection]
    mode: str = "generate"  # generate | refine
    style: Optional[str] = None
    language: Optional[str] = None
    length: str = "full"  # short | medium | full
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 800


class LyricsModelsRequest(BaseModel):
    provider: str = "lmstudio"
    base_url: Optional[str] = None

class StyleRequest(BaseModel):
    provider: str = "lmstudio"
    model: str
    base_url: Optional[str] = None
    seed_words: str = ""
    title: Optional[str] = None
    lyrics: Optional[str] = None
    language: Optional[str] = None
    target_genre: Optional[str] = None

class StructureRequest(BaseModel):
    provider: str = "lmstudio"
    model: str
    base_url: Optional[str] = None
    seed_words: str = ""
    title: Optional[str] = None
    language: Optional[str] = None
    target_genre: Optional[str] = None
    style: Optional[str] = None
    length: str = "full"

class RemixRequest(BaseModel):
    provider: str = "lmstudio"
    model: str
    base_url: Optional[str] = None
    song_model: str
    prompt: str
    title: Optional[str] = None
    language: Optional[str] = None
    sections: List[LyricsSection]
    genre: Optional[str] = None
    moods: Optional[List[str]] = None
    timbres: Optional[List[str]] = None
    instruments: Optional[List[str]] = None
    bpm: Optional[int] = None
    gender: Optional[str] = None
    custom_style: Optional[str] = None
    output_mode: Optional[str] = None
    reference_audio_id: Optional[str] = None
    arrangement_template: Optional[str] = None
    advanced: Optional[dict] = None
    length: str = "full"

class SongRequest(BaseModel):
    title: str = "Untitled"
    sections: List[Section]
    gender: str = "female"
    timbre: str = ""
    genre: str = ""
    emotion: str = ""
    instruments: str = ""
    custom_style: Optional[str] = None  # Additional free-text style descriptors
    bpm: int = 120
    output_mode: str = "mixed"
    auto_prompt_type: Optional[str] = None
    reference_audio_id: Optional[str] = None
    model: str = "songgeneration_base"
    memory_mode: str = "auto"
    # Advanced generation parameters
    cfg_coef: float = 2.2          # Classifier-free guidance (0.1-3.0)
    temperature: float = 0.7       # Sampling randomness (0.1-2.0)
    top_k: int = 60                # Top-K sampling (1-250)
    top_p: float = 0.9             # Nucleus sampling, 0 = disabled (0.0-1.0)
    extend_stride: int = 6         # Extension stride for longer songs
    allow_intro_outro_lyrics: bool = False
    use_genre_presets: bool = True
    num_candidates: int = 2
    auto_select_best: bool = True
    arrangement_template: Optional[str] = None


class UpdateGenerationRequest(BaseModel):
    title: Optional[str] = None
