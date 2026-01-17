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
    reference_audio_id: Optional[str] = None
    model: str = "songgeneration_base"
    # Inference settings (paper-aligned defaults)
    cfg_coef: float = 1.5
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.0
    extend_stride: int = 5

    class Config:
        extra = "ignore"


class UpdateGenerationRequest(BaseModel):
    title: Optional[str] = None
