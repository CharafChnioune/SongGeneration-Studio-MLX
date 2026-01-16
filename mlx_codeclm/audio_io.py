"""Audio I/O helpers without torch."""

import typing as tp

import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, target_sr: int = 48000) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.T
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=1)
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int = 48000, subtype: tp.Optional[str] = None) -> None:
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for audio save, got {audio.shape}")
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2:
        raise ValueError(f"Invalid audio shape {audio.shape}")
    sf.write(path, audio.T, sample_rate, subtype=subtype)
