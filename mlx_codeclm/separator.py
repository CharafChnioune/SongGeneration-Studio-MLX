"""Torch-free separator utilities (ONNX/CoreML) for Demucs-style stems."""

from __future__ import annotations

import os
import typing as tp

import librosa
import numpy as np
import soundfile as sf


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio, 2, axis=0)
        if audio.shape[0] > 2:
            return audio[:2]
        return audio
    raise ValueError(f"Unsupported audio shape {audio.shape}")


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.T
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=1)
    return _ensure_stereo(audio).astype(np.float32)


def _crop_audio(audio: np.ndarray, sample_rate: int, max_seconds: tp.Optional[float]) -> np.ndarray:
    if max_seconds is None:
        return audio
    max_len = int(max_seconds * sample_rate)
    return audio[:, :max_len]


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, axis=1)
    return audio.astype(np.float32)


def _normalize_stems(stems: np.ndarray, num_sources: int = 4) -> np.ndarray:
    arr = np.asarray(stems)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported separator output shape {arr.shape}")

    source_axis = None
    for idx, size in enumerate(arr.shape):
        if size == num_sources:
            source_axis = idx
            break
    if source_axis is None:
        raise ValueError(f"Could not infer source axis from {arr.shape}")

    channel_axis = None
    for idx, size in enumerate(arr.shape):
        if idx == source_axis:
            continue
        if size in (1, 2):
            channel_axis = idx
            break
    if channel_axis is None:
        raise ValueError(f"Could not infer channel axis from {arr.shape}")

    time_axis = [idx for idx in range(3) if idx not in (source_axis, channel_axis)][0]
    arr = np.moveaxis(arr, (source_axis, channel_axis, time_axis), (0, 1, 2))
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return arr.astype(np.float32)


class BaseSeparator:
    def __init__(
        self,
        sample_rate: int,
        chunk_seconds: float,
        overlap: float,
        vocals_index: int,
        max_seconds: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.overlap = overlap
        self.vocals_index = vocals_index
        self.max_seconds = max_seconds

    def run(self, audio_path: str, target_sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_audio = _load_audio(audio_path, target_sr)
        full_audio = _crop_audio(full_audio, target_sr, self.max_seconds)

        sep_audio = _load_audio(audio_path, self.sample_rate)

        stems = self._separate(sep_audio)
        if stems.shape[0] <= self.vocals_index:
            raise ValueError(f"Separator returned {stems.shape[0]} stems, cannot pick vocals index {self.vocals_index}")
        vocals = stems[self.vocals_index]
        vocals = _ensure_stereo(vocals)
        vocals = _resample(vocals, self.sample_rate, target_sr)
        vocals = _crop_audio(vocals, target_sr, self.max_seconds)

        min_len = min(full_audio.shape[-1], vocals.shape[-1])
        full_audio = full_audio[:, :min_len]
        vocals = vocals[:, :min_len]
        bgm = full_audio - vocals
        return full_audio, vocals, bgm

    def _separate(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OnnxSeparator(BaseSeparator):
    def __init__(
        self,
        model_path: str,
        sample_rate: int,
        chunk_seconds: float,
        overlap: float,
        vocals_index: int,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            vocals_index=vocals_index,
        )
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX separator has no inputs")
        self.mix_input = None
        self.spec_input = None
        for inp in inputs:
            name = inp.name
            if name == "mix" or self.mix_input is None:
                self.mix_input = inp
            if name == "spec" or self.spec_input is None:
                self.spec_input = inp
        if self.mix_input is None:
            self.mix_input = inputs[0]
        self.mix_samples = None
        if self.mix_input.shape and len(self.mix_input.shape) >= 3:
            mix_len = self.mix_input.shape[-1]
            if isinstance(mix_len, int):
                self.mix_samples = mix_len
        self.spec_bins = None
        self.spec_frames = None
        if self.spec_input is not None and self.spec_input.shape and len(self.spec_input.shape) >= 4:
            bins = self.spec_input.shape[-3]
            frames = self.spec_input.shape[-2]
            if isinstance(bins, int):
                self.spec_bins = bins
            if isinstance(frames, int):
                self.spec_frames = frames
        if self.spec_bins is not None:
            self.n_fft = self.spec_bins * 2
            self.hop_length = self.n_fft // 4
        else:
            self.n_fft = 4096
            self.hop_length = self.n_fft // 4

        outputs = self.session.get_outputs()
        self.output_name = None
        for out in outputs:
            shape = out.shape or []
            if len(shape) == 4 and shape[1] == 4 and shape[2] == 2:
                self.output_name = out.name
                break
        if self.output_name is None and outputs:
            self.output_name = outputs[-1].name

    def _compute_spec(self, audio: np.ndarray) -> np.ndarray:
        specs = []
        for ch in audio:
            spec = librosa.stft(
                ch,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window="hann",
                center=True,
                pad_mode="reflect",
            )
            spec = spec / np.sqrt(self.n_fft)
            spec = spec[:-1, :]
            specs.append(spec)
        spec = np.stack(specs, axis=0)
        if self.spec_frames is not None:
            frames = spec.shape[-1]
            if frames < self.spec_frames:
                pad = self.spec_frames - frames
                spec = np.pad(spec, ((0, 0), (0, 0), (0, pad)), mode="constant")
            elif frames > self.spec_frames:
                spec = spec[:, :, : self.spec_frames]
        spec = np.stack([spec.real, spec.imag], axis=-1)
        return spec.astype(np.float32)

    def _infer_chunk(self, audio: np.ndarray) -> np.ndarray:
        if self.mix_samples is not None:
            if audio.shape[-1] < self.mix_samples:
                pad = self.mix_samples - audio.shape[-1]
                audio = np.pad(audio, ((0, 0), (0, pad)), mode="constant")
            elif audio.shape[-1] > self.mix_samples:
                audio = audio[:, : self.mix_samples]
        inputs = {self.mix_input.name: audio[None, ...].astype(np.float32)}
        if self.spec_input is not None:
            spec = self._compute_spec(audio)
            inputs[self.spec_input.name] = spec[None, ...]
        if self.output_name is not None:
            outputs = self.session.run([self.output_name], inputs)
        else:
            outputs = self.session.run(None, inputs)
        if not outputs:
            raise ValueError("Separator returned no outputs")
        output = outputs[0]
        return _normalize_stems(output)

    def _separate(self, audio: np.ndarray) -> np.ndarray:
        total = audio.shape[-1]
        if self.mix_samples is not None:
            chunk_samples = self.mix_samples
        else:
            chunk_samples = int(self.chunk_seconds * self.sample_rate)
        if self.chunk_seconds <= 0:
            return self._infer_chunk(audio)[..., :total]
        if total <= chunk_samples:
            return self._infer_chunk(audio)[..., :total]

        overlap_samples = int(chunk_samples * self.overlap)
        hop = max(1, chunk_samples - overlap_samples)
        window = np.ones(chunk_samples, dtype=np.float32)
        if overlap_samples > 0:
            fade = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
            window[:overlap_samples] = fade
            window[-overlap_samples:] = fade[::-1]

        first_chunk = audio[:, :chunk_samples]
        if first_chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - first_chunk.shape[-1]
            first_chunk = np.pad(first_chunk, ((0, 0), (0, pad)), mode="constant")
        first_stems = self._infer_chunk(first_chunk)
        num_sources = first_stems.shape[0]
        output = np.zeros((num_sources, 2, total), dtype=np.float32)
        weight = np.zeros((total,), dtype=np.float32)

        def add_chunk(start: int, chunk_audio: np.ndarray, stems: np.ndarray) -> None:
            end = min(start + chunk_samples, total)
            chunk_len = end - start
            window_chunk = window[:chunk_len]
            output[:, :, start:end] += stems[..., :chunk_len] * window_chunk[None, None, :]
            weight[start:end] += window_chunk

        add_chunk(0, first_chunk, first_stems)
        for start in range(hop, total, hop):
            end = min(start + chunk_samples, total)
            chunk = audio[:, start:end]
            if chunk.shape[-1] < chunk_samples:
                pad = chunk_samples - chunk.shape[-1]
                chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")
            stems = self._infer_chunk(chunk)
            add_chunk(start, chunk, stems)

        weight = np.maximum(weight, 1e-8)
        output = output / weight[None, None, :]
        return output


class CoreMLSeparator(BaseSeparator):
    def __init__(
        self,
        model_path: str,
        sample_rate: int,
        chunk_seconds: float,
        overlap: float,
        vocals_index: int,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            vocals_index=vocals_index,
        )
        import coremltools as ct

        self.model = ct.models.MLModel(model_path)
        input_names = list(self.model.input_description.keys())
        if not input_names:
            raise ValueError("CoreML model has no inputs")
        self.input_name = input_names[0]
        output_names = list(self.model.output_description.keys())
        if not output_names:
            raise ValueError("CoreML model has no outputs")
        self.output_name = output_names[0]

    def _infer_chunk(self, audio: np.ndarray) -> np.ndarray:
        inputs = {self.input_name: audio[None, ...].astype(np.float32)}
        outputs = self.model.predict(inputs)
        if self.output_name not in outputs:
            raise ValueError("CoreML separator output not found")
        return _normalize_stems(outputs[self.output_name])

    def _separate(self, audio: np.ndarray) -> np.ndarray:
        total = audio.shape[-1]
        if self.chunk_seconds <= 0:
            return self._infer_chunk(audio)[..., :total]
        chunk_samples = int(self.chunk_seconds * self.sample_rate)
        if total <= chunk_samples:
            return self._infer_chunk(audio)[..., :total]

        overlap_samples = int(chunk_samples * self.overlap)
        hop = max(1, chunk_samples - overlap_samples)
        window = np.ones(chunk_samples, dtype=np.float32)
        if overlap_samples > 0:
            fade = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
            window[:overlap_samples] = fade
            window[-overlap_samples:] = fade[::-1]

        first_chunk = audio[:, :chunk_samples]
        if first_chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - first_chunk.shape[-1]
            first_chunk = np.pad(first_chunk, ((0, 0), (0, pad)), mode="constant")
        first_stems = self._infer_chunk(first_chunk)
        num_sources = first_stems.shape[0]
        output = np.zeros((num_sources, 2, total), dtype=np.float32)
        weight = np.zeros((total,), dtype=np.float32)

        def add_chunk(start: int, stems: np.ndarray) -> None:
            end = min(start + chunk_samples, total)
            chunk_len = end - start
            window_chunk = window[:chunk_len]
            output[:, :, start:end] += stems[..., :chunk_len] * window_chunk[None, None, :]
            weight[start:end] += window_chunk

        add_chunk(0, first_stems)
        for start in range(hop, total, hop):
            end = min(start + chunk_samples, total)
            chunk = audio[:, start:end]
            if chunk.shape[-1] < chunk_samples:
                pad = chunk_samples - chunk.shape[-1]
                chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")
            stems = self._infer_chunk(chunk)
            add_chunk(start, stems)

        weight = np.maximum(weight, 1e-8)
        output = output / weight[None, None, :]
        return output


def create_separator(
    backend: str,
    model_path: tp.Optional[str],
    sample_rate: int,
    chunk_seconds: float,
    overlap: float,
    vocals_index: int,
) -> BaseSeparator:
    if model_path is None:
        if backend == "onnx":
            model_path = os.path.join("third_party", "demucs", "ckpt", "htdemucs.onnx")
        elif backend == "coreml":
            model_path = os.path.join("third_party", "demucs", "ckpt", "htdemucs.mlpackage")
    if model_path is None:
        raise ValueError("separator_model is required")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Separator model not found: {model_path}")
    if backend == "onnx":
        return OnnxSeparator(
            model_path=model_path,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            vocals_index=vocals_index,
        )
    if backend == "coreml":
        return CoreMLSeparator(
            model_path=model_path,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            vocals_index=vocals_index,
        )
    raise ValueError(f"Unsupported separator backend {backend}")
