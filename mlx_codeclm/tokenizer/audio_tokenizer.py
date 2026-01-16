"""Audio tokenizer interface for MLX."""

from abc import ABC, abstractmethod
import os
import typing as tp

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_codeclm.tokenizer.flow1dvae import Flow1dVAE1rvq as Flow1dVAE1rvqModel
from mlx_codeclm.tokenizer.flow1dvae import Flow1dVAESeparate as Flow1dVAESeparateModel
from mlx_codeclm.tokenizer.flow1dvae import normalize_prompt_audio
from mlx_codeclm.tokenizer.musicfm import MusicFMModel
from mlx_codeclm.utils.weights import load_weights_npz_prefixed


class AudioTokenizer(ABC, nn.Module):
    @abstractmethod
    def encode(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        ...

    @abstractmethod
    def decode(self, codes: mx.array, *args, **kwargs) -> mx.array:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @staticmethod
    def get_pretrained(
        name: str,
        vae_config: str,
        vae_model: str,
        model_weights: str,
    ) -> "AudioTokenizer":
        if name.startswith("Flow1dVAE1rvq_"):
            return Flow1dVAE1rvq(name, vae_config, vae_model, model_weights)
        if name.startswith("Flow1dVAESeparate_"):
            return Flow1dVAESeparate(name, vae_config, vae_model, model_weights)
        raise NotImplementedError(f"Unsupported audio tokenizer {name}")


class Flow1dVAE1rvq(AudioTokenizer):
    def __init__(self, name: str, vae_config: str, vae_model: str, model_weights: str):
        super().__init__()
        gpt_config = {
            "vocab_size": 50257,
            "max_position_embeddings": 1000,
            "hidden_size": 1200,
            "num_hidden_layers": 39,
            "num_attention_heads": 30,
            "n_inner": 4800,
            "activation_function": "gelu_new",
            "mlp_in": 1200,
            "mlp_hidden": 1024,
            "mlp_out": 768,
        }
        if not model_weights.endswith(".npz"):
            raise ValueError("Flow1dVAE1rvq requires MLX .npz weights for audio tokenizer")
        if not vae_model.endswith(".npz"):
            raise ValueError("Flow1dVAE1rvq requires MLX .npz weights for VAE")
        self.model = Flow1dVAE1rvqModel(
            vae_config=vae_config,
            vae_weights=vae_model,
            gpt_config=gpt_config,
            weights_path=model_weights,
        )
        self.n_quantizers = 1
        self.layer_num = 6
        self._sample_rate = 48000
        self._bestrq_sample_rate = 24000
        self._encode_batch_size = 3
        w2v2_config_path = os.path.join(os.path.dirname(__file__), "w2v2_config.json")
        self.bestrq = MusicFMModel(
            {
                "label_rate": 25,
                "num_codebooks": 1,
                "codebook_dim": 16,
                "codebook_size": 4096,
                "features": ["melspec_2048"],
                "hop_length": 240,
                "n_mels": 128,
                "conv_dim": 512,
                "encoder_dim": 1024,
                "encoder_depth": 12,
                "sample_rate": self._bestrq_sample_rate,
                "w2v2_config_path": w2v2_config_path,
            }
        )
        load_weights_npz_prefixed(self.bestrq, model_weights, prefix="bestrq.", quiet=True)

    def _preprocess_audio(self, input_audios: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        if input_audios.ndim != 3:
            raise ValueError(f"Expected [B, C, T] audio, got {input_audios.shape}")
        flat = input_audios.reshape(input_audios.shape[0], -1)
        max_volume = np.max(np.abs(flat), axis=-1)
        norm_value = np.ones_like(max_volume)
        mask = max_volume > threshold
        norm_value[mask] = max_volume[mask] / threshold
        return input_audios / norm_value[:, None, None]

    def _resample_bestrq(self, audio: np.ndarray) -> np.ndarray:
        if self._sample_rate == self._bestrq_sample_rate:
            return audio.astype(np.float32)
        resampled = []
        max_len = 0
        for wav in audio:
            wav = librosa.resample(
                wav.astype(np.float32),
                orig_sr=self._sample_rate,
                target_sr=self._bestrq_sample_rate,
            )
            resampled.append(wav.astype(np.float32))
            max_len = max(max_len, wav.shape[-1])
        for idx in range(len(resampled)):
            if resampled[idx].shape[-1] < max_len:
                pad = max_len - resampled[idx].shape[-1]
                resampled[idx] = np.pad(resampled[idx], (0, pad), mode="constant")
        return np.stack(resampled, axis=0)

    def _fetch_codes_batch(self, input_audios: np.ndarray, layer: int) -> mx.array:
        input_audio_0 = input_audios[:, 0, :]
        input_audio_1 = input_audios[:, 1, :]
        input_wav_mean = (input_audio_0 + input_audio_1) / 2.0
        resampled = self._resample_bestrq(input_wav_mean)
        layer_results = self.bestrq(resampled, features_only=True)["layer_results"]
        if layer >= len(layer_results):
            raise ValueError(f"Requested layer {layer} but only {len(layer_results)} available")
        bestrq_emb = mx.transpose(layer_results[layer], (0, 2, 1))
        _, codes, *_ = self.model.model.rvq_bestrq_emb(bestrq_emb)
        return codes

    def encode(self, x: mx.array):
        audio = np.array(x)
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            if audio.shape[0] in (1, 2):
                audio = audio[None, ...]
            else:
                audio = audio[:, None, :]
        elif audio.ndim != 3:
            raise ValueError(f"Unsupported audio shape {audio.shape}")
        if audio.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for MLX prompt encoding")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
        audios = self._preprocess_audio(audio.astype(np.float32))
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        while audios.shape[-1] < min_samples:
            audios = np.concatenate([audios, audios], axis=-1)
        int_max_len = audios.shape[-1] // min_samples + 1
        audios = np.concatenate([audios, audios], axis=-1)
        audios = audios[:, : int(int_max_len * min_samples)]
        audio_input = audios.reshape(2, -1, min_samples).transpose(1, 0, 2)
        codes_list = []
        for idx in range(0, audio_input.shape[0], self._encode_batch_size):
            batch = audio_input[idx : idx + self._encode_batch_size]
            codes_list.append(self._fetch_codes_batch(batch, layer=self.layer_num))
        if not codes_list:
            raise ValueError("No audio segments to encode")
        codes = mx.concatenate(codes_list, axis=0)
        codes = mx.transpose(codes, (1, 0, 2))
        codes = mx.reshape(codes, (codes.shape[0], -1))
        codes = mx.expand_dims(codes, axis=0)
        codes = codes[:, :, :output_len]
        return codes.astype(mx.int32), None

    def decode(self, codes: mx.array, prompt: tp.Optional[mx.array] = None, *args, **kwargs) -> mx.array:
        if codes.ndim == 2:
            codes = codes[:, None, :]
        prompt_codes = None
        prompt_audio = None
        if prompt is not None:
            prompt_audio = normalize_prompt_audio(prompt, self.sample_rate)
            prompt_codes, _ = self.encode(prompt_audio)
        return self.model.decode(codes, prompt=prompt_audio, prompt_codes=prompt_codes, *args, **kwargs)

    @property
    def frame_rate(self) -> float:
        return 25

    @property
    def sample_rate(self) -> int:
        return 48000

    @property
    def cardinality(self) -> int:
        return 16384

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers


class Flow1dVAESeparate(AudioTokenizer):
    def __init__(self, name: str, vae_config: str, vae_model: str, model_weights: str):
        super().__init__()
        gpt_config = {
            "vocab_size": 50257,
            "max_position_embeddings": 1000,
            "hidden_size": 2200,
            "num_hidden_layers": 16,
            "num_attention_heads": 20,
            "n_inner": 4400,
            "activation_function": "gelu_new",
            "mlp_in": 2200,
            "mlp_hidden": 1024,
            "mlp_out": 768,
        }
        if not model_weights.endswith(".npz"):
            raise ValueError("Flow1dVAESeparate requires MLX .npz weights for audio tokenizer")
        if not vae_model.endswith(".npz"):
            raise ValueError("Flow1dVAESeparate requires MLX .npz weights for VAE")
        self.model = Flow1dVAESeparateModel(
            vae_config=vae_config,
            vae_weights=vae_model,
            gpt_config=gpt_config,
            weights_path=model_weights,
        )
        self.n_quantizers = 1
        self.layer_vocal = 7
        self.layer_bgm = 3
        self._sample_rate = 48000
        self._bestrq_sample_rate = 24000
        self._encode_batch_size = 3
        w2v2_config_path = os.path.join(os.path.dirname(__file__), "w2v2_config.json")
        self.bestrq = MusicFMModel(
            {
                "label_rate": 25,
                "num_codebooks": 1,
                "codebook_dim": 16,
                "codebook_size": 4096,
                "features": ["melspec_2048"],
                "hop_length": 240,
                "n_mels": 128,
                "conv_dim": 512,
                "encoder_dim": 1024,
                "encoder_depth": 12,
                "sample_rate": self._bestrq_sample_rate,
                "w2v2_config_path": w2v2_config_path,
            }
        )
        load_weights_npz_prefixed(self.bestrq, model_weights, prefix="bestrq.", quiet=True)

    def _preprocess_audio(self, input_audios: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        if input_audios.ndim != 3:
            raise ValueError(f"Expected [B, C, T] audio, got {input_audios.shape}")
        flat = input_audios.reshape(input_audios.shape[0], -1)
        max_volume = np.max(np.abs(flat), axis=-1)
        norm_value = np.ones_like(max_volume)
        mask = max_volume > threshold
        norm_value[mask] = max_volume[mask] / threshold
        return input_audios / norm_value[:, None, None]

    def _resample_bestrq(self, audio: np.ndarray) -> np.ndarray:
        if self._sample_rate == self._bestrq_sample_rate:
            return audio.astype(np.float32)
        resampled = []
        max_len = 0
        for wav in audio:
            wav = librosa.resample(
                wav.astype(np.float32),
                orig_sr=self._sample_rate,
                target_sr=self._bestrq_sample_rate,
            )
            resampled.append(wav.astype(np.float32))
            max_len = max(max_len, wav.shape[-1])
        for idx in range(len(resampled)):
            if resampled[idx].shape[-1] < max_len:
                pad = max_len - resampled[idx].shape[-1]
                resampled[idx] = np.pad(resampled[idx], (0, pad), mode="constant")
        return np.stack(resampled, axis=0)

    def _fetch_codes_batch(
        self, input_vocal: np.ndarray, input_bgm: np.ndarray, layer_vocal: int, layer_bgm: int
    ) -> tp.Tuple[mx.array, mx.array]:
        vocal_0 = input_vocal[:, 0, :]
        vocal_1 = input_vocal[:, 1, :]
        bgm_0 = input_bgm[:, 0, :]
        bgm_1 = input_bgm[:, 1, :]
        vocal_mean = (vocal_0 + vocal_1) / 2.0
        bgm_mean = (bgm_0 + bgm_1) / 2.0
        vocal_resampled = self._resample_bestrq(vocal_mean)
        bgm_resampled = self._resample_bestrq(bgm_mean)
        vocal_out = self.bestrq(vocal_resampled, features_only=True)
        bgm_out = self.bestrq(bgm_resampled, features_only=True)
        vocal_emb = vocal_out["layer_results"][layer_vocal]
        bgm_emb = bgm_out["layer_results"][layer_bgm]
        vocal_emb = mx.transpose(mx.array(vocal_emb), (0, 2, 1))
        bgm_emb = mx.transpose(mx.array(bgm_emb), (0, 2, 1))
        _, codes_vocal, _, _, _, _ = self.model.model.rvq_bestrq_emb(vocal_emb)
        _, codes_bgm, _, _, _, _ = self.model.model.rvq_bestrq_bgm_emb(bgm_emb)
        return codes_vocal.astype(mx.int32), codes_bgm.astype(mx.int32)

    def encode(self, x_vocal: np.ndarray, x_bgm: np.ndarray) -> tp.Tuple[mx.array, mx.array]:
        x_vocal = np.array(x_vocal)
        x_bgm = np.array(x_bgm)
        if x_vocal.ndim == 2:
            x_vocal = x_vocal[None, ...]
        if x_bgm.ndim == 2:
            x_bgm = x_bgm[None, ...]
        if x_vocal.shape[1] == 1:
            x_vocal = np.repeat(x_vocal, 2, axis=1)
        if x_bgm.shape[1] == 1:
            x_bgm = np.repeat(x_bgm, 2, axis=1)
        if x_vocal.shape[1] > 2:
            x_vocal = x_vocal[:, :2, :]
        if x_bgm.shape[1] > 2:
            x_bgm = x_bgm[:, :2, :]

        vocal = self._preprocess_audio(x_vocal.astype(np.float32))
        bgm = self._preprocess_audio(x_bgm.astype(np.float32))
        min_len = min(vocal.shape[-1], bgm.shape[-1])
        vocal = vocal[:, :, :min_len]
        bgm = bgm[:, :, :min_len]

        vocal = vocal.squeeze(0)
        bgm = bgm.squeeze(0)
        orig_length = vocal.shape[-1]
        min_samples = int(40 * self.sample_rate)
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1

        while vocal.shape[-1] < min_samples:
            vocal = np.concatenate([vocal, vocal], axis=-1)
            bgm = np.concatenate([bgm, bgm], axis=-1)
        int_max_len = vocal.shape[-1] // min_samples + 1
        vocal = np.concatenate([vocal, vocal], axis=-1)
        bgm = np.concatenate([bgm, bgm], axis=-1)
        vocal = vocal[:, : int(int_max_len * min_samples)]
        bgm = bgm[:, : int(int_max_len * min_samples)]

        vocal_input = vocal.reshape(2, -1, min_samples).transpose(1, 0, 2)
        bgm_input = bgm.reshape(2, -1, min_samples).transpose(1, 0, 2)
        codes_vocal_list = []
        codes_bgm_list = []
        for idx in range(0, vocal_input.shape[0], self._encode_batch_size):
            batch_vocal = vocal_input[idx : idx + self._encode_batch_size]
            batch_bgm = bgm_input[idx : idx + self._encode_batch_size]
            codes_vocal, codes_bgm = self._fetch_codes_batch(batch_vocal, batch_bgm, self.layer_vocal, self.layer_bgm)
            codes_vocal_list.append(codes_vocal)
            codes_bgm_list.append(codes_bgm)
        if not codes_vocal_list:
            raise ValueError("No audio segments to encode")

        codes_vocal = mx.concatenate(codes_vocal_list, axis=0)
        codes_bgm = mx.concatenate(codes_bgm_list, axis=0)
        codes_vocal = mx.transpose(codes_vocal, (1, 0, 2))
        codes_bgm = mx.transpose(codes_bgm, (1, 0, 2))
        codes_vocal = mx.reshape(codes_vocal, (codes_vocal.shape[0], -1))
        codes_bgm = mx.reshape(codes_bgm, (codes_bgm.shape[0], -1))
        codes_vocal = mx.expand_dims(codes_vocal, axis=0)
        codes_bgm = mx.expand_dims(codes_bgm, axis=0)
        codes_vocal = codes_vocal[:, :, :output_len]
        codes_bgm = codes_bgm[:, :, :output_len]
        return codes_vocal.astype(mx.int32), codes_bgm.astype(mx.int32)

    def decode(
        self,
        codes: tp.Union[tp.Sequence[mx.array], mx.array],
        prompt_vocal: tp.Optional[mx.array] = None,
        prompt_bgm: tp.Optional[mx.array] = None,
        *args,
        **kwargs,
    ) -> mx.array:
        if isinstance(codes, (list, tuple)):
            codes_vocal, codes_bgm = codes
        else:
            codes_vocal = codes[:, [0], :]
            codes_bgm = codes[:, [1], :]
        prompt_codes = None
        prompt_vocal_audio = None
        prompt_bgm_audio = None
        if prompt_vocal is not None and prompt_bgm is not None:
            prompt_vocal_audio = normalize_prompt_audio(prompt_vocal, self.sample_rate)
            prompt_bgm_audio = normalize_prompt_audio(prompt_bgm, self.sample_rate)
            prompt_codes = self.encode(prompt_vocal_audio, prompt_bgm_audio)
        return self.model.decode(
            [codes_vocal, codes_bgm],
            prompt_vocal=prompt_vocal_audio,
            prompt_bgm=prompt_bgm_audio,
            prompt_codes=prompt_codes,
            *args,
            **kwargs,
        )

    @property
    def frame_rate(self) -> float:
        return 25

    @property
    def sample_rate(self) -> int:
        return 48000

    @property
    def cardinality(self) -> int:
        return 16384

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers
