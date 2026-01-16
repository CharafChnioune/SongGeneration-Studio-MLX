"""MLX port of the MusicFM encoder used for prompt audio embeddings."""

from __future__ import annotations

import json
import os
import typing as tp

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_codeclm.utils.module_list import ModuleList


def _swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _activation_fn(name: str):
    name = name.lower()
    if name in ("swish", "silu"):
        return _swish
    if name in ("gelu", "gelu_new"):
        return _gelu
    if name in ("relu",):
        return lambda x: mx.maximum(x, 0)
    raise ValueError(f"Unsupported activation {name}")


def _power_to_db(x: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    amin = 1e-10
    ref_value = 1.0
    x = np.maximum(x, amin)
    log_spec = 10.0 * np.log10(x)
    log_spec -= 10.0 * np.log10(ref_value)
    max_val = np.max(log_spec)
    if top_db is not None:
        log_spec = np.maximum(log_spec, max_val - top_db)
    return log_spec


class BatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.num_batches_tracked = mx.zeros((1,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = self.running_mean[None, :, None]
        var = self.running_var[None, :, None]
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class BatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.num_batches_tracked = mx.zeros((1,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = self.running_mean[None, :, None, None]
        var = self.running_var[None, :, None, None]
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = mx.random.normal((out_channels, in_channels // groups, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        weight = mx.transpose(self.weight, (0, 2, 1))
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        y = mx.transpose(y, (0, 2, 1))
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: tuple[int, int] = (1, 1),
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = mx.random.normal((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        weight = mx.transpose(self.weight, (0, 2, 3, 1))
        x = mx.transpose(x, (0, 2, 3, 1))
        padding = self.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        y = mx.conv2d(
            x,
            weight,
            stride=self.stride,
            padding=padding,
        )
        y = mx.transpose(y, (0, 3, 1, 2))
        if self.bias is not None:
            y = y + self.bias[None, :, None, None]
        return y


class Res2dModule(nn.Module):
    def __init__(self, idim: int, odim: int, stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv1 = Conv2d(idim, odim, 3, padding=1, stride=stride)
        self.bn1 = BatchNorm2d(odim)
        self.conv2 = Conv2d(odim, odim, 3, padding=1)
        self.bn2 = BatchNorm2d(odim)
        self.diff = (idim != odim) or (stride[0] > 1) or (stride[1] > 1)
        if self.diff:
            self.conv3 = Conv2d(idim, odim, 3, padding=1, stride=stride)
            self.bn3 = BatchNorm2d(odim)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.bn2(self.conv2(mx.maximum(self.bn1(self.conv1(x)), 0)))
        if self.diff:
            x = self.bn3(self.conv3(x))
        out = out + x
        return mx.maximum(out, 0)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, hdim: int, odim: int, strides: list[int] = None, n_bands: int = 64):
        super().__init__()
        if strides is None:
            strides = [2, 2]
        self.conv = ModuleList(
            [
                Res2dModule(idim, hdim, (2, strides[0])),
                Res2dModule(hdim, hdim, (2, strides[1])),
            ]
        )
        self.linear = nn.Linear(hdim * n_bands // 4, odim)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=1)
        for layer in self.conv:
            x = layer(x)
        b, c, f, t = x.shape
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (b, t, c * f))
        return self.linear(x)


class _MelScale(nn.Module):
    def __init__(self, n_fft: int, n_mels: int):
        super().__init__()
        self.fb = mx.zeros((n_fft // 2 + 1, n_mels))


class _Spectrogram(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        self.window = mx.zeros((n_fft,))


class _MelSpectrogram(nn.Module):
    def __init__(self, n_fft: int, n_mels: int):
        super().__init__()
        self.mel_scale = _MelScale(n_fft, n_mels)
        self.spectrogram = _Spectrogram(n_fft)


class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 240,
        n_mels: int = 128,
        is_db: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.is_db = is_db
        self.mel_stft = _MelSpectrogram(n_fft, n_mels)
        self._fb_cache = None
        self._window_cache = None

    def _get_fb_window(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fb_cache is None:
            self._fb_cache = np.array(self.mel_stft.mel_scale.fb, dtype=np.float32)
        if self._window_cache is None:
            self._window_cache = np.array(self.mel_stft.spectrogram.window, dtype=np.float32)
        return self._fb_cache, self._window_cache

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        fb, window = self._get_fb_window()
        specs = []
        for wav in waveform:
            stft = librosa.stft(
                wav.astype(np.float32),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
            )
            spec = np.abs(stft) ** 2
            mel = fb.T @ spec
            if self.is_db:
                mel = _power_to_db(mel)
            specs.append(mel.astype(np.float32))
        return np.stack(specs, axis=0)


class ConformerConfig:
    def __init__(self, cfg: dict):
        self.hidden_size = cfg["hidden_size"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.intermediate_size = cfg["intermediate_size"]
        self.layer_norm_eps = cfg.get("layer_norm_eps", 1e-5)
        self.attention_dropout = cfg.get("attention_dropout", 0.0)
        self.hidden_dropout = cfg.get("hidden_dropout", 0.0)
        self.activation_dropout = cfg.get("activation_dropout", 0.0)
        self.hidden_act = cfg.get("hidden_act", "swish")
        self.conformer_conv_dropout = cfg.get("conformer_conv_dropout", 0.0)
        self.conv_depthwise_kernel_size = cfg.get("conv_depthwise_kernel_size", 31)
        self.position_embeddings_type = cfg.get("position_embeddings_type", "rotary")
        self.rotary_embedding_base = cfg.get("rotary_embedding_base", 10000)
        self.num_conv_pos_embeddings = cfg.get("num_conv_pos_embeddings", 128)
        self.num_conv_pos_embedding_groups = cfg.get("num_conv_pos_embedding_groups", 16)
        self.layerdrop = cfg.get("layerdrop", 0.0)
        self.num_hidden_layers = cfg["num_hidden_layers"]

    @staticmethod
    def from_json(path: str) -> "ConformerConfig":
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        return ConformerConfig(cfg)


class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        sequence_length = hidden_states.shape[1]
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding
        self.cached_sequence_length = sequence_length
        time_stamps = mx.arange(sequence_length, dtype=self.inv_freq.dtype)
        freqs = mx.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = mx.concatenate([freqs, freqs], axis=-1)
        cos_embeddings = mx.cos(embeddings)[:, None, None, :]
        sin_embeddings = mx.sin(embeddings)[:, None, None, :]
        self.cached_rotary_positional_embedding = mx.stack([cos_embeddings, sin_embeddings], axis=0)
        return self.cached_rotary_positional_embedding.astype(hidden_states.dtype)


class Wav2Vec2ConformerFeedForward(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = _activation_fn(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class Wav2Vec2ConformerConvolutionModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("conv_depthwise_kernel_size should be odd for SAME padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.depthwise_conv = Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_depthwise_kernel_size,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.batch_norm = BatchNorm1d(config.hidden_size)
        self.pointwise_conv2 = Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.act = _activation_fn(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1))
        hidden_states = self.pointwise_conv1(hidden_states)
        a, b = mx.split(hidden_states, 2, axis=1)
        hidden_states = a * mx.sigmoid(b)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1))
        return hidden_states


class Wav2Vec2ConformerSelfAttention(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    def _apply_rotary_embedding(self, hidden_states: mx.array, rotary: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads, self.head_size))
        cos = rotary[0, :seq_len, ...]
        sin = rotary[1, :seq_len, ...]
        hidden_states = mx.transpose(hidden_states, (1, 0, 2, 3))
        rotated_begin = hidden_states[..., : self.head_size // 2]
        rotated_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = mx.concatenate([-rotated_end, rotated_begin], axis=-1)
        hidden_states = hidden_states * cos + rotated_states * sin
        hidden_states = mx.transpose(hidden_states, (1, 0, 2, 3))
        return mx.reshape(hidden_states, (batch_size, seq_len, hidden_size))

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: tp.Optional[mx.array] = None,
        relative_position_embeddings: tp.Optional[mx.array] = None,
    ) -> tuple[mx.array, tp.Optional[mx.array]]:
        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError("rotary embeddings requested but missing")
            hidden_states = self._apply_rotary_embedding(hidden_states, relative_position_embeddings)
        batch_size, seq_len, _ = hidden_states.shape
        query = self.linear_q(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_size)
        key = self.linear_k(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_size)
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / np.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = mx.softmax(scores, axis=-1)
        hidden_states = mx.matmul(probs, value)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_size))
        return self.linear_out(hidden_states), probs


class Wav2Vec2ConformerEncoderLayer(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: tp.Optional[mx.array] = None,
        relative_position_embeddings: tp.Optional[mx.array] = None,
    ) -> tuple[mx.array, tp.Optional[mx.array]]:
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, attn_weights


class Wav2Vec2ConformerEncoder(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        if config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = ModuleList([Wav2Vec2ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: tp.Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> dict:
        all_hidden_states = [] if output_hidden_states else None
        relative_position_embeddings = self.embed_positions(hidden_states) if self.embed_positions is not None else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
            )
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class MusicFM25Hz(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 1,
        codebook_dim: int = 16,
        codebook_size: int = 4096,
        features: tp.Sequence[str] = ("melspec_2048",),
        hop_length: int = 240,
        n_mels: int = 128,
        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,
        stat_path: tp.Optional[str] = None,
        w2v2_config_path: tp.Optional[str] = None,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = list(features)
        if stat_path is not None and os.path.exists(stat_path):
            with open(stat_path, "r", encoding="utf-8") as f:
                self.stat = json.load(f)
        else:
            self.stat = {
                "melspec_2048_mean": 6.768444971712967,
                "melspec_2048_std": 18.417922652295623,
            }
        self.preprocessor_melspec_2048 = MelSTFT(n_fft=2048, hop_length=hop_length, n_mels=n_mels, is_db=True)
        self.conv = Conv2dSubsampling(1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels)
        if w2v2_config_path is None or not os.path.exists(w2v2_config_path):
            w2v2_config_path = os.path.join(os.path.dirname(__file__), "w2v2_config.json")
        cfg = ConformerConfig.from_json(w2v2_config_path)
        cfg.num_hidden_layers = encoder_depth
        cfg.hidden_size = encoder_dim
        self.conformer = Wav2Vec2ConformerEncoder(cfg)
        self.linear = nn.Linear(encoder_dim, codebook_size)
        self.cls_token = mx.random.normal((encoder_dim,))

    def preprocessing(self, x: np.ndarray, features: tp.Sequence[str]) -> dict[str, np.ndarray]:
        out = {}
        for key in features:
            layer = getattr(self, f"preprocessor_{key}")
            out[key] = layer(x)[..., :-1]
        return out

    def normalize(self, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key in x.keys():
            x[key] = (x[key] - self.stat[f"{key}_mean"]) / self.stat[f"{key}_std"]
        return x

    def encoder(self, x: np.ndarray) -> tuple[dict[str, mx.array], list[mx.array]]:
        x = self.conv(mx.array(x))
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    def get_predictions(self, x: np.ndarray) -> tuple[dict[str, mx.array], list[mx.array]]:
        x = self.preprocessing(x, features=["melspec_2048"])
        x = self.normalize(x)
        return self.encoder(x["melspec_2048"])


class MusicFMModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.model = MusicFM25Hz(
            num_codebooks=cfg.get("num_codebooks", 1),
            codebook_dim=cfg.get("codebook_dim", 16),
            codebook_size=cfg.get("codebook_size", 4096),
            features=cfg.get("features", ["melspec_2048"]),
            hop_length=cfg.get("hop_length", 240),
            n_mels=cfg.get("n_mels", 128),
            conv_dim=cfg.get("conv_dim", 512),
            encoder_dim=cfg.get("encoder_dim", 1024),
            encoder_depth=cfg.get("encoder_depth", 12),
            stat_path=cfg.get("stat_path"),
            w2v2_config_path=cfg.get("w2v2_config_path"),
        )

    def __call__(self, source: np.ndarray, features_only: bool = False) -> dict:
        label_rate = self.cfg.get("label_rate", 25)
        sample_rate = self.cfg.get("sample_rate", 24000)
        trim = sample_rate // label_rate
        source = source[..., : (source.shape[-1] // trim) * trim]
        if features_only:
            _, hidden_states = self.model.get_predictions(source)
            return {"layer_results": hidden_states}
        logits, hidden_states = self.model.get_predictions(source)
        return {"logits": logits, "hidden_emb": hidden_states}
