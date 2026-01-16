"""MLX Llama building blocks for SongGeneration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.utils.module_list import ModuleList


@dataclass
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class CausalLMOutput:
    logits: mx.array
    hidden_states: mx.array
    past_key_values: tp.List[tp.Tuple[mx.array, mx.array]]


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.mean(x * x, axis=-1, keepdims=True)
        x = x / mx.sqrt(norm + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq
        self._set_cache(max_position_embeddings)

    def _set_cache(self, seq_len: int) -> None:
        t = mx.arange(seq_len)
        freqs = t[:, None] * self.inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)[None, None, :, :]
        self.sin_cached = mx.sin(emb)[None, None, :, :]
        self.max_seq_len_cached = seq_len

    def __call__(self, seq_len: int) -> tp.Tuple[mx.array, mx.array]:
        if seq_len > self.max_seq_len_cached:
            self._set_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tp.Tuple[mx.array, mx.array]:
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    x = mx.reshape(x, (b, h, 1, t, d))
    x = mx.broadcast_to(x, (b, h, n_rep, t, d))
    return mx.reshape(x, (b, h * n_rep, t, d))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        past_key_value: tp.Optional[tp.Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> tp.Tuple[mx.array, tp.Optional[tp.Tuple[mx.array, mx.array]]]:
        bsz, q_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = mx.transpose(query.reshape((bsz, q_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key = mx.transpose(key.reshape((bsz, q_len, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))
        value = mx.transpose(value.reshape((bsz, q_len, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))

        seq_len = q_len
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
            seq_len += past_len

        cos, sin = self.rotary_emb(seq_len)
        cos = cos[:, :, past_len:past_len + q_len, :]
        sin = sin[:, :, past_len:past_len + q_len, :]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            key = mx.concatenate([past_key_value[0], key], axis=2)
            value = mx.concatenate([past_key_value[1], value], axis=2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)

        attn_weights = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / math.sqrt(self.head_dim)

        # causal mask
        k_len = key.shape[2]
        q_pos = mx.arange(past_len, past_len + q_len)[:, None]
        k_pos = mx.arange(k_len)[None, :]
        causal = q_pos >= k_pos
        attn_weights = mx.where(
            causal[None, None, :, :],
            attn_weights,
            mx.full(attn_weights.shape, -1e9, dtype=attn_weights.dtype),
        )

        attn_weights = attn_weights - mx.max(attn_weights, axis=-1, keepdims=True)
        attn_probs = mx.exp(attn_weights)
        attn_probs = attn_probs / mx.sum(attn_probs, axis=-1, keepdims=True)
        attn_output = mx.matmul(attn_probs, value)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3)).reshape((bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)
        return attn_output, present


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        gate = gate / (1.0 + mx.exp(-gate))
        return self.down_proj(gate * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        past_key_value: tp.Optional[tp.Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> tp.Tuple[mx.array, tp.Optional[tp.Tuple[mx.array, mx.array]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(hidden_states, past_key_value=past_key_value, use_cache=use_cache)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.layers = ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        past_key_values: tp.Optional[tp.List[tp.Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> tp.Tuple[mx.array, tp.List[tp.Tuple[mx.array, mx.array]]]:
        hidden_states = inputs_embeds
        next_past = []
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, past_key_value=past, use_cache=use_cache)
            if use_cache:
                next_past.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_past


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs_embeds: mx.array,
        past_key_values: tp.Optional[tp.List[tp.Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, next_past = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, hidden_states=hidden_states, past_key_values=next_past)
