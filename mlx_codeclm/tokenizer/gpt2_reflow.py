"""MLX GPT-2 variant used by Flow1dVAE CFM estimator."""

from __future__ import annotations

import math
import types
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.utils.module_list import ModuleList


def gelu_new(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


def _act_fn(name: str):
    if name in ("gelu_new", "gelu"):
        return gelu_new
    if name == "silu":
        return lambda x: x * mx.sigmoid(x)
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    raise ValueError(f"Unsupported activation {name}")


class Conv1D(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = mx.random.normal((in_features, out_features)) * 0.02
        self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x, self.weight) + self.bias


def precompute_freqs_cis(dim: int, end: int, constant: float = 10000.0) -> tuple[mx.array, mx.array]:
    freqs = 1.0 / (constant ** (mx.arange(0, dim, 2) / dim))
    t = mx.arange(end)
    angles = mx.outer(t, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    b, t, h, d = x.shape
    x = mx.reshape(x, (b, t, h, d // 2, 2))
    x0 = x[..., 0]
    x1 = x[..., 1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = mx.stack([out0, out1], axis=-1)
    return mx.reshape(out, (b, t, h, d))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = x * mx.sigmoid(x)
        x = self.linear_2(x)
        return x


class PixArtAlphaCombinedFlowEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.flow_t_size = 512
        self.outdim = size_emb_dim
        self.timestep_embedder = TimestepEmbedding(in_channels=self.flow_t_size, time_embed_dim=embedding_dim)

    def timestep_embedding(self, timesteps: mx.array, max_period: int = 10000, scale: float = 1000.0) -> mx.array:
        half = self.flow_t_size // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half) / half)
        args = timesteps[:, None] * freqs[None] * scale
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.flow_t_size % 2:
            embedding = mx.concatenate([embedding, mx.zeros((embedding.shape[0], 1))], axis=-1)
        return embedding

    def __call__(self, timestep: mx.array) -> mx.array:
        timesteps_proj = self.timestep_embedding(timestep)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNormSingleFlow(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedFlowEmbeddings(embedding_dim, size_emb_dim=embedding_dim // 3)
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def __call__(self, timestep: mx.array, batch_size: int) -> tuple[mx.array, mx.array]:
        if timestep.ndim == 0:
            timestep = mx.full((batch_size,), timestep)
        elif timestep.ndim == 1 and timestep.shape[0] != batch_size:
            timestep = mx.broadcast_to(timestep, (batch_size,))
        embedded = self.emb(timestep)
        out = self.linear(embedded * mx.sigmoid(embedded))
        return out, embedded


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, hidden_size: int, activation: str):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.act = _act_fn(activation)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = Conv1D(3 * hidden_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, hidden_size)

    def _split_heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        x = mx.reshape(x, (b, t, self.num_heads, self.head_dim))
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1, 3))
        b, t, h, d = x.shape
        return mx.reshape(x, (b, t, h * d))

    def __call__(self, hidden_states: mx.array, attention_mask: tp.Optional[mx.array]) -> mx.array:
        qkv = self.c_attn(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        # Apply rotary embedding
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        cos, sin = precompute_freqs_cis(self.head_dim, q_t.shape[1])
        q_t = apply_rotary_emb(q_t, cos, sin)
        k_t = apply_rotary_emb(k_t, cos, sin)
        q = mx.transpose(q_t, (0, 2, 1, 3))
        k = mx.transpose(k_t, (0, 2, 1, 3))
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.c_proj(out)
        return out


class GPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, inner_dim: int, activation: str):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = GPT2Attention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = GPT2MLP(inner_dim, hidden_size, activation)
        self.scale_shift_table = mx.random.normal((6, hidden_size)) / math.sqrt(hidden_size)

    def __call__(self, hidden_states: mx.array, attention_mask: tp.Optional[mx.array], time_step: mx.array) -> mx.array:
        batch = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            self.scale_shift_table[None] + mx.reshape(time_step, (batch, 6, -1)),
            6,
            axis=1,
        )
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = hidden_states * (1 + scale_msa) + shift_msa
        attn_out = self.attn(hidden_states, attention_mask)
        attn_out = attn_out * gate_msa
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = hidden_states * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(hidden_states)
        mlp_out = mlp_out * gate_mlp
        hidden_states = residual + mlp_out
        return hidden_states


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        n_inner: tp.Optional[int],
        activation_function: str,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_position_embeddings, hidden_size)
        inner_dim = n_inner if n_inner is not None else 4 * hidden_size
        self.h = ModuleList(
            [
                GPT2Block(hidden_size, num_attention_heads, inner_dim, activation_function)
                for _ in range(num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.proj_out = nn.Linear(hidden_size, hidden_size)
        self.adaln_single = AdaLayerNormSingleFlow(hidden_size)
        self.scale_shift_table = mx.random.normal((2, hidden_size)) / math.sqrt(hidden_size)

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: tp.Optional[mx.array],
        time_step: mx.array,
    ):
        batch, seq_len, _ = inputs_embeds.shape
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = (1.0 - attention_mask) * -1e4
            else:
                mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - mask) * -1e4
        time_step_emb, embedded_timestep = self.adaln_single(time_step, batch_size=batch)
        hidden_states = inputs_embeds
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask, time_step_emb)
        shift, scale = mx.split(self.scale_shift_table[None] + embedded_timestep[:, None], 2, axis=1)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        return types.SimpleNamespace(last_hidden_state=hidden_states)
