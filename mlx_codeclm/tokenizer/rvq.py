"""MLX port of Residual Vector Quantization used by Flow1dVAE."""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.utils.module_list import ModuleList
from mlx_codeclm.tokenizer.stable_audio_vae import WNConv1d


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stale_tolerance: int = 100):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.stale_tolerance = stale_tolerance

    def decode_code(self, embed_id: mx.array) -> mx.array:
        # embed_id: [B, T]
        flat = mx.reshape(embed_id, (-1,))
        embeds = mx.take(self.codebook.weight, flat, axis=0)
        embeds = mx.reshape(embeds, (embed_id.shape[0], embed_id.shape[1], self.codebook_dim))
        return mx.transpose(embeds, (0, 2, 1))

    def decode_latents(self, latents: mx.array) -> tuple[mx.array, mx.array]:
        # latents: [B, D, T]
        enc = mx.transpose(latents, (0, 2, 1))
        enc = mx.reshape(enc, (-1, enc.shape[-1]))
        enc_norm = enc / (mx.sqrt(mx.sum(enc * enc, axis=1, keepdims=True)) + 1e-8)
        codebook = self.codebook.weight
        code_norm = codebook / (mx.sqrt(mx.sum(codebook * codebook, axis=1, keepdims=True)) + 1e-8)
        dist = (
            mx.sum(enc_norm * enc_norm, axis=1, keepdims=True)
            - 2 * enc_norm @ mx.transpose(code_norm)
            + mx.sum(code_norm * code_norm, axis=1, keepdims=True).T
        )
        indices = mx.argmax(-dist, axis=1)
        indices = mx.reshape(indices, (latents.shape[0], latents.shape[2]))
        z_q = self.decode_code(indices)
        return z_q, indices

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = mx.mean((z_e - mx.stop_gradient(z_q)) ** 2, axis=(1, 2))
        codebook_loss = mx.mean((mx.stop_gradient(z_e) - z_q) ** 2, axis=(1, 2))
        z_q = z_e + mx.stop_gradient(z_q - z_e)
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: tp.Union[int, tp.Sequence[int]] = 8,
        quantizer_dropout: float = 0.0,
        stale_tolerance: int = 100,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = list(codebook_dim)
        self.codebook_size = codebook_size
        self.quantizers = ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, self.codebook_dim[i], stale_tolerance=stale_tolerance)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def __call__(self, z: mx.array, n_quantizers: tp.Optional[int] = None):
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        z_q = mx.zeros_like(z)
        residual = z
        commitment_loss = 0.0
        codebook_loss = 0.0
        codebook_indices = []
        latents = []

        for i, quantizer in enumerate(self.quantizers):
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            if i >= n_quantizers:
                continue
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            commitment_loss = commitment_loss + mx.mean(commitment_loss_i)
            codebook_loss = codebook_loss + mx.mean(codebook_loss_i)
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = mx.stack(codebook_indices, axis=1)
        latents = mx.concatenate(latents, axis=1) if latents else mx.zeros_like(z)
        return z_q, codes, latents, commitment_loss, codebook_loss, n_quantizers - 1

    def from_codes(self, codes: mx.array):
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, mx.concatenate(z_p, axis=1), codes
