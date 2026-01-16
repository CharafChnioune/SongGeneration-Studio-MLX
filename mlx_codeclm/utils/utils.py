"""Utility helpers for MLX inference."""

import typing as tp
import numpy as np
import mlx.core as mx


def _sanitize_prob_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    if not np.isfinite(row).all():
        row = np.where(np.isfinite(row), row, 0.0)
    total = row.sum()
    if not np.isfinite(total) or total <= 0:
        row = np.ones_like(row, dtype=np.float64)
        total = row.sum()
    return row / total


def length_to_mask(lengths: mx.array, max_len: tp.Optional[int] = None) -> mx.array:
    if lengths.ndim != 1:
        raise ValueError("lengths must be 1D")
    final_length = int(mx.max(lengths).item()) if max_len is None else int(max_len)
    final_length = max(final_length, 1)
    rng = mx.arange(final_length)[None, :]
    return rng < lengths[:, None]


def multinomial(probs: mx.array, num_samples: int = 1) -> mx.array:
    probs_np = np.array(probs)
    flat = probs_np.reshape(-1, probs_np.shape[-1])
    out = []
    for row in flat:
        row = _sanitize_prob_row(row)
        out.append(np.random.choice(len(row), size=num_samples, p=row))
    out = np.array(out, dtype=np.int64)
    out = out.reshape(*probs_np.shape[:-1], num_samples)
    return mx.array(out)


def sample_top_k(probs: mx.array, k: int) -> mx.array:
    if k <= 0:
        return multinomial(probs, num_samples=1)
    topk = mx.sort(probs, axis=-1)[:, :, -k:]
    min_val = topk[:, :, :1]
    mask = probs >= min_val
    probs = probs * mask
    probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    return multinomial(probs, num_samples=1)


def sample_top_p(probs: mx.array, p: float) -> mx.array:
    if p <= 0.0:
        return multinomial(probs, num_samples=1)
    probs_np = np.array(probs)
    flat = probs_np.reshape(-1, probs_np.shape[-1])
    out = []
    for row in flat:
        row = _sanitize_prob_row(row)
        idx = np.argsort(-row)
        sorted_probs = row[idx]
        cumsum = np.cumsum(sorted_probs)
        mask = cumsum - sorted_probs > p
        sorted_probs[mask] = 0.0
        sorted_probs = _sanitize_prob_row(sorted_probs)
        sampled = np.random.choice(len(sorted_probs), size=1, p=sorted_probs)
        out.append(idx[sampled])
    out = np.array(out, dtype=np.int64)
    out = out.reshape(*probs_np.shape[:-1], 1)
    return mx.array(out)
