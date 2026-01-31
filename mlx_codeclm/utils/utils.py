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


def _sanitize_probs(probs: mx.array) -> mx.array:
    """Ensure probabilities are finite, non-negative, and normalized."""
    if not hasattr(mx, "isfinite"):
        probs_np = np.array(probs, dtype=np.float64)
        probs_np = np.where(np.isfinite(probs_np), probs_np, 0.0)
        probs_np = np.maximum(probs_np, 0.0)
        denom = probs_np.sum(axis=-1, keepdims=True)
        vocab = probs_np.shape[-1] if probs_np.ndim > 0 else 0
        if vocab <= 0:
            return mx.array(probs_np)
        uniform = np.full_like(probs_np, 1.0 / float(vocab))
        probs_np = np.where(denom > 0, probs_np / denom, uniform)
        return mx.array(probs_np)

    probs = probs.astype(mx.float32)
    probs = mx.where(mx.isfinite(probs), probs, 0.0)
    probs = mx.maximum(probs, 0.0)
    denom = mx.sum(probs, axis=-1, keepdims=True)
    vocab = probs.shape[-1] if probs.ndim > 0 else 0
    if vocab <= 0:
        return probs
    uniform = mx.full_like(probs, 1.0 / float(vocab))
    probs = mx.where(denom > 0, probs / denom, uniform)
    return probs


def length_to_mask(lengths: mx.array, max_len: tp.Optional[int] = None) -> mx.array:
    if lengths.ndim != 1:
        raise ValueError("lengths must be 1D")
    final_length = int(mx.max(lengths).item()) if max_len is None else int(max_len)
    final_length = max(final_length, 1)
    rng = mx.arange(final_length)[None, :]
    return rng < lengths[:, None]


def multinomial(probs: mx.array, num_samples: int = 1) -> mx.array:
    if not hasattr(mx.random, "categorical"):
        probs_np = np.array(probs)
        flat = probs_np.reshape(-1, probs_np.shape[-1])
        out = []
        for row in flat:
            row = _sanitize_prob_row(row)
            out.append(np.random.choice(len(row), size=num_samples, p=row))
        out = np.array(out, dtype=np.int64)
        out = out.reshape(*probs_np.shape[:-1], num_samples)
        return mx.array(out)
    probs = _sanitize_probs(probs)
    logits = mx.log(probs)
    samples = mx.random.categorical(logits, axis=-1, num_samples=num_samples)
    return samples.astype(mx.int32)


def sample_top_k(probs: mx.array, k: int) -> mx.array:
    if k <= 0:
        return multinomial(probs, num_samples=1)
    topk = mx.sort(probs, axis=-1)[:, :, -k:]
    min_val = topk[:, :, :1]
    mask = probs >= min_val
    probs = _sanitize_probs(probs * mask)
    return multinomial(probs, num_samples=1)


def sample_top_p(probs: mx.array, p: float) -> mx.array:
    if p <= 0.0:
        return multinomial(probs, num_samples=1)
    if not (hasattr(mx, "argsort") and hasattr(mx, "take_along_axis") and hasattr(mx, "cumsum")):
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

    probs = _sanitize_probs(probs)
    vocab = probs.shape[-1]
    if vocab <= 0:
        return mx.zeros(probs.shape[:-1] + (1,), dtype=mx.int32)
    sorted_idx = mx.argsort(probs, axis=-1)
    rev = mx.arange(vocab - 1, -1, -1)
    sorted_idx = mx.take(sorted_idx, rev, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    mask = (cumsum - sorted_probs) > p
    sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)
    sorted_probs = _sanitize_probs(sorted_probs)
    sampled_sorted = multinomial(sorted_probs, num_samples=1)
    return mx.take_along_axis(sorted_idx, sampled_sorted, axis=-1)
