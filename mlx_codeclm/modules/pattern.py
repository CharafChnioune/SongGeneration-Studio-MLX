"""Pattern utilities for interleaving codebooks in MLX."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import logging
import typing as tp

import numpy as np
import mlx.core as mx

LayoutCoord = namedtuple("LayoutCoord", ["t", "q"])  # (timestep, codebook index)
PatternLayout = tp.List[tp.List[LayoutCoord]]
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    layout: PatternLayout
    timesteps: int
    code_depth: int

    def __post_init__(self) -> None:
        if not self.layout:
            raise ValueError("Pattern layout is empty")
        if self.layout[0] != []:
            raise ValueError("Pattern layout must start with an empty list")
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(self._build_reverted_sequence_scatter_indexes)
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(self._build_pattern_sequence_scatter_indexes)
        logger.info("New pattern, time steps: %d, sequence steps: %d", self.timesteps, len(self.layout))

    def _validate_layout(self) -> None:
        q_timesteps = {q: 0 for q in range(self.code_depth)}
        for s, seq_coords in enumerate(self.layout):
            if not seq_coords:
                continue
            qs = set()
            for coord in seq_coords:
                qs.add(coord.q)
                last_q_timestep = q_timesteps[coord.q]
                q_timesteps[coord.q] = coord.t
            if len(qs) != len(seq_coords):
                raise ValueError(f"Multiple entries for a codebook at step {s}")

    @property
    def num_sequence_steps(self) -> int:
        return len(self.layout) - 1

    @property
    def max_delay(self) -> int:
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self) -> PatternLayout:
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def get_sequence_coords_with_timestep(self, t: int, q: tp.Optional[int] = None):
        if t > self.timesteps:
            raise ValueError("t exceeds pattern timesteps")
        if q is not None and q > self.code_depth:
            raise ValueError("q exceeds pattern code depth")
        coords = []
        for s, seq_codes in enumerate(self.layout):
            for code in seq_codes:
                if code.t == t and (q is None or code.q == q):
                    coords.append((s, code))
        return coords

    def get_steps_with_timestep(self, t: int, q: tp.Optional[int] = None) -> tp.List[int]:
        return [step for step, _coords in self.get_sequence_coords_with_timestep(t, q)]

    def get_first_step_with_timesteps(self, t: int, q: tp.Optional[int] = None) -> tp.Optional[int]:
        steps = self.get_steps_with_timestep(t, q)
        return steps[0] if steps else None

    def _build_pattern_sequence_scatter_indexes(
        self,
        timesteps: int,
        code_depth: int,
        keep_only_valid_steps: bool,
    ) -> tp.Tuple[mx.array, mx.array]:
        if code_depth != self.code_depth:
            raise ValueError("invalid code depth for pattern sequence")
        if timesteps > self.timesteps:
            raise ValueError("timesteps exceed pattern")
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        indexes = np.zeros((code_depth, len(ref_layout)), dtype=np.int64)
        mask = np.zeros((code_depth, len(ref_layout)), dtype=np.bool_)
        indexes[:] = code_depth * timesteps
        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = True
        return mx.array(indexes), mx.array(mask)

    def build_pattern_sequence(self, z: mx.array, special_token: int, keep_only_valid_steps: bool = False):
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps
        )
        z = z.reshape((B, -1))
        pad = mx.full((B, 1), special_token, dtype=z.dtype)
        z = mx.concatenate([z, pad], axis=1)
        flat_idx = indexes.reshape((-1,))
        values = mx.take(z, flat_idx, axis=1)
        values = values.reshape((B, K, indexes.shape[-1]))
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(
        self,
        sequence_steps: int,
        code_depth: int,
        keep_only_valid_steps: bool = False,
        is_model_output: bool = False,
    ) -> tp.Tuple[mx.array, mx.array]:
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        timesteps = self.timesteps
        if code_depth != self.code_depth:
            raise ValueError("invalid code depth for pattern sequence")
        if sequence_steps > len(ref_layout):
            raise ValueError("sequence to revert is longer than pattern")

        if is_model_output:
            ref_layout = ref_layout[1:]

        indexes = np.zeros((code_depth, timesteps), dtype=np.int64)
        mask = np.zeros((code_depth, timesteps), dtype=np.bool_)
        indexes[:] = code_depth * sequence_steps
        for s, sequence_coords in enumerate(ref_layout):
            if s < sequence_steps:
                for coords in sequence_coords:
                    if coords.t < timesteps:
                        indexes[coords.q, coords.t] = s + coords.q * sequence_steps
                        mask[coords.q, coords.t] = True
        return mx.array(indexes), mx.array(mask)

    def revert_pattern_sequence(
        self,
        sequence: mx.array,
        special_token: int,
        keep_only_valid_steps: bool = False,
    ):
        B, K, S = sequence.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S,
            K,
            keep_only_valid_steps=keep_only_valid_steps,
            is_model_output=False,
        )
        seq_flat = sequence.reshape((B, -1))
        pad = mx.full((B, 1), special_token, dtype=sequence.dtype)
        seq_flat = mx.concatenate([seq_flat, pad], axis=1)
        flat_idx = indexes.reshape((-1,))
        values = mx.take(seq_flat, flat_idx, axis=1)
        values = values.reshape((B, K, indexes.shape[-1]))
        return values, indexes, mask


class CodebooksPatternProvider:
    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError


class DelayedPatternProvider(CodebooksPatternProvider):
    def __init__(self, code_depth: int, delays: tp.List[int], flatten_first: int = 0, empty_initial: int = 0):
        self.code_depth = code_depth
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial

    def get_pattern(self, timesteps: int) -> Pattern:
        layout: PatternLayout = [[] for _ in range(self.empty_initial + 1)]
        for t in range(timesteps + max(self.delays)):
            step: tp.List[LayoutCoord] = []
            for q in range(self.code_depth):
                if t - self.delays[q] >= 0 and t - self.delays[q] < timesteps:
                    step.append(LayoutCoord(t - self.delays[q], q))
            layout.append(step)
        return Pattern(layout=layout, timesteps=timesteps, code_depth=self.code_depth)
