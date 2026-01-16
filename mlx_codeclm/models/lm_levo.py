"""MLX implementation of the LeVo language model."""

from __future__ import annotations

import math
import random
import typing as tp
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.models.llama import LlamaConfig, LlamaForCausalLM
from mlx_codeclm.modules.streaming import StreamingModule
from mlx_codeclm.utils.module_list import ModuleList
from mlx_codeclm.modules.conditioners import (
    ConditioningAttributes,
    AudioCondition,
    ConditionerProvider,
    ConditionFuser,
)
from mlx_codeclm.modules.pattern import CodebooksPatternProvider
from mlx_codeclm.utils.utils import sample_top_k, sample_top_p, multinomial

ConditionTensors = tp.Dict[str, tp.Tuple[mx.array, mx.array, mx.array]]


@dataclass
class LMOutput:
    logits: mx.array
    mask: mx.array


def gelu(x: mx.array) -> mx.array:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x * x * x)))


class LmModel(StreamingModule):
    def __init__(
        self,
        pattern_provider: CodebooksPatternProvider,
        condition_provider: ConditionerProvider,
        fuser: ConditionFuser,
        code_depth: int = 8,
        code_size: int = 1024,
        dim: int = 128,
        intermediate_size: int = 4096,
        num_heads: int = 8,
        num_layers: int = 16,
        num_layers_sub: int = 12,
        max_position_embeddings: int = 8196,
        max_position_embeddings_sub: int = 10000,
        rope_theta: float = 100000.0,
        rope_theta_sub: float = 500000.0,
        cfg_coef: float = 1.0,
        cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.code_size = code_size + 1
        input_emb_dim = code_size + 2
        self.code_depth = code_depth
        self.dim = dim
        self.cfg = cfg
        self.pattern_provider = pattern_provider

        self.emb = ModuleList([nn.Embedding(input_emb_dim, dim)])
        self.layer2_emb = ModuleList([nn.Embedding(input_emb_dim, dim) for _ in range(self.code_depth)])

        model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=num_heads,
            vocab_size=self.code_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-5,
            rope_theta=rope_theta,
        )
        self.transformer = LlamaForCausalLM(model_cfg)

        sub_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers_sub,
            num_key_value_heads=num_heads,
            vocab_size=self.code_size,
            max_position_embeddings=max_position_embeddings_sub,
            rms_norm_eps=1e-5,
            rope_theta=rope_theta_sub,
        )
        self.transformer2 = LlamaForCausalLM(sub_cfg)

        self.mlp = ModuleList(
            [
                nn.Linear(dim * 2, dim),
                nn.Linear(dim, dim),
            ]
        )

        if code_depth > 1:
            self.linears = ModuleList([nn.Linear(dim, self.code_size, bias=False) for _ in range(code_depth - 1)])
        else:
            self.linears = ModuleList([])

    @property
    def special_token_id(self) -> int:
        return self.code_size

    @property
    def eos_token_id(self) -> int:
        return self.code_size - 1

    def prepare_condition_tensors(
        self,
        batch_size: int = 1,
        text: tp.Optional[tp.List[str]] = None,
        descriptions: tp.Optional[tp.List[str]] = None,
        audio_qt_emb: tp.Optional[tp.List[mx.array]] = None,
        prepare_null_condition: bool = False,
    ) -> ConditionTensors:
        conditions = []
        for i in range(batch_size):
            attr = ConditioningAttributes()
            if "description" in self.condition_provider.conditioners:
                attr.text["description"] = text[i] if text is not None else ""
            if "type_info" in self.condition_provider.conditioners:
                attr.text["type_info"] = descriptions[i] if descriptions is not None else ""
            if "prompt_audio" in self.condition_provider.conditioners:
                audio_qt_seq = mx.concatenate(
                    [
                        mx.full((1, audio_qt_emb[i].shape[0], 1), self.eos_token_id, dtype=mx.int32),
                        audio_qt_emb[i][None, ...],
                    ],
                    axis=-1,
                )
                mask = (audio_qt_emb[i][None, :, 0] == 16385)[:, :, None]
                mask = mx.broadcast_to(mask, audio_qt_seq.shape)
                audio_qt_seq = mx.where(mask, mx.full(audio_qt_seq.shape, 16385, dtype=audio_qt_seq.dtype), audio_qt_seq)
                attr.audio["prompt_audio"] = AudioCondition(
                    wav=audio_qt_seq.astype(mx.int32),
                    length=mx.array([audio_qt_seq.shape[-1]], dtype=mx.int32),
                    sample_rate=[self.cfg.sample_rate],
                )
            conditions.append(attr)

        if prepare_null_condition:
            null_conditions = []
            for sample in conditions:
                null_sample = ConditioningAttributes(text=dict(sample.text), audio=dict(sample.audio))
                for key in null_sample.text.keys():
                    null_sample.text[key] = None
                for key, audio_cond in null_sample.audio.items():
                    null_sample.audio[key] = AudioCondition(
                        wav=mx.full(audio_cond.wav.shape, 16385, dtype=audio_cond.wav.dtype),
                        length=mx.array([0], dtype=mx.int32),
                        sample_rate=audio_cond.sample_rate,
                    )
                null_conditions.append(null_sample)
            conditions = conditions + null_conditions

        tokenized = self.condition_provider.tokenize(conditions)
        return self.condition_provider(tokenized)

    def __call__(self, sequence: mx.array, condition_tensors: ConditionTensors) -> mx.array:
        B, K, S = sequence.shape
        if K != self.code_depth:
            raise ValueError("Sequence shape must match code depth")
        input_1 = self.emb[0](sequence[:, 0])
        input_2 = self.layer2_emb[1](sequence[:, 1]) if K > 1 else mx.zeros_like(input_1)
        if K > 2:
            for k in range(2, K):
                input_2 = input_2 + self.layer2_emb[k](sequence[:, k])
        fused_input1, fused_input2 = self.fuser(input_1, input_2, condition_tensors)

        output = self.transformer(
            inputs_embeds=fused_input1,
            past_key_values=self._streaming_state.get("past_key_values_1"),
            use_cache=self._is_streaming,
        )
        if self._is_streaming:
            self._streaming_state["past_key_values_1"] = output.past_key_values
        logits = output.logits[:, None, :, :]

        if K > 1:
            fused_input2 = mx.concatenate([fused_input2, output.hidden_states], axis=-1)
            fused_input2 = self.mlp[1](gelu(self.mlp[0](fused_input2)))
            output2 = self.transformer2(
                inputs_embeds=fused_input2,
                past_key_values=self._streaming_state.get("past_key_values_2"),
                use_cache=self._is_streaming,
            )
            if self._is_streaming:
                self._streaming_state["past_key_values_2"] = output2.past_key_values
            res_logits = [linear(output2.hidden_states) for linear in self.linears]
            res_logits = mx.stack(res_logits, axis=1)
            logits = mx.concatenate([logits, res_logits], axis=1)

        if len(self.fuser.fuse2cond.get("prepend", [])) > 0:
            logits = logits[:, :, -S:, :]

        return logits

    def generate(
        self,
        texts=None,
        descriptions=None,
        audio_qt_embs=None,
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        record_tokens: bool = True,
        record_window: int = 150,
    ) -> mx.array:
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif texts:
            possible_num_samples.append(len(texts))
        elif audio_qt_embs is not None:
            possible_num_samples.append(int(audio_qt_embs.shape[0]))
        else:
            possible_num_samples.append(1)
        num_samples = possible_num_samples[0]

        condition_tensors = self.prepare_condition_tensors(
            batch_size=num_samples,
            text=texts,
            descriptions=descriptions,
            audio_qt_emb=audio_qt_embs,
            prepare_null_condition=True,
        )

        record_token_pool = [] if record_tokens else None
        start_offset = 0
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1
        B = num_samples
        gen_codes = mx.full((B, self.code_depth, max_gen_len), unknown_token, dtype=mx.int32)
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        gen_sequence_np = np.array(gen_sequence)
        mask_np = np.array(mask)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        is_end = mx.zeros((B, self.code_depth, 1), dtype=mx.bool_)

        ignore_tokens = np.array(audio_qt_embs[0][0])
        ignore_tokens = ignore_tokens[ignore_tokens < 16384]
        ignore_tokens = mx.array(ignore_tokens)

        with self.streaming():
            gen_sequence_len = gen_sequence_np.shape[-1]
            prev_offset = 0
            for offset in tqdm(range(start_offset_sequence, gen_sequence_len)):
                curr_sequence = mx.array(gen_sequence_np[..., prev_offset:offset], dtype=mx.int32)
                next_token = self._sample_next_token(
                    curr_sequence,
                    condition_tensors,
                    use_sampling,
                    temp,
                    top_k,
                    top_p,
                    cfg_coef=cfg_coef,
                    sampled_token_pool=record_token_pool[-record_window:] if record_tokens else None,
                    ignore_tokens=ignore_tokens,
                )
                next_token_np = np.array(next_token)
                valid_mask = mask_np[..., offset:offset + 1]
                next_token_np = np.where(valid_mask, next_token_np, self.special_token_id)
                next_token_np = np.where(np.array(is_end), self.special_token_id, next_token_np)
                is_end = mx.logical_or(is_end, mx.array(next_token_np == self.eos_token_id))
                gen_sequence_np[..., offset:offset + 1] = np.where(
                    gen_sequence_np[..., offset:offset + 1] == unknown_token,
                    next_token_np,
                    gen_sequence_np[..., offset:offset + 1],
                )
                if record_tokens:
                    record_token_pool.append(mx.array(next_token_np).squeeze())
                if bool(np.all(np.array(is_end))):
                    gen_sequence_np = gen_sequence_np[..., : offset + 1]
                    break
                prev_offset = offset

        output_codes = np.full_like(gen_sequence_np, self.code_size)
        output_codes[..., : gen_sequence_np.shape[-1]] = gen_sequence_np
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(mx.array(output_codes), special_token=unknown_token)
        return out_codes

    def _sample_next_token(
        self,
        sequence: mx.array,
        condition_tensors: ConditionTensors,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        sampled_token_pool: tp.Optional[list] = None,
        ignore_tokens: tp.Optional[mx.array] = None,
    ) -> mx.array:
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        sequence = mx.concatenate([sequence, sequence], axis=0)
        all_logits = self(sequence, condition_tensors=condition_tensors)
        cond_logits = all_logits[:B]
        uncond_logits = all_logits[B:]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
        logits = mx.transpose(logits, (0, 1, 3, 2))
        logits = logits[..., -1]

        if sampled_token_pool:
            pool_np = np.stack([np.array(x) for x in sampled_token_pool], axis=-1)
            for q in range(self.code_depth):
                uniq = np.unique(pool_np[q])
                if uniq.size == 0:
                    continue
                q_count = np.bincount(uniq.astype(np.int64))
                tmp = min(q_count.shape[-1], self.code_size - 1)
                if tmp > 0:
                    penalty = (1.1 ** q_count[:tmp]).astype(np.float32)
                    log_np = np.array(logits)
                    log_np[:, q, :tmp] = log_np[:, q, :tmp] / penalty
                    logits = mx.array(log_np)

        if ignore_tokens is not None and ignore_tokens.size > 0:
            log_np = np.array(logits)
            log_np[0, 0, np.array(ignore_tokens).astype(np.int32)] = -1e9
            logits = mx.array(log_np)

        if use_sampling and temp > 0.0:
            scaled_logits = logits / temp
            scaled_logits = scaled_logits - mx.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = mx.exp(scaled_logits)
            probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)
            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token_first = sample_top_k(probs[:, [0], :], k=top_k)
                next_token_res = sample_top_k(probs[:, 1:, :], k=1)
                next_token = mx.concatenate([next_token_first, next_token_res], axis=1)
            else:
                next_token = multinomial(probs, num_samples=1)
        else:
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
        return next_token
