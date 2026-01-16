"""Conditioner modules for MLX inference."""

import typing as tp
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import chain
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.utils.utils import length_to_mask
from mlx_codeclm.modules.streaming import StreamingModule
from mlx_codeclm.utils.module_list import ModuleList

ConditionType = tp.Tuple[mx.array, mx.array, mx.array]


class AudioCondition(tp.NamedTuple):
    wav: mx.array
    length: mx.array
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    audio: tp.Dict[str, AudioCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def audio_attributes(self):
        return self.audio.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "audio": self.audio_attributes,
        }


class BaseConditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, input_token: bool = False, padding_idx: int = 0):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if input_token:
            self.output_proj = nn.Embedding(dim, output_dim)
        else:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TextConditioner(BaseConditioner):
    pass


class QwTokenizerConditioner(TextConditioner):
    def __init__(self, output_dim: int, token_path: str = "", max_len: int = 300, add_token_list: tp.List[str] = None):
        import os
        from transformers import Qwen2Tokenizer
        add_token_list = add_token_list or []
        if token_path and os.path.isdir(token_path):
            self.text_tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
        else:
            repo_id = "Qwen/Qwen2-7B"
            self.text_tokenizer = Qwen2Tokenizer.from_pretrained(repo_id)
        if add_token_list:
            self.text_tokenizer.add_tokens(add_token_list, special_tokens=True)
        vocab_size = len(self.text_tokenizer.get_vocab())
        super().__init__(vocab_size, output_dim, input_token=True, padding_idx=151643)
        self.max_len = max_len
        self.padding_idx = 151643

        vocab = self.text_tokenizer.get_vocab()
        struct_tokens = [tok for tok in add_token_list if tok.startswith("[") and tok.endswith("]")]
        self.struct_token_ids = [vocab[tok] for tok in struct_tokens]
        self.pad_token_idx = 151643
        self.structure_emb = nn.Embedding(200, output_dim)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, np.ndarray]:
        x = ["<|im_start|>" + xi if xi is not None else "<|im_start|>" for xi in x]
        return self.text_tokenizer(x, return_tensors="np", padding=True)

    def forward(self, inputs: tp.Dict[str, np.ndarray]) -> ConditionType:
        tokens = mx.array(inputs["input_ids"], dtype=mx.int32)
        mask = mx.array(inputs["attention_mask"], dtype=mx.int32)
        tokens_np = np.array(tokens)
        mask_np = np.array(mask)
        tp_cover_range = np.zeros_like(tokens_np)
        if self.struct_token_ids:
            struct_ids = np.array(self.struct_token_ids)
            for b in range(tokens_np.shape[0]):
                is_sp = np.isin(tokens_np[b], struct_ids)
                sp_list = np.where(is_sp)[0].tolist()
                sp_list.append(int(mask_np[b].sum()))
                for i, st in enumerate(sp_list[:-1]):
                    tp_cover_range[b, st:sp_list[i + 1]] = tokens_np[b, st] - 151645

        if self.max_len is not None:
            tokens = self._pad_2d_tensor(tokens, self.max_len, self.pad_token_idx)
            mask = self._pad_2d_tensor(mask, self.max_len, 0)
            tp_cover_range = self._pad_2d_tensor(mx.array(tp_cover_range), self.max_len, 0)

        content_embeds = self.output_proj(tokens)
        structure_embeds = self.structure_emb(tp_cover_range)
        embeds = content_embeds + structure_embeds
        return embeds, embeds, mask

    def _pad_2d_tensor(self, x: mx.array, max_len: int, pad_id: int) -> mx.array:
        batch_size, seq_len = x.shape
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad = mx.full((batch_size, pad_len), pad_id, dtype=x.dtype)
            return mx.concatenate([x, pad], axis=1)
        if pad_len < 0:
            return x[:, :max_len]
        return x


class QwTextConditioner(TextConditioner):
    def __init__(self, output_dim: int, token_path: str = "", max_len: int = 300):
        import os
        from transformers import Qwen2Tokenizer
        if token_path and os.path.isdir(token_path):
            self.text_tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
        else:
            repo_id = "Qwen/Qwen2-7B"
            self.text_tokenizer = Qwen2Tokenizer.from_pretrained(repo_id)
        vocab_size = len(self.text_tokenizer.get_vocab())
        super().__init__(vocab_size, output_dim, input_token=True, padding_idx=151643)
        self.max_len = max_len

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, np.ndarray]:
        x = ["<|im_start|>" + xi if xi is not None else "<|im_start|>" for xi in x]
        return self.text_tokenizer(x, return_tensors="np", padding=True)

    def forward(self, inputs: tp.Dict[str, np.ndarray]) -> ConditionType:
        tokens = mx.array(inputs["input_ids"], dtype=mx.int32)
        mask = mx.array(inputs["attention_mask"], dtype=mx.int32)
        if self.max_len is not None:
            tokens = self._pad_2d_tensor(tokens, self.max_len, 151643)
            mask = self._pad_2d_tensor(mask, self.max_len, 0)
        embeds = self.output_proj(tokens)
        return embeds, embeds, mask

    def _pad_2d_tensor(self, x: mx.array, max_len: int, pad_id: int) -> mx.array:
        batch_size, seq_len = x.shape
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad = mx.full((batch_size, pad_len), pad_id, dtype=x.dtype)
            return mx.concatenate([x, pad], axis=1)
        if pad_len < 0:
            return x[:, :max_len]
        return x


class AudioConditioner(BaseConditioner):
    pass


class QuantizedEmbeddingConditioner(AudioConditioner):
    def __init__(self, dim: int, code_size: int, code_depth: int, max_len: int, **kwargs):
        super().__init__(dim, dim, input_token=True)
        self.code_depth = code_depth
        self.emb = ModuleList([nn.Embedding(code_size + 2, dim) for _ in range(code_depth)])
        self.EOT_emb = mx.random.normal((1, dim))
        self.layer2_EOT_emb = mx.random.normal((1, dim))
        self.output_proj = None
        self.max_len = max_len
        self.vocab_size = code_size

    def tokenize(self, x: AudioCondition) -> AudioCondition:
        return x

    def forward(self, x: AudioCondition):
        wav, lengths, *_ = x
        B = wav.shape[0]
        wav = wav.reshape((B, self.code_depth, -1)).astype(mx.int32)
        if wav.shape[2] < self.max_len - 1:
            pad = mx.full((B, self.code_depth, self.max_len - 1 - wav.shape[2]), self.vocab_size + 1, dtype=wav.dtype)
            wav = mx.concatenate([wav, pad], axis=2)
        else:
            wav = wav[:, :, : self.max_len - 1]

        embeds1 = self.emb[0](wav[:, 0])
        eot = mx.broadcast_to(self.EOT_emb[None, :, :], (B, 1, self.EOT_emb.shape[-1]))
        embeds1 = mx.concatenate([eot, embeds1], axis=1)

        embeds2 = None
        if self.code_depth > 1:
            embeds2 = self.emb[1](wav[:, 1])
            for k in range(2, self.code_depth):
                embeds2 = embeds2 + self.emb[k](wav[:, k])
        else:
            embeds2 = mx.zeros_like(embeds1[:, 1:])
        eot2 = mx.broadcast_to(self.layer2_EOT_emb[None, :, :], (B, 1, self.layer2_EOT_emb.shape[-1]))
        embeds2 = mx.concatenate([eot2, embeds2], axis=1)

        lengths = lengths + 1
        lengths = mx.minimum(lengths, self.max_len)
        mask = length_to_mask(lengths, max_len=embeds1.shape[1]).astype(mx.int32)
        return embeds1, embeds2, mask


class ConditionerProvider(nn.Module):
    def __init__(self, conditioners: tp.Dict[str, BaseConditioner]):
        super().__init__()
        self.conditioners = conditioners
        self._conditioner_keys = list(conditioners.keys())
        for key, module in conditioners.items():
            setattr(self, key, module)

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def audio_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, AudioConditioner)]

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        if not all(isinstance(x, ConditioningAttributes) for x in inputs):
            raise ValueError("Unexpected input types for conditioner")
        output = {}
        text = self._collate_text(inputs)
        audios = self._collate_audios(inputs)
        for attribute, batch in chain(text.items(), audios.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        output = {}
        for attribute, inputs in tokenized.items():
            condition1, condition2, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition1, condition2, mask)
        return output

    def __call__(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        return self.forward(tokenized)

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        for sample in samples:
            for condition in self.text_conditions:
                out[condition].append(sample.text.get(condition))
        return out

    def _collate_audios(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, AudioCondition]:
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, AudioCondition] = {}

        for sample in samples:
            for attribute in self.audio_conditions:
                wav, length, sample_rate, path, seek_time = sample.audio[attribute]
                if wav.ndim != 3:
                    raise ValueError("Expected wav [1, C, T]")
                if wav.shape[0] != 1:
                    raise ValueError("Expected B == 1 for wav")
                wavs[attribute].append(wav.reshape((-1,)))
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        for attribute in self.audio_conditions:
            items = wavs[attribute]
            max_len = max(int(x.shape[0]) for x in items)
            padded = []
            for item in items:
                pad_len = max_len - int(item.shape[0])
                if pad_len > 0:
                    pad = mx.zeros((pad_len,), dtype=item.dtype)
                    item = mx.concatenate([item, pad], axis=0)
                padded.append(item)
            stacked = mx.stack(padded, axis=0)
            length_tensor = mx.concatenate(lengths[attribute], axis=0)
            out[attribute] = AudioCondition(
                stacked[:, None, :],
                length_tensor,
                sample_rates[attribute],
                paths[attribute],
                seek_times[attribute],
            )
        return out


class ConditionFuser(StreamingModule):
    FUSING_METHODS = ["sum", "prepend"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]]):
        super().__init__()
        self.fuse2cond = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def __call__(self, input1: mx.array, input2: mx.array, conditions: tp.Dict[str, ConditionType]):
        if not set(conditions.keys()).issubset(set(self.cond2fuse.keys())):
            raise ValueError("Unknown conditioning attributes for fuser")
        if "offsets" in self._streaming_state:
            first_step = False
            offsets = self._streaming_state["offsets"]
        else:
            first_step = True
            offsets = mx.zeros((input1.shape[0],), dtype=mx.int32)
        fused_input_1 = input1
        fused_input_2 = input2
        for fuse_op in self.fuse2cond.keys():
            fuse_op_conditions = self.fuse2cond[fuse_op]
            if fuse_op == "sum" and fuse_op_conditions:
                for cond in fuse_op_conditions:
                    this_cond_1, this_cond_2, _ = conditions[cond]
                    fused_input_1 = fused_input_1 + this_cond_1
                    fused_input_2 = fused_input_2 + this_cond_2
            elif fuse_op == "prepend" and fuse_op_conditions:
                if not first_step:
                    continue
                for cond in reversed(fuse_op_conditions):
                    this_cond_1, this_cond_2, _ = conditions[cond]
                    fused_input_1 = mx.concatenate([this_cond_1, fused_input_1], axis=1)
                    fused_input_2 = mx.concatenate([this_cond_2, fused_input_2], axis=1)
            elif fuse_op not in self.FUSING_METHODS:
                raise ValueError(f"Unknown fuse op {fuse_op}")
        if self._is_streaming:
            self._streaming_state["offsets"] = offsets + input1.shape[1]
        return fused_input_1, fused_input_2
