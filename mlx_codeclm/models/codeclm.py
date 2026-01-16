"""Main MLX model wrapper for SongGeneration."""

import typing as tp
import mlx.core as mx

from mlx_codeclm.models.lm_levo import LmModel
from mlx_codeclm.tokenizer.audio_tokenizer import AudioTokenizer

MelodyList = tp.List[tp.Optional[mx.array]]


class CodecLM:
    def __init__(
        self,
        name: str,
        audiotokenizer: tp.Optional[AudioTokenizer],
        lm: LmModel,
        max_duration: tp.Optional[float] = None,
        seperate_tokenizer: tp.Optional[AudioTokenizer] = None,
    ):
        self.name = name
        self.audiotokenizer = audiotokenizer
        self.seperate_tokenizer = seperate_tokenizer
        self.frame_rate = self.audiotokenizer.frame_rate if self.audiotokenizer else 25
        self.lm = lm
        if max_duration is None:
            raise ValueError("max_duration is required for MLX CodecLM")
        self.max_duration = max_duration
        self.duration = 15.0
        self.extend_stride = self.max_duration // 2
        self.generation_params = {}
        self._progress_callback = None

    def set_generation_params(
        self,
        use_sampling: bool = True,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        duration: float = 30.0,
        cfg_coef: float = 3.0,
        extend_stride: float = 18,
        record_tokens: bool = False,
        record_window: int = 50,
    ):
        if extend_stride > self.max_duration:
            raise ValueError("extend_stride exceeds max_duration")
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            "use_sampling": use_sampling,
            "temp": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_coef": cfg_coef,
            "record_tokens": record_tokens,
            "record_window": record_window,
        }

    def generate(
        self,
        lyrics: tp.List[str],
        descriptions: tp.List[str],
        melody_wavs: tp.Optional[mx.array] = None,
        melody_is_wav: bool = True,
        vocal_wavs: tp.Optional[mx.array] = None,
        bgm_wavs: tp.Optional[mx.array] = None,
        return_tokens: bool = False,
    ) -> tp.Union[mx.array, tp.Tuple[mx.array, mx.array]]:
        if melody_wavs is not None and melody_wavs.ndim == 2:
            melody_wavs = melody_wavs[None, ...]
        if vocal_wavs is not None and vocal_wavs.ndim == 2:
            vocal_wavs = vocal_wavs[None, ...]
        if bgm_wavs is not None and bgm_wavs.ndim == 2:
            bgm_wavs = bgm_wavs[None, ...]

        texts, audio_qt_embs = self._prepare_tokens_and_attributes(
            lyrics=lyrics,
            melody_wavs=melody_wavs,
            vocal_wavs=vocal_wavs,
            bgm_wavs=bgm_wavs,
            melody_is_wav=melody_is_wav,
        )
        tokens = self._generate_tokens(texts, descriptions, audio_qt_embs)
        if return_tokens:
            return tokens
        return self.generate_audio(tokens)

    def _prepare_tokens_and_attributes(
        self,
        lyrics: tp.Sequence[tp.Optional[str]],
        melody_wavs: tp.Optional[MelodyList] = None,
        vocal_wavs: tp.Optional[MelodyList] = None,
        bgm_wavs: tp.Optional[MelodyList] = None,
        melody_is_wav: bool = True,
    ) -> tp.Tuple[tp.List[str], mx.array]:
        if len(lyrics) != 1:
            raise ValueError("Only batch size 1 is supported in MLX path")
        texts = [lyric for lyric in lyrics]
        prompt_len = getattr(self.lm.cfg, "prompt_len", None)
        if prompt_len is None and hasattr(self.lm.cfg, "lyric_processor"):
            prompt_len = self.lm.cfg.lyric_processor.prompt_len
        if prompt_len is None:
            raise ValueError("prompt_len not found in config")
        target_len = int(prompt_len * self.frame_rate)
        if melody_wavs is None:
            melody_tokens = mx.full((1, 1, target_len), 16385, dtype=mx.int32)
        else:
            if melody_is_wav:
                melody_tokens, _ = self.audiotokenizer.encode(melody_wavs)
            else:
                melody_tokens = melody_wavs
            if melody_tokens.shape[-1] > target_len:
                melody_tokens = melody_tokens[..., :target_len]
            elif melody_tokens.shape[-1] < target_len:
                pad = mx.full((1, 1, target_len - melody_tokens.shape[-1]), 16385, dtype=mx.int32)
                melody_tokens = mx.concatenate([melody_tokens, pad], axis=-1)

        if bgm_wavs is None:
            if vocal_wavs is not None:
                raise ValueError("vocal_wavs provided without bgm_wavs")
            bgm_tokens = mx.full((1, 1, target_len), 16385, dtype=mx.int32)
            vocal_tokens = mx.full((1, 1, target_len), 16385, dtype=mx.int32)
        else:
            if melody_is_wav:
                vocal_tokens, bgm_tokens = self.seperate_tokenizer.encode(vocal_wavs, bgm_wavs)
            else:
                vocal_tokens, bgm_tokens = vocal_wavs, bgm_wavs
            if bgm_tokens.shape[-1] > target_len:
                bgm_tokens = bgm_tokens[..., :target_len]
            elif bgm_tokens.shape[-1] < target_len:
                pad = mx.full((1, 1, target_len - bgm_tokens.shape[-1]), 16385, dtype=mx.int32)
                bgm_tokens = mx.concatenate([bgm_tokens, pad], axis=-1)
            if vocal_tokens.shape[-1] > target_len:
                vocal_tokens = vocal_tokens[..., :target_len]
            elif vocal_tokens.shape[-1] < target_len:
                pad = mx.full((1, 1, target_len - vocal_tokens.shape[-1]), 16385, dtype=mx.int32)
                vocal_tokens = mx.concatenate([vocal_tokens, pad], axis=-1)

        melody_tokens = mx.concatenate([melody_tokens, vocal_tokens, bgm_tokens], axis=1)
        return texts, melody_tokens.astype(mx.int32)

    def _generate_tokens(
        self,
        texts: tp.Optional[tp.List[str]] = None,
        descriptions: tp.Optional[tp.List[str]] = None,
        audio_qt_embs: tp.Optional[mx.array] = None,
    ) -> mx.array:
        total_gen_len = int(self.duration * self.frame_rate)
        if self.duration > self.max_duration:
            raise NotImplementedError("extend generation is not implemented")
        gen_tokens = self.lm.generate(
            texts=texts,
            descriptions=descriptions,
            audio_qt_embs=audio_qt_embs,
            max_gen_len=total_gen_len,
            **self.generation_params,
        )
        return gen_tokens

    def generate_audio(
        self,
        gen_tokens: mx.array,
        gen_type: str = "mixed",
        prompt: tp.Optional[mx.array] = None,
        vocal_prompt: tp.Optional[mx.array] = None,
        bgm_prompt: tp.Optional[mx.array] = None,
        *args,
        **kwargs,
    ) -> mx.array:
        if gen_tokens.ndim == 2:
            gen_tokens = gen_tokens[:, None, :]
        if prompt is not None:
            prompt = mx.array(prompt)
        if vocal_prompt is not None:
            vocal_prompt = mx.array(vocal_prompt)
        if bgm_prompt is not None:
            bgm_prompt = mx.array(bgm_prompt)
        decode_duration = kwargs.pop("duration", None)
        if decode_duration is None:
            decode_duration = 40.0
        if self.seperate_tokenizer is not None:
            if gen_tokens.shape[1] < 3:
                raise ValueError("Expected 3 codebooks for separate tokenizer output")
            gen_tokens_vocal = gen_tokens[:, [1], :]
            gen_tokens_bgm = gen_tokens[:, [2], :]
            if gen_type == "bgm":
                gen_tokens_vocal = mx.full(gen_tokens_vocal.shape, 3142, dtype=gen_tokens_vocal.dtype)
                if vocal_prompt is not None:
                    vocal_prompt = mx.zeros_like(vocal_prompt)
            elif gen_type == "vocal":
                gen_tokens_bgm = mx.full(gen_tokens_bgm.shape, 9670, dtype=gen_tokens_bgm.dtype)
                if bgm_prompt is not None:
                    bgm_prompt = mx.zeros_like(bgm_prompt)
            elif gen_type != "mixed":
                raise ValueError(f"Unsupported gen_type {gen_type}")
            kwargs = dict(kwargs)
            kwargs.setdefault("duration", decode_duration)
            return self.seperate_tokenizer.decode([gen_tokens_vocal, gen_tokens_bgm], vocal_prompt, bgm_prompt, *args, **kwargs)
        if self.audiotokenizer is None:
            raise ValueError("Audio tokenizer is not configured")
        mixed_tokens = gen_tokens[:, [0], :] if gen_tokens.shape[1] > 1 else gen_tokens
        kwargs = dict(kwargs)
        kwargs.setdefault("duration", decode_duration)
        return self.audiotokenizer.decode(mixed_tokens, prompt=prompt, *args, **kwargs)
