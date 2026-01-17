"""MLX inference runner for SongGeneration."""

import argparse
import json
import os
import time
import random
import subprocess
import sys
import gc
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from mlx_codeclm.models import builders
from mlx_codeclm.models.codeclm import CodecLM
from mlx_codeclm.utils.weights import load_weights_npz
from mlx_codeclm.audio_io import save_audio
from mlx_codeclm.separator import create_separator


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio, 2, axis=0)
        if audio.shape[0] > 2:
            return audio[:2]
        return audio
    raise ValueError(f"Unsupported audio shape {audio.shape}")


def load_audio(path: str, target_sr: int, max_seconds: Optional[float] = None) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.T
    if sr != target_sr:
        resampled = []
        for ch in audio:
            resampled.append(librosa.resample(ch.astype(np.float32), orig_sr=sr, target_sr=target_sr))
        max_len = max(len(ch) for ch in resampled)
        resampled = [
            np.pad(ch, (0, max_len - len(ch)), mode="constant") if len(ch) < max_len else ch
            for ch in resampled
        ]
        audio = np.stack(resampled, axis=0)
    audio = ensure_stereo(audio)
    if max_seconds:
        max_samples = int(target_sr * max_seconds)
        audio = audio[:, :max_samples]
    return audio.astype(np.float32)


def split_prompt_audio(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2]
    else:
        raise ValueError(f"Unsupported prompt audio shape {audio.shape}")
    left, right = audio[0], audio[1]
    mid = 0.5 * (left + right)
    vocal = np.stack([mid, mid], axis=0)
    bgm = np.stack([left - mid, right - mid], axis=0)
    return vocal.astype(np.float32), bgm.astype(np.float32)


def save_audio_outputs(
    flac_path: str,
    audio: np.ndarray,
    sample_rate: int,
    max_seconds: Optional[float] = None,
) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    if max_seconds is not None:
        max_samples = max(1, int(max_seconds * sample_rate))
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
        fade_seconds = min(0.25, max_seconds * 0.1)
        fade_samples = int(fade_seconds * sample_rate)
        if fade_samples > 1 and audio.shape[-1] >= fade_samples:
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            audio[..., -fade_samples:] *= fade
    save_audio(flac_path, audio, sample_rate=sample_rate, subtype="PCM_16")
    base, _ = os.path.splitext(flac_path)
    wav_path = base + ".wav"
    save_audio(wav_path, audio, sample_rate=sample_rate, subtype="PCM_16")
    return wav_path


def parse_args():
    parser = argparse.ArgumentParser(description="SongGeneration MLX Runner")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint dir")
    parser.add_argument("--weights", required=True, help="Path to MLX weights .npz")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL")
    parser.add_argument("--save_dir", required=True, help="Output directory")
    parser.add_argument("--tokens_only", action="store_true", help="Only write tokens (no audio)")
    parser.add_argument(
        "--generate_type",
        type=str,
        default="mixed",
        help='Type of generation: "vocal" or "bgm" or "separate" or "mixed" (default: "mixed")',
    )
    parser.add_argument(
        "--auto_prompt_path",
        type=str,
        default="tools/new_prompt.npz",
        help="Path to auto prompt tokens (.npz)",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override generation duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument(
        "--separator_backend",
        type=str,
        default="onnx",
        choices=["onnx", "coreml", "none"],
        help="Separator backend for prompt_audio_path when stems are not provided (default: onnx)",
    )
    parser.add_argument(
        "--separator_model",
        type=str,
        default=None,
        help="Path to Demucs-compatible separator model (.onnx or .mlpackage)",
    )
    parser.add_argument(
        "--separator_sample_rate",
        type=int,
        default=44100,
        help="Sample rate expected by the separator model (default: 44100)",
    )
    parser.add_argument(
        "--separator_chunk_seconds",
        type=float,
        default=10.0,
        help="Separator chunk size in seconds for long audio (default: 10)",
    )
    parser.add_argument(
        "--separator_overlap",
        type=float,
        default=0.25,
        help="Separator overlap ratio for chunking (default: 0.25)",
    )
    parser.add_argument(
        "--separator_vocals_index",
        type=int,
        default=3,
        help="Index of vocals stem in separator output (default: 3)",
    )
    parser.add_argument("--cfg_coef", type=float, default=1.5, help="Classifier-free guidance scale")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling (0 to disable)")
    parser.add_argument(
        "--extend_stride",
        type=int,
        default=5,
        help="Extension stride in seconds for long generations",
    )
    return parser.parse_args()

def load_auto_prompts(path: str) -> dict[str, np.ndarray]:
    if not path.endswith(".npz"):
        raise ValueError(f"Auto prompt file must be .npz, got {path}")
    data = np.load(path)
    prompts: dict[str, np.ndarray] = {}
    for key in data.files:
        prompts[key] = np.asarray(data[key])
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def ensure_runtime_assets(base_dir: Path, needs_separator: bool) -> None:
    required = [
        base_dir / "ckpt" / "vae" / "stable_audio_1920_vae.json",
        base_dir / "ckpt" / "vae" / "autoencoder_music_1320k.npz",
    ]
    if needs_separator:
        required.append(base_dir / "third_party" / "demucs" / "ckpt" / "htdemucs.onnx")
    missing = [path for path in required if not path.exists()]
    if not missing:
        return
    missing_list = ", ".join(str(path) for path in missing)
    raise FileNotFoundError(
        "Missing runtime assets. Download by selecting a model in the UI "
        f"or run tools/fetch_runtime.py manually. Missing: {missing_list}"
    )


def main():
    args = parse_args()
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    cfg_path = os.path.join(args.ckpt_path, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = "inference"

    with open(args.input_jsonl, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    new_items = []
    for line in lines:
        item = json.loads(line)
        item["idx"] = f"{item['idx']}"
        new_items.append(item)

    needs_prompt_audio = any("prompt_audio_path" in item for item in new_items)
    needs_prompt_stems = any(
        ("prompt_vocal_path" in item) or ("prompt_bgm_path" in item) for item in new_items
    )
    needs_audio_tokenizer = (not args.tokens_only) or needs_prompt_audio or needs_prompt_stems
    needs_sep_tokenizer = not args.tokens_only
    needs_separator = any(
        ("prompt_audio_path" in item)
        and not (("prompt_vocal_path" in item) or ("prompt_bgm_path" in item))
        for item in new_items
    )

    if needs_audio_tokenizer or (needs_separator and args.separator_backend != "none"):
        ensure_runtime_assets(Path(__file__).resolve().parent, needs_separator and args.separator_backend != "none")

    lm = builders.get_lm_model(cfg)
    load_weights_npz(lm, args.weights)

    audiotokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg) if needs_audio_tokenizer else None
    seperate_tokenizer = None
    if needs_sep_tokenizer and hasattr(cfg, "audio_tokenizer_checkpoint_sep"):
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)

    max_dur = cfg.get("max_dur") if hasattr(cfg, "get") else getattr(cfg, "max_dur", None)
    if max_dur is None and hasattr(cfg, "lyric_processor"):
        max_dur = cfg.lyric_processor.max_dur
    if max_dur is None:
        raise ValueError("max_dur not found in config")

    model = CodecLM(
        name="mlx",
        audiotokenizer=audiotokenizer,
        lm=lm,
        max_duration=max_dur,
        seperate_tokenizer=seperate_tokenizer,
    )

    cfg_coef = args.cfg_coef
    temp = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    record_tokens = True
    record_window = 50
    duration = args.duration if args.duration is not None else max_dur
    duration = max(1.0, min(float(duration), float(max_dur)))
    model.set_generation_params(
        duration=duration,
        extend_stride=args.extend_stride,
        temperature=temp,
        cfg_coef=cfg_coef,
        top_k=top_k,
        top_p=top_p,
        record_tokens=record_tokens,
        record_window=record_window,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    if args.tokens_only:
        os.makedirs(os.path.join(args.save_dir, "tokens"), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.save_dir, "audios"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "jsonl"), exist_ok=True)

    separator = None

    auto_prompt = None
    for item in new_items:
        lyric = item["gt_lyric"]
        descriptions = item.get("descriptions")
        melody_wavs = None
        vocal_wavs = None
        bgm_wavs = None
        prompt_audio = None
        prompt_vocal = None
        prompt_bgm = None
        melody_is_wav = True
        has_prompt_audio = "prompt_audio_path" in item
        has_prompt_stems = ("prompt_vocal_path" in item) or ("prompt_bgm_path" in item)

        if has_prompt_audio and "auto_prompt_audio_type" in item:
            raise ValueError("auto_prompt_audio_type and prompt_audio_path cannot be used together")
        if has_prompt_stems and "auto_prompt_audio_type" in item:
            raise ValueError("auto_prompt_audio_type and prompt stem paths cannot be used together")

        if has_prompt_stems:
            if not ("prompt_vocal_path" in item and "prompt_bgm_path" in item):
                raise ValueError("prompt_vocal_path and prompt_bgm_path must be provided together")
            prompt_vocal_path = item["prompt_vocal_path"]
            prompt_bgm_path = item["prompt_bgm_path"]
            if not os.path.exists(prompt_vocal_path):
                raise FileNotFoundError(f"prompt_vocal_path {prompt_vocal_path} not found")
            if not os.path.exists(prompt_bgm_path):
                raise FileNotFoundError(f"prompt_bgm_path {prompt_bgm_path} not found")
            prompt_vocal = load_audio(prompt_vocal_path, cfg.sample_rate, max_seconds=10.0)
            prompt_bgm = load_audio(prompt_bgm_path, cfg.sample_rate, max_seconds=10.0)
            if has_prompt_audio:
                prompt_path = item["prompt_audio_path"]
                if not os.path.exists(prompt_path):
                    raise FileNotFoundError(f"prompt_audio_path {prompt_path} not found")
                prompt_audio = load_audio(prompt_path, cfg.sample_rate, max_seconds=10.0)
            else:
                min_len = min(prompt_vocal.shape[-1], prompt_bgm.shape[-1])
                prompt_audio = prompt_vocal[:, :min_len] + prompt_bgm[:, :min_len]
            melody_wavs = prompt_audio[None, ...]
            vocal_wavs = prompt_vocal[None, ...]
            bgm_wavs = prompt_bgm[None, ...]
            melody_is_wav = True
            if audiotokenizer is None:
                raise ValueError("Audio tokenizer is required for prompt stems but was not initialized")
        elif has_prompt_audio:
            prompt_path = item["prompt_audio_path"]
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"prompt_audio_path {prompt_path} not found")
            if audiotokenizer is None:
                raise ValueError("Audio tokenizer is required for prompt_audio_path but was not initialized")
            if seperate_tokenizer is not None:
                prompt_audio = load_audio(prompt_path, cfg.sample_rate, max_seconds=10.0)
                prompt_vocal = None
                prompt_bgm = None
                if args.separator_backend != "none":
                    try:
                        if separator is None:
                            separator = create_separator(
                                backend=args.separator_backend,
                                model_path=args.separator_model,
                                sample_rate=args.separator_sample_rate,
                                chunk_seconds=args.separator_chunk_seconds,
                                overlap=args.separator_overlap,
                                vocals_index=args.separator_vocals_index,
                            )
                        prompt_audio, prompt_vocal, prompt_bgm = separator.run(prompt_path, target_sr=cfg.sample_rate)
                    except Exception as exc:
                        print(f"[WARN] Separator failed ({exc}); falling back to mid/side split.", flush=True)
                    finally:
                        if separator is not None:
                            separator.close()
                        separator = None
                        gc.collect()
                if prompt_vocal is None or prompt_bgm is None:
                    prompt_vocal, prompt_bgm = split_prompt_audio(prompt_audio)
                melody_wavs = prompt_audio[None, ...]
                vocal_wavs = prompt_vocal[None, ...]
                bgm_wavs = prompt_bgm[None, ...]
            else:
                prompt_audio = load_audio(prompt_path, cfg.sample_rate, max_seconds=10.0)
                melody_wavs = prompt_audio[None, ...]
            melody_is_wav = True
        elif "auto_prompt_audio_type" in item:
            if auto_prompt is None:
                auto_prompt = load_auto_prompts(args.auto_prompt_path)
            prompt_type = item["auto_prompt_audio_type"]
            if prompt_type not in auto_prompt:
                raise ValueError(f"auto_prompt_audio_type {prompt_type} not found in {args.auto_prompt_path}")
            prompts = auto_prompt[prompt_type]
            if prompts is None:
                raise ValueError(f"No prompts available for {prompt_type} in {args.auto_prompt_path}")
            if isinstance(prompts, np.ndarray):
                if prompts.size == 0:
                    raise ValueError(f"No prompts available for {prompt_type} in {args.auto_prompt_path}")
                count = prompts.shape[0]
            else:
                if len(prompts) == 0:
                    raise ValueError(f"No prompts available for {prompt_type} in {args.auto_prompt_path}")
                count = len(prompts)
            prompt_token = prompts[np.random.randint(0, count)]
            prompt_token = np.asarray(prompt_token, dtype=np.int32)
            melody_wavs = mx.array(prompt_token[:, [0], :], dtype=mx.int32)
            vocal_wavs = mx.array(prompt_token[:, [1], :], dtype=mx.int32)
            bgm_wavs = mx.array(prompt_token[:, [2], :], dtype=mx.int32)
            melody_is_wav = False
        start_time = time.time()
        tokens = model.generate(
            lyrics=[lyric.replace("  ", " ")],
            descriptions=[descriptions],
            melody_wavs=melody_wavs,
            vocal_wavs=vocal_wavs,
            bgm_wavs=bgm_wavs,
            melody_is_wav=melody_is_wav,
            return_tokens=True,
        )
        mid_time = time.time()
        if args.tokens_only:
            tokens_path = os.path.join(args.save_dir, "tokens", f"{item['idx']}.npz")
            np.savez(tokens_path, tokens=np.array(tokens))
            item["tokens_path"] = tokens_path
        else:
            target_wav_name = os.path.join(args.save_dir, "audios", f"{item['idx']}.flac")
            if args.generate_type == "separate":
                audio_mixed = model.generate_audio(
                    tokens,
                    gen_type="mixed",
                    prompt=prompt_audio,
                    vocal_prompt=prompt_vocal,
                    bgm_prompt=prompt_bgm,
                    chunked=True,
                    chunk_size=128,
                    duration=duration,
                )
                audio_vocal = model.generate_audio(
                    tokens,
                    gen_type="vocal",
                    prompt=prompt_audio,
                    vocal_prompt=prompt_vocal,
                    bgm_prompt=prompt_bgm,
                    chunked=True,
                    chunk_size=128,
                    duration=duration,
                )
                audio_bgm = model.generate_audio(
                    tokens,
                    gen_type="bgm",
                    prompt=prompt_audio,
                    vocal_prompt=prompt_vocal,
                    bgm_prompt=prompt_bgm,
                    chunked=True,
                    chunk_size=128,
                    duration=duration,
                )
                vocal_path = target_wav_name.replace(".flac", "_vocal.flac")
                bgm_path = target_wav_name.replace(".flac", "_bgm.flac")
                mixed_wav = save_audio_outputs(
                    target_wav_name, np.array(audio_mixed), sample_rate=cfg.sample_rate, max_seconds=duration
                )
                vocal_wav = save_audio_outputs(
                    vocal_path, np.array(audio_vocal), sample_rate=cfg.sample_rate, max_seconds=duration
                )
                bgm_wav = save_audio_outputs(
                    bgm_path, np.array(audio_bgm), sample_rate=cfg.sample_rate, max_seconds=duration
                )
                item["wav_path"] = target_wav_name
                item["wav_path_wav"] = mixed_wav
                item["vocal_path"] = vocal_path
                item["vocal_wav_path"] = vocal_wav
                item["bgm_path"] = bgm_path
                item["bgm_wav_path"] = bgm_wav
            else:
                audio = model.generate_audio(
                    tokens,
                    gen_type=args.generate_type,
                    prompt=prompt_audio,
                    vocal_prompt=prompt_vocal,
                    bgm_prompt=prompt_bgm,
                    chunked=True,
                    chunk_size=128,
                    duration=duration,
                )
                wav_path = save_audio_outputs(
                    target_wav_name, np.array(audio), sample_rate=cfg.sample_rate, max_seconds=duration
                )
                item["wav_path"] = target_wav_name
                item["wav_path_wav"] = wav_path
        end_time = time.time()
        print(f"process{item['idx']}, lm cost {mid_time - start_time}s, total {end_time - start_time}s")

    src_jsonl_name = os.path.split(args.input_jsonl)[-1]
    with open(os.path.join(args.save_dir, "jsonl", f"{src_jsonl_name}.jsonl"), "w", encoding="utf-8") as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
