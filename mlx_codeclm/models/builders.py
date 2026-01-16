"""Builders for MLX SongGeneration models."""

import typing as tp
import os
import omegaconf

from mlx_codeclm.modules.pattern import CodebooksPatternProvider, DelayedPatternProvider
from mlx_codeclm.modules.conditioners import (
    ConditionerProvider,
    ConditionFuser,
    QwTokenizerConditioner,
    QwTextConditioner,
    QuantizedEmbeddingConditioner,
)
from mlx_codeclm.models.lm_levo import LmModel
from mlx_codeclm.tokenizer.audio_tokenizer import AudioTokenizer


def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(dct, dict):
        raise ValueError("Config is not a dict")
    return dct


def get_codebooks_pattern_provider(code_depth: int, cfg: omegaconf.DictConfig) -> CodebooksPatternProvider:
    pattern_providers = {
        "delay": DelayedPatternProvider,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name)) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(code_depth, **kwargs)


def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    fuser_cfg = getattr(cfg, "fuser")
    fuser_methods = ["sum", "prepend"]
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    _ = kwargs  # unused
    return ConditionFuser(fuse2cond=fuse2cond)


def get_conditioner_provider(output_dim: int, cfg: omegaconf.DictConfig) -> ConditionerProvider:
    cfg = getattr(cfg, "conditioners")
    dict_cfg = {} if cfg is None else dict_from_config(cfg)
    conditioners: tp.Dict[str, tp.Any] = {}
    _ = dict_cfg.pop("args", {})

    for cond, cond_cfg in dict_cfg.items():
        model_type = cond_cfg["model"]
        model_args = cond_cfg[model_type]
        if model_type == "QwTokenizer":
            conditioners[str(cond)] = QwTokenizerConditioner(output_dim=output_dim, **model_args)
        elif model_type == "QwTextTokenizer":
            conditioners[str(cond)] = QwTextConditioner(output_dim=output_dim, **model_args)
        elif model_type == "qt_embedding":
            conditioners[str(cond)] = QuantizedEmbeddingConditioner(dim=output_dim, **model_args)
        else:
            raise ValueError(f"Unrecognized conditioner model: {model_type}")
    return ConditionerProvider(conditioners)


def get_lm_model(cfg: omegaconf.DictConfig) -> LmModel:
    lm_kwargs = dict_from_config(getattr(cfg, "lm"))
    if not hasattr(cfg, "prompt_len") and hasattr(cfg, "lyric_processor"):
        if hasattr(cfg.lyric_processor, "prompt_len"):
            cfg.prompt_len = cfg.lyric_processor.prompt_len
    if not hasattr(cfg, "max_dur") and hasattr(cfg, "lyric_processor"):
        if hasattr(cfg.lyric_processor, "max_dur"):
            cfg.max_dur = cfg.lyric_processor.max_dur
    code_depth = lm_kwargs["code_depth"]
    q_modeling = lm_kwargs.pop("q_modeling", None)

    condition_provider = get_conditioner_provider(lm_kwargs["dim"], cfg)

    codebooks_pattern_cfg = getattr(cfg, "codebooks_pattern")
    if codebooks_pattern_cfg.modeling is None:
        if q_modeling is None:
            raise ValueError("No codebook pattern defined")
        codebooks_pattern_cfg = omegaconf.OmegaConf.create(
            {"modeling": q_modeling, "delay": {"delays": list(range(code_depth))}}
        )
    pattern_provider = get_codebooks_pattern_provider(code_depth, codebooks_pattern_cfg)

    cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
    cfg_coef = cls_free_guidance["inference_coef"]

    fuser = get_condition_fuser(cfg)

    lm_type = lm_kwargs["lm_type"]
    if lm_type != "Llama":
        raise ValueError(f"Unexpected LM model {lm_type}")
    return LmModel(
        pattern_provider=pattern_provider,
        condition_provider=condition_provider,
        fuser=fuser,
        cfg_coef=cfg_coef,
        cfg=cfg,
        **lm_kwargs,
    )


def _mlx_weight_path(path: str, suffix: str = "_fp16.npz") -> str:
    if path.endswith(".npz"):
        return path
    base, _ = os.path.splitext(path)
    candidate = base + suffix
    if os.path.exists(candidate):
        return candidate
    candidate = base + ".npz"
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Missing MLX weights for {path}. Tried {candidate} and {base + suffix}")


def get_audio_tokenizer_model(checkpoint_path: str, cfg: omegaconf.DictConfig) -> tp.Optional[AudioTokenizer]:
    if checkpoint_path is None:
        return None
    name = checkpoint_path
    model_path = checkpoint_path.split("_", 1)[1] if "_" in checkpoint_path else checkpoint_path
    vae_config = cfg.vae_config if hasattr(cfg, "vae_config") else None
    vae_model = cfg.vae_model if hasattr(cfg, "vae_model") else None
    if vae_config is None or vae_model is None:
        raise ValueError("vae_config or vae_model missing from config")
    vae_model = _mlx_weight_path(vae_model, suffix=".npz")
    model_weights = _mlx_weight_path(model_path, suffix="_fp16.npz")
    return AudioTokenizer.get_pretrained(name, vae_config, vae_model, model_weights)
