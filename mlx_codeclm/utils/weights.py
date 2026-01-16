"""Weight loading helpers for MLX models."""

import typing as tp

import numpy as np
import mlx.core as mx


def _alias_weight_norm(name: str) -> tp.Optional[str]:
    if ".parametrizations.weight.original0" in name:
        return name.replace(".parametrizations.weight.original0", ".weight_g")
    if ".parametrizations.weight.original1" in name:
        return name.replace(".parametrizations.weight.original1", ".weight_v")
    return None


def _resolve_path(root, parts):
    obj = root
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def set_param(root, name, value):
    parts = name.split(".")
    if not parts:
        raise ValueError("Empty parameter name")
    obj = root
    for part in parts[:-1]:
        if part.isdigit():
            idx = int(part)
            if hasattr(obj, "_modules_list") and idx >= len(obj) and idx == 2 and len(obj) == 2:
                idx = 1
            obj = obj[idx]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        if hasattr(obj, "_modules_list") and idx >= len(obj) and idx == 2 and len(obj) == 2:
            idx = 1
        obj[idx] = value
    else:
        setattr(obj, last, value)


def load_weights_npz(model, path: str, quiet: bool = False, ignore_prefixes: tuple[str, ...] = ()) -> None:
    data = np.load(path)
    scales = {}
    for name in data.files:
        if name.endswith("__scale"):
            scales[name[:-7]] = data[name]
    for name in data.files:
        if name.endswith("__scale"):
            continue
        arr = data[name]
        if name in scales and arr.dtype in (np.int8, np.uint8):
            arr = arr.astype(np.float32) * scales[name]
        try:
            set_param(model, name, mx.array(arr))
        except (AttributeError, KeyError, IndexError) as exc:
            alias = _alias_weight_norm(name)
            if alias is not None:
                try:
                    set_param(model, alias, mx.array(arr))
                    continue
                except (AttributeError, KeyError, IndexError):
                    pass
            if not quiet and not any(name.startswith(prefix) for prefix in ignore_prefixes):
                print(f"skip {name}: {exc}")
            continue


def load_weights_npz_prefixed(
    model,
    path: str,
    prefix: str,
    strip_prefix: bool = True,
    quiet: bool = False,
) -> None:
    data = np.load(path)
    prefix = prefix if prefix.endswith(".") else prefix + "."
    scales = {}
    for name in data.files:
        if name.endswith("__scale"):
            scales[name[:-7]] = data[name]
    for name in data.files:
        if not name.startswith(prefix) or name.endswith("__scale"):
            continue
        key = name[len(prefix) :] if strip_prefix else name
        arr = data[name]
        if name in scales and arr.dtype in (np.int8, np.uint8):
            arr = arr.astype(np.float32) * scales[name]
        try:
            set_param(model, key, mx.array(arr))
        except (AttributeError, KeyError, IndexError) as exc:
            alias = _alias_weight_norm(key)
            if alias is not None:
                try:
                    set_param(model, alias, mx.array(arr))
                    continue
                except (AttributeError, KeyError, IndexError):
                    pass
            if not quiet:
                print(f"skip {key}: {exc}")
