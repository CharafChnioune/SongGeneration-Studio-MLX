# SongGeneration Studio MLX

PyTorch-free SongGeneration for Apple Silicon, powered by MLX. Includes a full web UI and MLX-only inference.

## Highlights

- 100% MLX inference (no PyTorch).
- Web UI for full songs, vocals, instrumentals, or separated stems.
- Lyrics + optional text descriptions + optional reference audio (paper-aligned).
- Optional prompt-audio conditioning with ONNX/CoreML separator.

## Requirements

- macOS Apple Silicon
- Python 3.11 or 3.12 (recommended; `onnxruntime` is not available on Python 3.13/3.14).
- Optional: `ffmpeg`/`ffprobe` for media metadata in the UI library.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mlx.txt
```

## Runtime assets (ckpt + third_party)

```bash
python tools/fetch_runtime.py --local-dir .
```

This downloads `ckpt/` and `third_party/` from the runtime assets repo. The Model Manager will also pull runtime assets automatically when you download a model.

## MLX model weights (download when selected)

Open **Manage Models** in the UI to download or remove models. The app does not auto-download models on first run.

By default, models are pulled from these repos:

- `AITRADER/SongGeneration-Base-MLX`
- `AITRADER/SongGeneration-Base-New-MLX`
- `AITRADER/SongGeneration-Base-Full-MLX`
- `AITRADER/SongGeneration-Large-MLX`

If you need to override the source repos, set:

```
SONGGEN_MLX_HF_REPO
SONGGEN_MLX_HF_LAYOUT
SONGGEN_MLX_HF_PREFIX
```

## Run the Web UI

```bash
python main.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## API (for mobile/desktop apps)

OpenAPI docs are available at:

- `/api/docs`
- `/api/openapi.json`

Example: start a generation

```bash
curl -X POST http://127.0.0.1:8000/api/generate \\
  -H 'Content-Type: application/json' \\
  -d '{
    "title": "Demo",
    "sections": [{"type":"verse","lyrics":"Night drive. neon sky."}],
    "genre": "pop",
    "emotion": "uplifting",
    "timbre": "soft",
    "instruments": "piano, synthesizer",
    "bpm": 120,
    "output_mode": "mixed"
  }'
```

Example: generate with reference audio (single request)

```bash
curl -X POST http://127.0.0.1:8000/api/generate-with-reference \\
  -F 'payload={"title":"Demo","sections":[{"type":"verse","lyrics":"Night drive. neon sky."}],"output_mode":"mixed"}' \\
  -F 'file=@/path/to/reference.wav' \\
  -F 'trim_start=0' \\
  -F 'trim_duration=10'
```

## CLI (MLX)

Make sure you have downloaded the model first (via the UI or by placing weights in the model folder).

```bash
python generate_mlx.py \
  --ckpt_path songgeneration_large \
  --weights songgeneration_large/model_fp16.npz \
  --input_jsonl sample/lyrics.jsonl \
  --save_dir output/cli
```

Outputs `.flac` + `.wav` in `output/cli/audios/`.

## Self-test (MLX)

Quick end-to-end check that generates a short sample if a local model is present:

```bash
python tools/self_test_mlx.py --duration 12
```

Optional flags:

- `--tokens_only` to skip audio decode.
- `--fetch_runtime` to download runtime assets if missing.
- `--download_model` to fetch the selected model if missing.
- `--debug` or `SONGGEN_MLX_DEBUG=1` for MLX debug logs.

## Lyrics format notes

- `[intro-short]`, `[intro-medium]`, `[inst-short]`, `[inst-medium]`, `[outro-short]`, `[outro-medium]` are instrumental (lyrics are ignored).
- `[inst-*]` is supported but less stable than intro/outro tags.
- If a `-long` tag is used, it is normalized to `-medium` for compatibility.

## Prompt audio (optional)

- Provide `prompt_audio_path` in the JSONL to condition on a reference.
- For Demucs-style separation without PyTorch, place an ONNX or CoreML model at:
  - `third_party/demucs/ckpt/htdemucs.onnx` or
  - `third_party/demucs/ckpt/htdemucs.mlpackage`
- You can override backend/model via `generate_mlx.py --separator_backend` and `--separator_model`.
- You can combine reference audio with text tags, but avoid conflicting prompts for best alignment.

## Output & logs

Each generation writes logs to:

```
output/<generation_id>/generation.log
```

## Credits

Based on Tencent AI Lab's SongGeneration research and assets, adapted here as a fully MLX-only studio.

## Upcoming v1.5 models (prep)

The v1.5 models are marked as **Coming soon** in the Model Manager. When the MLX weights are ready, you can:

1) Add the Hugging Face repo to `MODEL_REGISTRY` (or flip `available: true`).
2) Validate the local model layout:

```bash
python tools/validate_mlx_model.py --model-dir /path/to/songgeneration_v1_5_small
```

Expected files for each model:

- `config.yaml`
- `model_fp16.npz` (required)
- `model_int8.npz` (optional, recommended)
