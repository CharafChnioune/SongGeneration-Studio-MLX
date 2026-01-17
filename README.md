# SongGeneration Studio MLX

PyTorch-free SongGeneration for Apple Silicon, powered by MLX. Includes a full web UI, lyrics/style AI, and MLX-only inference.

## Highlights

- 100% MLX inference (no PyTorch).
- Web UI for full songs, vocals, instrumentals, or separated stems.
- Built-in AI for style + lyrics (LM Studio or Ollama).
- Optional prompt-audio conditioning with ONNX/CoreML separator.
- Multi-candidate generation + auto-select best take.
- Auto-arrangement templates + genre presets.

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

This downloads `ckpt/` and `third_party/` from the runtime assets repo.

## MLX model weights (auto-download)

On first run, the app will **automatically download** the recommended model (based on your available memory) from Hugging Face.

You can also open **Manage Models** in the UI to download or remove models at any time.

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

## Prompt audio (optional)

- Provide `prompt_audio_path` in the JSONL to condition on a reference.
- For Demucs-style separation without PyTorch, place an ONNX or CoreML model at:
  - `third_party/demucs/ckpt/htdemucs.onnx` or
  - `third_party/demucs/ckpt/htdemucs.mlpackage`
- You can override backend/model via `generate_mlx.py --separator_backend` and `--separator_model`.

## Lyrics + Style AI (LM Studio / Ollama)

In the UI, set the provider, base URL, and model, then use **AI Create (Style + Lyrics)** to generate both in one step.

## Output & logs

Each generation writes logs to:

```
output/<generation_id>/generation.log
```

## Credits

Based on Tencent AI Lab's SongGeneration research and assets, adapted here as a fully MLX-only studio.
