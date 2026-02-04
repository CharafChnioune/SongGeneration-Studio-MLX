# SongGeneration Studio API

Base URL: `http://127.0.0.1:8000`

No auth by default. All endpoints return JSON unless noted.

OpenAPI docs:
- `GET /api/docs`
- `GET /api/openapi.json`

## Content Types

- JSON: `application/json`
- Multipart (file uploads): `multipart/form-data`

## Core Request Schemas

### Section
```json
{
  "type": "verse",
  "lyrics": "..."
}
```

### SongRequest
```json
{
  "title": "Untitled",
  "sections": [{"type":"verse","lyrics":"..."}],
  "gender": "female",
  "timbre": "",
  "genre": "",
  "emotion": "",
  "instruments": "",
  "custom_style": "",
  "bpm": 120,
  "output_mode": "mixed",
  "reference_audio_id": null,
  "model": "songgeneration_base",
  "cfg_coef": 1.5,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.0,
  "extend_stride": 5
}
```
Notes:
- `output_mode` values: `mixed`, `vocal`, `bgm`, `separate`.
- `reference_audio_id` comes from the upload endpoints.
- Extra fields are ignored.

### UpdateGenerationRequest
```json
{ "title": "New title" }
```

### AIAssistRequest
```json
{
  "provider": "lmstudio",
  "base_url": null,
  "model": "...",
  "prompt": "...",
  "language": "English",
  "length": "medium"
}
```

### AIAssistStepRequest
```json
{
  "provider": "lmstudio",
  "base_url": null,
  "model": "...",
  "prompt": "...",
  "language": "English",
  "length": "medium",
  "step": "...",
  "state": {},
  "section_index": 0,
  "instruction": "..."
}
```

---

## System

### `GET /api/health`
Returns basic status and cached GPU info.

### `GET /api/info`
Returns name, version, docs URLs.

### `GET /api/gpu`
Returns refreshed GPU info.

### `GET /api/timing-stats`
Returns timing history stats.

### `GET /api/test-sse`
Test SSE endpoint that counts to 10.

---

## Models

### `GET /api/models`
List all models, ready models, recommended model.

### `POST /api/models/{model_id}/download`
Start model download.

### `DELETE /api/models/{model_id}/download`
Cancel download (currently returns error).

### `DELETE /api/models/{model_id}`
Delete a downloaded model.

---

## Reference Audio Uploads

### `POST /api/upload-reference`
Multipart upload. Returns `{id, filename}`.

**Form fields**:
- `file` (required)

### `POST /api/upload-and-trim-reference`
Multipart upload + server-side trim (ffmpeg). Returns `{id, filename}`.

**Form fields**:
- `file` (required)
- `trim_start` (seconds, default 0)
- `trim_duration` (seconds, default 10, capped to 10)

### `GET /api/reference/{ref_id}`
Returns WAV audio file for a reference id.

---

## Generation

### `POST /api/generate`
Create a generation job.

**Body**: `SongRequest`

**Response**:
```json
{ "generation_id": "abcd1234" }
```

### `POST /api/generate-with-reference`
Single multipart request: JSON payload + reference audio upload.

**Form fields**:
- `payload` (JSON string, SongRequest fields)
- `file` (audio)
- `trim_start` (optional)
- `trim_duration` (optional)

### `GET /api/generation/{gen_id}`
Get generation status (adds `elapsed_seconds` if running).

### `POST /api/stop/{gen_id}`
Stop a pending generation (not possible if processing).

### `DELETE /api/generation/{gen_id}`
Delete a generation (not possible if processing).

### `PUT /api/generation/{gen_id}`
Update generation metadata (title).

**Body**: `UpdateGenerationRequest`

---

## Covers

### `POST /api/generation/{gen_id}/cover`
Upload album cover image (multipart).

**Form fields**:
- `file` (jpg/png/gif/webp)

### `GET /api/generation/{gen_id}/cover`
Return cover image or `204` if none.

### `DELETE /api/generation/{gen_id}/cover`
Delete cover image.

---

## Video Export

### `GET /api/generation/{gen_id}/video`
Export MP4 video with waveform. Requires ffmpeg/ffprobe.

---

## Library & Audio

### `GET /api/generations`
List all generations with metadata.

### `GET /api/audio/{gen_id}/{track_idx}`
Download audio track.

**Query params**:
- `format` = `wav|flac|mp3` (optional)

---

## Queue

### `GET /api/queue`
List queue items.

### `POST /api/queue`
Add item to queue (payload is similar to `SongRequest`).

### `DELETE /api/queue/{item_id}`
Remove one item.

### `DELETE /api/queue`
Clear the queue.

---

## AI Assist

### `POST /api/ai/assist`
Body: `AIAssistRequest`

### `POST /api/ai/assist-step`
Body: `AIAssistStepRequest`

### `GET /api/ai/models?provider=lmstudio|ollama&base_url=...`
List available AI models from the selected provider.

---

## Events

### `GET /api/events`
SSE endpoint is disabled and returns `410`.

---

## Example Calls

### Generate (JSON)
```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Demo",
    "sections":[{"type":"verse","lyrics":"Night drive. neon sky."}],
    "output_mode":"mixed"
  }'
```

### Generate with reference (multipart)
```bash
curl -X POST http://127.0.0.1:8000/api/generate-with-reference \
  -F 'payload={"title":"Demo","sections":[{"type":"verse","lyrics":"Night drive. neon sky."}],"output_mode":"mixed"}' \
  -F 'file=@/path/to/reference.wav' \
  -F 'trim_start=0' \
  -F 'trim_duration=10'
```
