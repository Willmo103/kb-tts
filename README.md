# kb-tts: OpenAI-Compliant Piper TTS Service

A high-performance, local Text-to-Speech (TTS) service that implements the OpenAI Speech API (`/v1/audio/speech`) format using **Piper TTS** and the multi-speaker **LibriTTS** dataset models. 

Fully containerized with automated model pre-downloading and persistent host-mounted caching.

---

## Features

- **OpenAI Compliance**: Fully compatible with OpenAI TTS client libraries and requests. Exposes `/v1/audio/speech` endpoint.
- **Voice Mapping**: Maps standard OpenAI voices (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) to distinct speaker IDs in the `en_US-libritts-high` model, customizable via `voices_config.json`.
- **Dynamic Downloader**: Dynamically downloads additional model configurations and ONNX binaries directly from the official Hugging Face Piper repository.
- **In-Memory Caching**: Caches loaded model runtimes in memory to reduce invocation latency on subsequent requests.
- **Audio Formats**: Subprocess-based `ffmpeg` audio conversion supporting output formats: `mp3`, `wav`, `opus`, `ogg`, `aac`, `flac`, and raw `pcm`.
- **Speed Control**: Respects the standard OpenAI `speed` parameter (from `0.25` to `4.0`).
- **Standardized Code Quality**: Full static checking and formatting suite utilizing Ruff, Pyright, and Pytest.

---

## Installation & Setup

Ensure you have the [uv package manager](https://github.com/astral-sh/uv) installed.

### Local Development Setup

1. **Sync dependencies**:
   ```bash
   uv sync
   ```
2. **Run Formatting and Lints**:
   ```bash
   # Run ruff formatter
   uv run ruff format .

   # Run ruff check
   uv run ruff check .

   # Run pyright type checking
   uv run pyright .
   ```
3. **Run Unit Tests**:
   ```bash
   uv run pytest
   ```

### Running Locally

Start the local server (defaulting to port `8000`):
```bash
uv run kb-tts
```

---

## Docker Usage

The Docker setup is configured to pre-download the default `en_US-libritts-high` model during the build stage (`/app/default_voices`) and copy it to the host volume on startup. This allows the service to run completely offline immediately.

### Build and Run with Docker Compose

1. **Build and start the container**:
   ```bash
   docker compose up --build
   ```
2. The server will start on `http://localhost:8000`. The default voice models will be copied to `./data/voices` on the host machine.

---

## Configuration

Configuration is loaded from `voices_config.json` (custom path can be set via `TTS_CONFIG_PATH` environment variable):

```json
{
  "default_model": "en_US-libritts-high",
  "voice_map": {
    "alloy": {
      "model": "en_US-libritts-high",
      "speaker_id": 0
    },
    "echo": {
      "model": "en_US-libritts-high",
      "speaker_id": 1
    },
    "fable": {
      "model": "en_US-libritts-high",
      "speaker_id": 2
    },
    "onyx": {
      "model": "en_US-libritts-high",
      "speaker_id": 3
    },
    "nova": {
      "model": "en_US-libritts-high",
      "speaker_id": 4
    },
    "shimmer": {
      "model": "en_US-libritts-high",
      "speaker_id": 5
    }
  }
}
```

### Client Selection Formats:
When calling the API's `voice` parameter, the server supports:
1. **Mapped Names**: e.g., `voice: "alloy"` (maps to speaker 0 of `en_US-libritts-high`).
2. **Speaker Numeric ID**: e.g., `voice: "123"` (uses speaker ID 123 of the default model).
3. **Model & Speaker Combination**: e.g., `voice: "en_US-libritts-high:102"` or `voice: "en_US-amy-medium"`.

---

## API Documentation

### `/v1/audio/speech` (POST)
Synthesizes the input text into audio.

**Example Request (curl)**:
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test of local speech synthesis.",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output output.mp3
```

### `/health` (GET)
Returns the system status, list of loaded models in memory, and the currently parsed config.
