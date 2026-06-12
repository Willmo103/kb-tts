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

### `/audio/speech` (POST)
Synthesizes the input text into audio.

**Example Request (curl)**:
```bash
curl -X POST http://localhost:8000/audio/speech \
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

### `/audio/transcriptions` (POST)
OpenAI-compliant speech-to-text transcription endpoint using local Whisper.

**Example Request (curl)**:
```bash
curl -X POST http://localhost:8000/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=base" \
  -F "response_format=json" \
  -o transcription.json
```

Supported `response_format` types: `json`, `text`, `srt`, `vtt`.

### `/health` (GET)
Returns the system status, list of loaded models in memory, and the currently parsed config.

---

## Custom Voice Training & Deployment

This project includes a complete pipeline for training custom Piper voice models using transcribed YouTube audio and deploying them directly back into the API.

### 1. Training Dashboard UI (YouTube Aggregator)
The dashboard is served directly by the API server at:
**`http://localhost:8000/training`**

Use this dashboard to:
- **Aggregate YouTube Audio**: Enter any YouTube link. The background generator downloads the audio, transcribes it using `openai-whisper` (support for `tiny`, `base`, `small`, `medium` models), slices the audio into short sentence segments, and outputs a clean **LJSpeech-formatted dataset** (comprising `wav/` folder and `metadata.csv`).
- **Monitor Generation Logs**: Live console viewer and progress bar showing downloading, transcription, and segmentation tasks.
- **Download Datasets**: Download generated datasets as `.zip` files to run training on external compute rigs.
- **Upload Trained Models**: Directly drag-and-drop your finished `.onnx` and `.onnx.json` model files.
- **Voice Mapping Registry**: Register new custom voice aliases (e.g. mapping `my_voice` to your uploaded model).
- **Training Center (Baremetal)**: Initiate, monitor, and stop training/fine-tuning runs directly on your AI server. You can select datasets, choose/download pre-trained base checkpoints, adjust epochs and batch sizes, monitor logs in real time, and export checkpoints directly to ONNX.
- **TTS Playground**: Immediately test custom voices and speeds inside the browser.

### 2. GPU-Accelerated Docker Training
For hardware-rich environments with NVIDIA GPUs, use the training-specific Docker stack to train your model:

1. **Spin up the training container**:
   ```bash
   docker compose -f docker-compose.train.yml up -d --build
   ```
2. **Preprocess your generated dataset** inside the container:
   ```bash
   docker compose -f docker-compose.train.yml exec piper-train python3 -m piper_train.preprocess \
     --language en_US \
     --input-dir /datasets/your-dataset-name \
     --output-dir /training_runs/your-dataset-processed \
     --dataset-format ljspeech \
     --sample-rate 22050
   ```
3. **Fine-tune matching a pre-trained base model**:
   - Download a base checkpoint matching your language (from the [Rhasspy checkpoints Hugging Face page](https://huggingface.co/datasets/rhasspy/piper-checkpoints)).
   - Run the trainer:
   ```bash
   docker compose -f docker-compose.train.yml exec piper-train python3 -m piper_train \
     --dataset-dir /training_runs/your-dataset-processed \
     --accelerator gpu \
     --devices 1 \
     --batch-size 16 \
     --max_epochs 1000 \
     --resume_from_checkpoint /training_runs/base-checkpoint.ckpt
   ```
4. **Export checkpoints to ONNX**:
   - Run the export utility:
   ```bash
   docker compose -f docker-compose.train.yml exec piper-train python3 -m piper_train.export_onnx \
     --checkpoint /training_runs/your-dataset-processed/lightning_logs/version_0/checkpoints/epoch=199-step=2000.ckpt \
     --output-file /training_runs/my_custom_voice.onnx
   ```
   - This outputs `my_custom_voice.onnx` and `my_custom_voice.onnx.json`. Upload these using the Model Deployment tab in the Dashboard UI.

### 3. Interactive Jupyter Notebook
Alternatively, if you pull the repository to run training in a notebook (local Jupyter or Google Colab):
- Open **[notebooks/train_piper.ipynb](file:///c:/src/kb-tts/notebooks/train_piper.ipynb)**.
- Follow the interactive steps to install dependencies, aggregate YouTube clips, run the preprocessing script, fine-tune models, and export to ONNX.

