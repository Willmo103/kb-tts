import json
import os
import sys

import requests

VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(os.getcwd(), "data", "voices"))
TTS_CONFIG_PATH = os.environ.get(
    "TTS_CONFIG_PATH", os.path.join(os.getcwd(), "voices_config.json")
)


def load_config():
    if not os.path.exists(TTS_CONFIG_PATH):
        default_config = {
            "default_model": "en_US-libritts-high",
            "clean_markdown": True,
            "strip_code_blocks": False,
            "voice_map": {
                "alloy": {"model": "en_US-libritts-high", "speaker_id": 0},
                "echo": {"model": "en_US-libritts-high", "speaker_id": 1},
                "fable": {"model": "en_US-libritts-high", "speaker_id": 2},
                "onyx": {"model": "en_US-libritts-high", "speaker_id": 3},
                "nova": {"model": "en_US-libritts-high", "speaker_id": 4},
                "shimmer": {"model": "en_US-libritts-high", "speaker_id": 5},
            },
        }
        os.makedirs(os.path.dirname(TTS_CONFIG_PATH), exist_ok=True)
        with open(TTS_CONFIG_PATH, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(TTS_CONFIG_PATH, "r") as f:
        return json.load(f)


CONFIG = load_config()


def get_download_urls(model_name: str) -> tuple[str, str]:
    """Parse locale, voice name, and quality from model_name to construct Hugging Face download URLs."""
    parts = model_name.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid model name format: '{model_name}'. "
            f"Expected format: locale-name-quality (e.g., en_US-libritts-high)"
        )
    locale = parts[0]
    quality = parts[-1]
    voice_name = "-".join(parts[1:-1])
    language = locale.split("_")[0]

    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{language}/{locale}/{voice_name}/{quality}"
    onnx_url = f"{base_url}/{model_name}.onnx"
    json_url = f"{base_url}/{model_name}.onnx.json"
    return onnx_url, json_url


def download_voice_files(model_name: str) -> tuple[str, str]:
    """Download ONNX and config JSON files from Hugging Face if not already present locally."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    onnx_path = os.path.join(VOICES_DIR, f"{model_name}.onnx")
    json_path = os.path.join(VOICES_DIR, f"{model_name}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        return onnx_path, json_path

    onnx_url, json_url = get_download_urls(model_name)

    # Download JSON config
    if not os.path.exists(json_path):
        print(f"Downloading model configuration: {json_url}")
        r = requests.get(json_url)
        if r.status_code == 404:
            raise FileNotFoundError(
                f"Voice model '{model_name}' was not found in the official Piper repository."
            )
        elif r.status_code != 200:
            raise IOError(f"Failed to fetch voice configuration: HTTP {r.status_code}")
        with open(json_path, "wb") as f:
            f.write(r.content)

    # Download ONNX model binary
    if not os.path.exists(onnx_path):
        print(f"Downloading voice model binary: {onnx_url}")
        r = requests.get(onnx_url, stream=True)
        if r.status_code != 200:
            if os.path.exists(json_path):
                os.remove(json_path)
            raise IOError(f"Failed to fetch voice model binary: HTTP {r.status_code}")
        with open(onnx_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    return onnx_path, json_path


def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = [CONFIG.get("default_model", "en_US-libritts-high")]

    print(f"Target voices directory: {VOICES_DIR}")
    for model in models:
        try:
            print(f"Starting download process for model: {model}")
            download_voice_files(model)
        except Exception as e:
            print(f"ERROR downloading model '{model}': {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
