import os
import shutil
import subprocess
import tempfile
import wave
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from piper import PiperVoice
from piper.config import SynthesisConfig
from pydantic import BaseModel, Field

from kb_tts.download import CONFIG, TTS_CONFIG_PATH, VOICES_DIR, download_voice_files

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI Compliant Piper TTS Service",
    description="A local Text-to-Speech service utilizing Piper TTS, fully compliant with the OpenAI Speech API format.",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global voice cache to avoid loading models from disk on every request
_voices_cache = {}

# Supported response audio formats and their Content-Types
CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/l16",
}


class SpeechRequest(BaseModel):
    model: str = Field(
        default="tts-1",
        description="Model name. Ignored as Piper is used under the hood.",
    )
    input: str = Field(
        ..., max_length=4096, description="The text to generate audio for."
    )
    voice: str = Field(
        ...,
        description="Voice mapping name, numeric speaker ID, or colon-separated model:speaker (e.g. 'alloy', '12', or 'en_US-libritts-high:123')",
    )
    response_format: str = Field(
        default="mp3",
        description="Audio output format (mp3, wav, opus, ogg, aac, flac, pcm).",
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speed multiplier for speech rate."
    )


def get_loaded_voice(model_name: str) -> PiperVoice:
    """Load model into memory or fetch from global cache."""
    if model_name in _voices_cache:
        return _voices_cache[model_name]

    onnx_path, json_path = download_voice_files(model_name)
    print(f"Loading voice model into memory: {model_name}")
    try:
        voice = PiperVoice.load(onnx_path, config_path=json_path)
        _voices_cache[model_name] = voice
        return voice
    except Exception as e:
        # Clear files on failure to avoid corrupted files blocking future attempts
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize ONNX Runtime for voice model '{model_name}': {str(e)}",
        )


def resolve_voice_params(voice_input: str) -> tuple[str, Optional[int]]:
    """
    Resolve model_name and speaker_id based on voice_input.
    Handles:
    - Standard keys ('alloy', 'echo', etc.)
    - Pure speaker ID numbers ('123') -> uses default_model and speaker_id 123
    - Colon-separated values ('en_US-libritts-high:123') -> uses custom model and speaker_id 123
    - Pure model names ('en_US-amy-medium') -> uses model and speaker_id None
    """
    voice_map = CONFIG.get("voice_map", {})
    default_model = CONFIG.get("default_model", "en_US-libritts-high")

    # 1. Check if it matches a mapped name in the JSON configuration
    if voice_input in voice_map:
        mapping = voice_map[voice_input]
        return mapping.get("model", default_model), mapping.get("speaker_id")

    # 2. Check if it is a pure numeric speaker ID
    if voice_input.isdigit():
        return default_model, int(voice_input)

    # 3. Check if it contains a colon (model:speaker_id)
    if ":" in voice_input:
        parts = voice_input.split(":", 1)
        model_name = parts[0]
        try:
            speaker_id = int(parts[1])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid speaker ID format in voice: '{voice_input}'. Speaker ID must be an integer.",
            )
        return model_name, speaker_id

    # 4. Fallback: treat voice_input as a full model name
    return voice_input, None


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary on the system (supporting local Windows and Docker environments)."""
    # 1. Check FFMPEG_PATH environment variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Check if ffmpeg is in standard system PATH
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    # 3. Windows-specific fallback searches
    if os.name == "nt":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            winget_root = os.path.join(user_profile, ".winget_portable_root")
            if os.path.exists(winget_root):
                for root, _, files in os.walk(winget_root):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")

    return "ffmpeg"


def convert_audio(wav_path: str, response_format: str) -> bytes:
    """Run ffmpeg as a subprocess to convert the source WAV into the requested audio format."""
    if response_format == "wav":
        with open(wav_path, "rb") as f:
            return f.read()

    ffmpeg_bin = find_ffmpeg()
    with tempfile.NamedTemporaryFile(
        suffix=f".{response_format}", delete=False
    ) as temp_out:
        temp_out_path = temp_out.name

    try:
        if response_format == "mp3":
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                temp_out_path,
            ]
        elif response_format in ("opus", "ogg"):
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                wav_path,
                "-c:a",
                "libopus",
                "-b:a",
                "96k",
                temp_out_path,
            ]
        elif response_format == "aac":
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                wav_path,
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                temp_out_path,
            ]
        elif response_format == "flac":
            cmd = [ffmpeg_bin, "-y", "-i", wav_path, "-c:a", "flac", temp_out_path]
        elif response_format == "pcm":
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                wav_path,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                temp_out_path,
            ]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio response format: {response_format}",
            )

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Audio conversion failed via ffmpeg: {result.stderr}",
            )

        with open(temp_out_path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)


@app.get("/health")
def health():
    """Service health check containing loaded models and config settings."""
    return {
        "status": "healthy",
        "loaded_models": list(_voices_cache.keys()),
        "voices_dir": VOICES_DIR,
        "config_path": TTS_CONFIG_PATH,
        "config": CONFIG,
    }


@app.post("/v1/audio/speech")
def generate_speech(request: SpeechRequest):
    """OpenAI compliant Speech synthesis endpoint."""
    response_format = request.response_format.lower()
    if response_format not in CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format. Supported formats are: {list(CONTENT_TYPES.keys())}",
        )

    # Resolve the model and speaker
    model_name, speaker_id = resolve_voice_params(request.voice)

    # Fetch/Load the voice
    voice = get_loaded_voice(model_name)

    # Check if speaker_id is valid for this model if the model is multi-speaker
    if voice.config.num_speakers > 1 and speaker_id is not None:
        if speaker_id < 0 or speaker_id >= voice.config.num_speakers:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker ID {speaker_id} is out of bounds. Voice model '{model_name}' has {voice.config.num_speakers} speakers.",
            )

    # Set synthesis config
    # speed maps to length_scale = 1.0 / speed
    length_scale = 1.0 / request.speed if request.speed > 0 else 1.0
    syn_config = SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale)

    # Synthesize to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    try:
        with wave.open(temp_wav_path, "wb") as wav_file:
            voice.synthesize_wav(request.input, wav_file, syn_config=syn_config)

        # Convert WAV to output format
        audio_data = convert_audio(temp_wav_path, response_format)

        return Response(content=audio_data, media_type=CONTENT_TYPES[response_format])

    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
