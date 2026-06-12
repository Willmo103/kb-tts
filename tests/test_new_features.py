from fastapi.testclient import TestClient
from kb_tts.api import app

client = TestClient(app)

def test_markdown_cleaning_api():
    """Verify speech synthesis with markdown cleaning options using the TestClient."""
    # 1. Clean markdown (default or explicit)
    payload = {
        "model": "tts-1",
        "input": "This is **bold** text and `inline code`.",
        "voice": "alloy",
        "response_format": "mp3",
        "clean_markdown": True
    }
    r = client.post("/audio/speech", json=payload)
    assert r.status_code == 200, f"Failed with {r.text}"
    assert len(r.content) > 0, "Returned empty audio"

    # 2. Raw markdown (no cleaning)
    payload["clean_markdown"] = False
    r_raw = client.post("/audio/speech", json=payload)
    assert r_raw.status_code == 200, f"Failed with {r_raw.text}"
    assert len(r_raw.content) > 0, "Returned empty audio"

def test_custom_model_override_api(monkeypatch):
    """Verify custom model resolution overrides using the TestClient."""
    # Mock download_voice_files to avoid downloading weights during testing
    import kb_tts.api
    monkeypatch.setattr(
        kb_tts.api, 
        "download_voice_files", 
        lambda model_name: ("data/voices/en_US-libritts-high.onnx", "data/voices/en_US-libritts-high.onnx.json")
    )
    # Mock PiperVoice.load to avoid loading runtime ONNX weights
    from piper import PiperVoice
    class DummyConfig:
        num_speakers = 100
    class DummyVoice:
        config = DummyConfig()
        def synthesize_wav(self, text, wav_file, syn_config=None):
            # Write dummy wav frames directly to the Wave_write object
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b'\x00' * 100)
    
    monkeypatch.setattr(PiperVoice, "load", lambda *args, **kwargs: DummyVoice())

    # Use restored LibriTTS model for alloy
    payload = {
        "model": "en_US-libritts_r-medium",
        "input": "Testing dynamic loading of LibriTTS restored voice.",
        "voice": "alloy",
        "response_format": "wav"
    }
    r = client.post("/audio/speech", json=payload)
    assert r.status_code == 200, f"Failed with {r.text}"
    assert len(r.content) > 0, "Returned empty audio"
