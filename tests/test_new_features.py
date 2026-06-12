import os
import requests

BASE_URL = "http://localhost:8000"

def test_markdown_cleaning_api():
    print("Testing /v1/audio/speech with markdown cleaning...")
    
    # 1. Clean markdown (default or explicit)
    payload = {
        "model": "tts-1",
        "input": "This is **bold** text and `inline code`.",
        "voice": "alloy",
        "response_format": "mp3",
        "clean_markdown": True
    }
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    assert r.status_code == 200, f"Failed with {r.text}"
    print(f"Clean markdown request successful. Audio size: {len(r.content)} bytes")

    # 2. Raw markdown (no cleaning)
    payload["clean_markdown"] = False
    r_raw = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    assert r_raw.status_code == 200, f"Failed with {r_raw.text}"
    print(f"Raw markdown request successful. Audio size: {len(r_raw.content)} bytes")

def test_custom_model_override_api():
    print("Testing custom model override...")
    
    # Use restored LibriTTS model for alloy
    payload = {
        "model": "en_US-libritts_r-medium",
        "input": "Testing dynamic loading of LibriTTS restored voice.",
        "voice": "alloy",
        "response_format": "mp3"
    }
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    assert r.status_code == 200, f"Failed with {r.text}"
    print(f"Custom model request successful. Audio size: {len(r.content)} bytes")

if __name__ == "__main__":
    test_markdown_cleaning_api()
    test_custom_model_override_api()
    print("ALL API NEW FEATURE TESTS PASSED!")
