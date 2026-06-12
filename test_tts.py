import os
import sys

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    print("Testing /health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("Health Check SUCCESS:")
            print(r.json())
            return True
        else:
            print(f"Health Check FAILED: Status code {r.status_code}")
            print(r.text)
            return False
    except Exception as e:
        print(f"Health Check FAILED: Could not connect to server. Error: {e}")
        return False


def test_speech(voice, response_format, text, filename, speed=1.0):
    print(
        f"Generating speech for voice='{voice}', format='{response_format}', speed={speed}..."
    )
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }

    try:
        r = requests.post(f"{BASE_URL}/audio/speech", json=payload)
        if r.status_code == 200:
            output_dir = "test_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "wb") as f:
                f.write(r.content)
            print(
                f"SUCCESS: Audio written to {output_path} (Size: {len(r.content)} bytes)"
            )
            return True
        else:
            print(f"FAILED: Status code {r.status_code}")
            print(r.text)
            return False
    except Exception as e:
        print(f"FAILED: Request exception: {e}")
        return False


def main():
    print("Starting TTS server validation tests...")

    if not test_health():
        print("Server is not running. Please start the server first.")
        sys.exit(1)

    # Test 1: Mapped voice "alloy" -> MP3 format
    test_speech(
        voice="alloy",
        response_format="mp3",
        text="Hello! This is a test of the alloy voice generated as an MP3 file using local Piper text-to-speech.",
        filename="test_alloy_0.mp3",
    )

    # Test 2: Mapped voice "alloy" -> WAV format
    test_speech(
        voice="alloy",
        response_format="wav",
        text="This is alloy speaking again, this time as a standard WAV file.",
        filename="test_alloy_0.wav",
    )

    # Test 3: Specific speaker numeric ID "15" in default model -> MP3 format
    test_speech(
        voice="15",
        response_format="mp3",
        text="This is speaker fifteen, configured via a numeric voice ID, generated as an MP3.",
        filename="test_speaker_15.mp3",
    )

    # Test 4: Custom model and speaker ID "en_US-libritts-high:102"
    test_speech(
        voice="en_US-libritts-high:102",
        response_format="mp3",
        text="This is speaker one hundred and two from the LibriTTS high quality model, generated as an MP3.",
        filename="test_libritts_102.mp3",
    )

    # Test 5: Speed parameter (fast speed 1.5x)
    test_speech(
        voice="alloy",
        response_format="mp3",
        text="This is the alloy voice speaking very quickly at one point five times speed.",
        filename="test_alloy_fast.mp3",
        speed=1.5,
    )

    # Test 6: Speed parameter (slow speed 0.7x)
    test_speech(
        voice="alloy",
        response_format="mp3",
        text="This is the alloy voice speaking slowly at zero point seven times speed.",
        filename="test_alloy_slow.mp3",
        speed=0.7,
    )


if __name__ == "__main__":
    main()
