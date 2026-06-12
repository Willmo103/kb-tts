from kb_tts.api import resolve_voice_params

def test_resolve_standard_voices():
    # Test mapping of standard OpenAI voices
    model, speaker = resolve_voice_params("alloy")
    assert model == "en_US-libritts-high"
    assert speaker == 0

    model, speaker = resolve_voice_params("echo")
    assert model == "en_US-libritts-high"
    assert speaker == 1

    model, speaker = resolve_voice_params("shimmer")
    assert model == "en_US-libritts-high"
    assert speaker == 5

def test_resolve_numeric_speaker():
    # Test numeric speaker ID mapping to default model
    model, speaker = resolve_voice_params("42")
    assert model == "en_US-libritts-high"
    assert speaker == 42

def test_resolve_custom_model_and_speaker():
    # Test colon-separated custom model and speaker
    model, speaker = resolve_voice_params("en_US-libritts_r-medium:10")
    assert model == "en_US-libritts_r-medium"
    assert speaker == 10

def test_resolve_fallback_model():
    # Test fallback to full model name
    model, speaker = resolve_voice_params("en_US-amy-medium")
    assert model == "en_US-amy-medium"
    assert speaker is None
