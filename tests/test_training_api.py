from fastapi.testclient import TestClient
from kb_tts.api import app
from kb_tts.training.data_generator import JOBS, _jobs_lock

client = TestClient(app)

def test_training_dashboard_endpoint():
    """Verify that the dashboard HTML page is served correctly."""
    response = client.get("/training")
    assert response.status_code == 200
    assert "Piper TTS Training Dashboard" in response.text

def test_training_jobs_list_empty():
    """Verify that fetching the jobs list is successful."""
    response = client.get("/api/training/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_generate_endpoint_invalid_url():
    """Verify URL validation for youtube_url."""
    payload = {
        "youtube_url": "invalid_url",
        "dataset_name": "test_dataset",
        "whisper_model_size": "tiny",
        "language": "en_US"
    }
    response = client.post("/api/training/generate", json=payload)
    assert response.status_code == 400
    assert "Invalid YouTube URL" in response.json()["detail"]

def test_generate_endpoint_success(monkeypatch):
    """Verify registering a generation job with mock job creation."""
    import kb_tts.training.api
    
    mock_job_id = "job_test_12345"
    
    def mock_create_job(youtube_url, dataset_name, language, whisper_model_size):
        return mock_job_id
        
    monkeypatch.setattr(kb_tts.training.api, "create_dataset_job", mock_create_job)
    
    payload = {
        "youtube_url": "https://www.youtube.com/watch?v=dummy_id",
        "dataset_name": "test_dataset",
        "whisper_model_size": "tiny",
        "language": "en_US"
    }
    response = client.post("/api/training/generate", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["job_id"] == mock_job_id

def test_get_job_status_not_found():
    """Verify 404 response on unknown job IDs."""
    response = client.get("/api/training/jobs/nonexistent_job")
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]

def test_get_datasets_list():
    """Verify that fetching dataset folders on disk returns a list."""
    response = client.get("/api/training/datasets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_voices_list():
    """Verify that fetching custom models lists returns a list."""
    response = client.get("/api/training/voices")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_whisper_transcription_api(monkeypatch):
    """Verify OpenAI-compliant Whisper transcription with JSON and SRT output formats using mocked model loading."""
    import whisper
    
    class MockWhisperModel:
        def transcribe(self, audio_path, **kwargs):
            return {
                "text": "This is a mock transcription.",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "This is a mock transcription."}
                ]
            }
            
    def mock_load_model(model_size):
        return MockWhisperModel()
        
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    
    files = {
        "file": ("test.wav", b"riff wave header content dummy", "audio/wav")
    }
    data = {
        "model": "tiny",
        "response_format": "json"
    }
    
    # Test JSON transcription response
    response = client.post("/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["text"] == "This is a mock transcription."
    
    # Test SRT transcription response
    data["response_format"] = "srt"
    response_srt = client.post("/audio/transcriptions", files=files, data=data)
    assert response_srt.status_code == 200
    assert "1\n00:00:00,000 --> 00:00:02,000\nThis is a mock transcription." in response_srt.text

def test_training_run_endpoints(monkeypatch):
    """Verify the start, stop, status, checkpoints list, and ONNX export endpoints for baremetal training runs."""
    import kb_tts.training.api
    
    monkeypatch.setattr(kb_tts.training.api, "start_training", lambda **k: True)
    monkeypatch.setattr(kb_tts.training.api, "get_run", lambda dataset_name: {
        "dataset_name": dataset_name,
        "status": "training",
        "progress": 55,
        "current_epoch": 10,
        "total_epochs": 100,
        "logs": ["Preprocessing finished.", "Training epoch 10..."]
    })
    monkeypatch.setattr(kb_tts.training.api, "stop_training", lambda dataset_name: True)
    monkeypatch.setattr(kb_tts.training.api, "list_checkpoints", lambda dataset_name: [
        {"filename": "epoch=9.ckpt", "rel_path": "lightning_logs/epoch=9.ckpt", "size_bytes": 1000, "created_at": "2026-06-12T00:00:00"}
    ])
    monkeypatch.setattr(kb_tts.training.api, "trigger_onnx_export", lambda **k: True)
    
    # 1. Trigger run
    run_payload = {
        "dataset_name": "test_dataset",
        "epochs": 100,
        "batch_size": 16,
        "device": "cpu"
    }
    res = client.post("/api/training/run", json=run_payload)
    assert res.status_code == 200
    assert "triggered" in res.json()["message"]
    
    # 2. Get status
    res = client.get("/api/training/run/status/test_dataset")
    assert res.status_code == 200
    assert res.json()["status"] == "training"
    assert res.json()["progress"] == 55
    
    # 3. Stop run
    res = client.post("/api/training/run/stop/test_dataset")
    assert res.status_code == 200
    assert "termination triggered" in res.json()["message"]
    
    # 4. List checkpoints
    res = client.get("/api/training/run/checkpoints/test_dataset")
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["filename"] == "epoch=9.ckpt"
    
    # 5. Export checkpoints to ONNX
    export_payload = {
        "dataset_name": "test_dataset",
        "checkpoint_rel_path": "lightning_logs/epoch=9.ckpt",
        "model_name": "test_model"
    }
    res = client.post("/api/training/run/export", json=export_payload)
    assert res.status_code == 200
    assert "Export triggered" in res.json()["message"]
