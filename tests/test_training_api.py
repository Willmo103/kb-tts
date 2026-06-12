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
