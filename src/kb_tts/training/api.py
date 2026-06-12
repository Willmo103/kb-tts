import os
import re
import json
import shutil
import tempfile
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from kb_tts.download import TTS_CONFIG_PATH, VOICES_DIR, load_config
from kb_tts.training.data_generator import (
    create_dataset_job, get_job, get_all_jobs,
    list_datasets, delete_dataset, DATASETS_DIR
)
from kb_tts.training.trainer import (
    start_training, stop_training, get_run,
    list_checkpoints, trigger_onnx_export
)

# Initialize APIRouter
router = APIRouter()

# Schema models
class TrainingRunRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the generated dataset to train on.")
    base_checkpoint_url: Optional[str] = Field(default=None, description="Optional URL to download a pre-trained base model checkpoint.")
    epochs: int = Field(default=100, description="Number of training epochs.")
    batch_size: int = Field(default=16, description="Training batch size.")
    device: str = Field(default="cpu", description="Hardware device to use: cpu or cuda/gpu.")

class ExportRunRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the training dataset.")
    checkpoint_rel_path: str = Field(..., description="Relative path of the checkpoint to export (e.g. lightning_logs/version_0/checkpoints/epoch=99.ckpt).")
    model_name: str = Field(..., description="Target custom ONNX model filename.")

# Schema models
class DatasetGenerateRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube URL of the speaker's audio.")
    dataset_name: str = Field(..., description="Name of the training dataset.")
    language: str = Field(default="en_US", description="Language tag (e.g. en_US, de_DE).")
    whisper_model_size: str = Field(default="base", description="Whisper model size: tiny, base, small, medium, large.")

class VoiceRegisterRequest(BaseModel):
    voice_name: str = Field(..., description="Unique voice mapping key/alias (e.g., my_voice).")
    model_name: str = Field(..., description="Target custom ONNX model filename (without extension).")
    speaker_id: int = Field(default=0, description="Speaker index within the model (typically 0 for single speaker).")

def save_config(config: Dict[str, Any]):
    """Saves updated JSON configuration to disk."""
    try:
        os.makedirs(os.path.dirname(TTS_CONFIG_PATH), exist_ok=True)
        with open(TTS_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write configuration: {e}")

# ==================== DATA GENERATION API ====================

@router.post("/api/training/generate")
def generate_dataset(request: DatasetGenerateRequest):
    """Starts a background task to download YouTube audio, transcribe, and segment it."""
    # Basic URL validation
    if not request.youtube_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL. Must start with http/https.")
        
    try:
        job_id = create_dataset_job(
            youtube_url=request.youtube_url,
            dataset_name=request.dataset_name,
            language=request.language,
            whisper_model_size=request.whisper_model_size
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/training/jobs")
def get_jobs_list():
    """Lists all active and completed dataset generation jobs."""
    return get_all_jobs()

@router.get("/api/training/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieves status and progress logs for a specific job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/api/training/datasets")
def get_datasets_list():
    """Lists all generated datasets available on disk."""
    return list_datasets()

@router.delete("/api/training/datasets/{dataset_name}")
def remove_dataset(dataset_name: str):
    """Deletes a generated dataset folder from disk."""
    deleted = delete_dataset(dataset_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found or could not be deleted")
    return {"status": "success", "message": f"Dataset '{dataset_name}' deleted."}

@router.get("/api/training/datasets/{dataset_name}/download")
def download_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """Packages a generated dataset into a ZIP file and serves it for download."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    dataset_path = os.path.join(DATASETS_DIR, clean_name)
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Create temp directory for ZIP
    temp_zip_dir = tempfile.mkdtemp()
    zip_base_name = os.path.join(temp_zip_dir, clean_name)
    
    try:
        shutil.make_archive(zip_base_name, "zip", dataset_path)
    except Exception as e:
        shutil.rmtree(temp_zip_dir)
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP archive: {e}")
        
    zip_file_path = f"{zip_base_name}.zip"
    
    # Cleanup temp folder after response completes
    def cleanup():
        try:
            shutil.rmtree(temp_zip_dir)
        except Exception:
            pass
            
    background_tasks.add_task(cleanup)
    
    return FileResponse(
        zip_file_path,
        media_type="application/zip",
        filename=f"{clean_name}.zip"
    )

# ==================== CUSTOM VOICE MODELS API ====================

@router.get("/api/training/voices")
def get_custom_voices():
    """Lists all local voice models inside VOICES_DIR and their aliases."""
    models = []
    if os.path.exists(VOICES_DIR):
        for file in os.listdir(VOICES_DIR):
            if file.endswith(".onnx"):
                model_name = file[:-5]
                onnx_path = os.path.join(VOICES_DIR, file)
                json_path = os.path.join(VOICES_DIR, f"{file}.json")
                size = os.path.getsize(onnx_path)
                
                # Check voice mappings
                config = load_config()
                voice_map = config.get("voice_map", {})
                mappings = []
                for voice_alias, mapping in voice_map.items():
                    if mapping.get("model") == model_name:
                        mappings.append({
                            "alias": voice_alias,
                            "speaker_id": mapping.get("speaker_id", 0)
                        })
                        
                models.append({
                    "model_name": model_name,
                    "size_bytes": size,
                    "has_config": os.path.exists(json_path),
                    "mappings": mappings
                })
    return models

@router.post("/api/training/voices/upload")
async def upload_voice(
    model_file: UploadFile = File(..., description="The custom model .onnx file."),
    config_file: UploadFile = File(..., description="The custom model config .onnx.json file."),
    model_name: Optional[str] = Form(None, description="Optional custom model name override.")
):
    """Uploads a trained custom model (.onnx and .onnx.json)."""
    if not model_file.filename.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Model file must end with .onnx")
    if not config_file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Config file must end with .json or .onnx.json")
        
    if not model_name:
        model_name = model_file.filename[:-5] # strip .onnx
        
    model_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
    if not model_name:
        raise HTTPException(status_code=400, detail="Invalid model name")
        
    os.makedirs(VOICES_DIR, exist_ok=True)
    onnx_path = os.path.join(VOICES_DIR, f"{model_name}.onnx")
    json_path = os.path.join(VOICES_DIR, f"{model_name}.onnx.json")
    
    try:
        with open(onnx_path, "wb") as f:
            f.write(await model_file.read())
        with open(json_path, "wb") as f:
            f.write(await config_file.read())
    except Exception as e:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")
        
    return {"status": "success", "model_name": model_name}

@router.post("/api/training/voices/register")
def register_voice(request: VoiceRegisterRequest):
    """Maps a custom voice alias to a model name in voices_config.json."""
    config = load_config()
    voice_map = config.setdefault("voice_map", {})
    
    # Check if files exist
    onnx_path = os.path.join(VOICES_DIR, f"{request.model_name}.onnx")
    if not os.path.exists(onnx_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_name}' does not exist locally. Upload it first."
        )
        
    voice_map[request.voice_name] = {
        "model": request.model_name,
        "speaker_id": request.speaker_id
    }
    
    save_config(config)
    return {"status": "success", "voice_name": request.voice_name, "mapping": voice_map[request.voice_name]}

@router.delete("/api/training/voices/{model_name}")
def delete_voice(model_name: str):
    """Deletes a custom model and removes its mappings from voices_config.json."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
    onnx_path = os.path.join(VOICES_DIR, f"{clean_name}.onnx")
    json_path = os.path.join(VOICES_DIR, f"{clean_name}.onnx.json")
    
    deleted = False
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        deleted = True
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted = True
        
    # Clean mappings
    config = load_config()
    voice_map = config.get("voice_map", {})
    updated_map = {k: v for k, v in voice_map.items() if v.get("model") != clean_name}
    
    if len(voice_map) != len(updated_map):
        config["voice_map"] = updated_map
        save_config(config)
        deleted = True
        
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice model not found")
        
    return {"status": "success", "message": f"Voice model '{model_name}' and mappings deleted."}

# ==================== TRAINING EXECUTION API ====================

@router.post("/api/training/run")
def trigger_training(request: TrainingRunRequest):
    """Triggers the baremetal fine-tuning pipeline for a specific dataset."""
    started = start_training(
        dataset_name=request.dataset_name,
        base_checkpoint_url=request.base_checkpoint_url,
        epochs=request.epochs,
        batch_size=request.batch_size,
        device=request.device
    )
    if not started:
        raise HTTPException(
            status_code=400,
            detail=f"Training run for dataset '{request.dataset_name}' is already active."
        )
    return {"status": "success", "message": f"Training pipeline triggered for '{request.dataset_name}'."}


@router.get("/api/training/run/status/{dataset_name}")
def check_training_status(dataset_name: str):
    """Retrieves status, epoch, progress and console logs for a training run."""
    run = get_run(dataset_name)
    if not run:
        raise HTTPException(status_code=404, detail="No active or previous run found for this dataset.")
    return run


@router.post("/api/training/run/stop/{dataset_name}")
def stop_training_run(dataset_name: str):
    """Kills the active subprocess training execution."""
    stopped = stop_training(dataset_name)
    if not stopped:
        raise HTTPException(status_code=400, detail="Run not found or already stopped.")
    return {"status": "success", "message": "Training run termination triggered."}


@router.get("/api/training/run/checkpoints/{dataset_name}")
def get_dataset_checkpoints(dataset_name: str):
    """Scans and lists generated model checkpoints (.ckpt) for a dataset."""
    return list_checkpoints(dataset_name)


@router.post("/api/training/run/export")
def export_checkpoint(request: ExportRunRequest):
    """Triggers the ONNX export process for a checkpoint in the background."""
    triggered = trigger_onnx_export(
        dataset_name=request.dataset_name,
        checkpoint_rel_path=request.checkpoint_rel_path,
        model_name=request.model_name
    )
    if not triggered:
        raise HTTPException(status_code=404, detail="Checkpoint file not found.")
    return {"status": "success", "message": f"Export triggered for model '{request.model_name}'."}


# ==================== DASHBOARD UI ====================

@router.get("/training", response_class=HTMLResponse)
def get_training_dashboard():
    """Serves the training UI dashboard HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found")
        
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load UI template: {e}")
