import os
import re
import uuid
import shutil
import tempfile
import subprocess
import threading
import datetime
from typing import Dict, List, Any, Optional

# Base directories
DATASETS_DIR = os.environ.get("DATASETS_DIR", os.path.join(os.getcwd(), "data", "training_datasets"))

# In-memory jobs tracking
_jobs_lock = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}

def find_ffmpeg() -> str:
    """Locate the ffmpeg binary on the system (supporting local Windows and Docker environments)."""
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    if os.name == "nt":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            winget_root = os.path.join(user_profile, ".winget_portable_root")
            if os.path.exists(winget_root):
                for root, _, files in os.walk(winget_root):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")
    return "ffmpeg"

def clean_text_for_piper(text: str) -> str:
    """Cleans up whisper transcription text by removing bracketed cues and normalizing whitespace."""
    if not text:
        return ""
    # Remove bracketed noises/notes like [music], (applause), etc.
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def update_job(job_id: str, **kwargs):
    """Thread-safe update of a job status."""
    with _jobs_lock:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)

def add_job_log(job_id: str, message: str):
    """Thread-safe addition of a log message to a job."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with _jobs_lock:
        if job_id in JOBS:
            JOBS[job_id]["logs"].append(formatted_msg)

def run_dataset_generation(
    job_id: str,
    youtube_url: str,
    dataset_name: str,
    language: str,
    whisper_model_size: str
):
    """Background task to download and segment YouTube video."""
    temp_dir = None
    try:
        add_job_log(job_id, f"Starting dataset generation for '{dataset_name}' using '{youtube_url}'")
        
        # 1. Setup folders
        dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
        dataset_wav_dir = os.path.join(dataset_dir, "wav")
        os.makedirs(dataset_wav_dir, exist_ok=True)
        
        # Temp dir for processing
        temp_dir = tempfile.mkdtemp()
        add_job_log(job_id, f"Created temporary workspace at {temp_dir}")

        # 2. Download YouTube Audio
        update_job(job_id, status="downloading", progress=10)
        add_job_log(job_id, "Downloading YouTube audio via yt-dlp...")
        
        import yt_dlp
        temp_audio_template = os.path.join(temp_dir, "audio.%(ext)s")
        
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "outtmpl": temp_audio_template,
            "quiet": True,
            "no_warnings": True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            
        downloaded_wav = os.path.join(temp_dir, "audio.wav")
        if not os.path.exists(downloaded_wav):
            raise FileNotFoundError("yt-dlp failed to output a WAV file.")
            
        add_job_log(job_id, "YouTube audio downloaded successfully.")

        # 3. Convert to Mono 22050Hz 16-bit WAV
        update_job(job_id, status="processing_audio", progress=30)
        add_job_log(job_id, "Converting audio format to mono 22.05 kHz 16-bit PCM WAV...")
        
        ffmpeg_bin = find_ffmpeg()
        converted_wav = os.path.join(temp_dir, "audio_22050.wav")
        
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", downloaded_wav,
            "-ac", "1",
            "-ar", "22050",
            "-sample_fmt", "s16",
            converted_wav
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio conversion failed: {result.stderr.decode('utf-8')}")
            
        add_job_log(job_id, "Audio conversion finished.")

        # 4. Transcribe using Whisper
        update_job(job_id, status="transcribing", progress=50)
        add_job_log(job_id, f"Loading Whisper model '{whisper_model_size}' (this may take a minute first time)...")
        
        import whisper
        model = whisper.load_model(whisper_model_size)
        
        add_job_log(job_id, "Transcribing and extracting segment timestamps...")
        # Run transcription
        result = model.transcribe(converted_wav, language=language.split("_")[0])
        segments = result.get("segments", [])
        
        add_job_log(job_id, f"Transcribed {len(segments)} segments.")

        # 5. Segment and generate training files
        update_job(job_id, status="segmenting", progress=75)
        add_job_log(job_id, "Segmenting audio and writing metadata.csv...")
        
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        num_saved = 0
        # Open in append mode if files exist, or write mode if new
        mode = "a" if os.path.exists(metadata_path) else "w"
        
        # Count existing segments to prevent overwriting during sequential runs
        existing_segments = 0
        if mode == "a":
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    existing_segments = len(f.readlines())
            except Exception:
                pass
                
        with open(metadata_path, mode, encoding="utf-8") as meta_file:
            for i, seg in enumerate(segments):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                text = seg.get("text", "").strip()
                
                cleaned_text = clean_text_for_piper(text)
                # Skip empty transcription segments or segments shorter than 0.5s or longer than 15s
                duration = end - start
                if not cleaned_text or duration < 0.5 or duration > 15.0:
                    continue
                    
                seg_idx = existing_segments + num_saved + 1
                segment_name = f"segment_{seg_idx:05d}"
                segment_filename = f"{segment_name}.wav"
                segment_path = os.path.join(dataset_wav_dir, segment_filename)
                
                # Slice segment
                slice_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}",
                    "-i", converted_wav,
                    "-ac", "1",
                    "-ar", "22050",
                    "-sample_fmt", "s16",
                    segment_path
                ]
                
                slice_res = subprocess.run(slice_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if slice_res.returncode == 0:
                    # Write metadata line: filename|text
                    meta_file.write(f"{segment_name}|{cleaned_text}\n")
                    num_saved += 1
                else:
                    add_job_log(job_id, f"Warning: Failed to slice segment {segment_name} ({start:.2f}s - {end:.2f}s)")
        
        add_job_log(job_id, f"Dataset generation completed. Generated {num_saved} audio segments.")
        update_job(
            job_id,
            status="completed",
            progress=100,
            completed_at=datetime.datetime.now().isoformat(),
            num_segments=num_saved
        )
        
    except Exception as e:
        error_msg = str(e)
        add_job_log(job_id, f"ERROR occurred: {error_msg}")
        update_job(
            job_id,
            status="failed",
            error=error_msg,
            completed_at=datetime.datetime.now().isoformat()
        )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                add_job_log(job_id, "Cleaned up temporary workspace.")
            except Exception as e:
                add_job_log(job_id, f"Warning: Failed to delete temp dir {temp_dir}: {e}")

def create_dataset_job(
    youtube_url: str,
    dataset_name: str,
    language: str = "en_US",
    whisper_model_size: str = "base"
) -> str:
    """Registers and starts a dataset generation background thread."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    
    # Clean dataset name to prevent directory traversal
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    if not clean_name:
        clean_name = "youtube_dataset"
        
    with _jobs_lock:
        JOBS[job_id] = {
            "id": job_id,
            "youtube_url": youtube_url,
            "dataset_name": clean_name,
            "language": language,
            "whisper_model_size": whisper_model_size,
            "status": "pending",
            "progress": 0,
            "error": None,
            "created_at": datetime.datetime.now().isoformat(),
            "completed_at": None,
            "num_segments": 0,
            "logs": []
        }
        
    # Start thread
    thread = threading.Thread(
        target=run_dataset_generation,
        args=(job_id, youtube_url, clean_name, language, whisper_model_size)
    )
    thread.daemon = True
    thread.start()
    
    return job_id

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status and logs details."""
    with _jobs_lock:
        return JOBS.get(job_id)

def get_all_jobs() -> List[Dict[str, Any]]:
    """Get all registered jobs."""
    with _jobs_lock:
        # Sort jobs by creation time descending
        return sorted(JOBS.values(), key=lambda x: x["created_at"], reverse=True)

def list_datasets() -> List[Dict[str, Any]]:
    """List all available datasets in the local datasets directory."""
    if not os.path.exists(DATASETS_DIR):
        return []
        
    datasets = []
    for entry in os.scandir(DATASETS_DIR):
        if entry.is_dir():
            metadata_file = os.path.join(entry.path, "metadata.csv")
            wav_dir = os.path.join(entry.path, "wav")
            
            num_samples = 0
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        num_samples = len(f.readlines())
                except Exception:
                    pass
                    
            datasets.append({
                "name": entry.name,
                "path": entry.path,
                "num_samples": num_samples,
                "has_wavs": os.path.exists(wav_dir) and len(os.listdir(wav_dir)) > 0
            })
    return datasets

def delete_dataset(dataset_name: str) -> bool:
    """Deletes a training dataset from disk."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    path = os.path.join(DATASETS_DIR, clean_name)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        return True
    return False
