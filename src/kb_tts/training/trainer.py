import os
import re
import sys
import shutil
import urllib.request
import subprocess
import threading
import datetime
from typing import Dict, List, Any, Optional

from kb_tts.training.data_generator import DATASETS_DIR, find_ffmpeg

# Paths
TRAINING_RUNS_DIR = os.environ.get("TRAINING_RUNS_DIR", os.path.join(os.getcwd(), "data", "training_runs"))
CHECKPOINTS_DIR = os.path.join(TRAINING_RUNS_DIR, "checkpoints")

# Thread-safe runs tracking
_runs_lock = threading.Lock()
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}

def add_run_log(dataset_name: str, message: str):
    """Thread-safe logging of training controller operations."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with _runs_lock:
        if dataset_name in ACTIVE_RUNS:
            ACTIVE_RUNS[dataset_name]["logs"].append(formatted_msg)

def update_run(dataset_name: str, **kwargs):
    """Thread-safe update of an active training run status."""
    with _runs_lock:
        if dataset_name in ACTIVE_RUNS:
            ACTIVE_RUNS[dataset_name].update(kwargs)

def download_file_with_progress(url: str, dest_path: str, dataset_name: str):
    """Downloads a file with HTTP progress logged to the job runs console."""
    add_run_log(dataset_name, f"Downloading base checkpoint from {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 64
        
        with open(dest_path, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    if percent % 10 == 0:  # Log every 10%
                        update_run(dataset_name, progress=30 + int(percent * 0.2)) # Maps to 30%-50% progress bar range
        
    add_run_log(dataset_name, f"Base checkpoint successfully saved to {dest_path}")

def get_subprocess_env() -> Dict[str, str]:
    """Generates the environment variables including the Piper codebase path in PYTHONPATH."""
    env = os.environ.copy()
    piper_path = os.path.abspath(os.path.join(os.getcwd(), "piper", "src", "python"))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{piper_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = piper_path
    return env

def run_training_pipeline_thread(
    dataset_name: str,
    base_checkpoint_url: str,
    epochs: int,
    batch_size: int,
    device: str
):
    """Runs the baremetal training pipeline in a background thread."""
    proc = None
    processed_dir = os.path.join(TRAINING_RUNS_DIR, f"{dataset_name}_processed")
    os.makedirs(TRAINING_RUNS_DIR, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if the piper repository has been cloned in the workspace
    piper_path = os.path.abspath(os.path.join(os.getcwd(), "piper", "src", "python"))
    if not os.path.exists(piper_path):
        add_run_log(
            dataset_name,
            "WARNING: './piper' folder not found in workspace. "
            "Make sure to clone the repository first: "
            "git clone https://github.com/rhasspy/piper.git ./piper "
            "and build monotonic_align to run baremetal training."
        )
    
    try:
        # 1. Preprocessing stage
        add_run_log(dataset_name, "Starting Preprocessing...")
        update_run(dataset_name, status="preprocessing", progress=10)
        
        # Run python -m piper_train.preprocess
        preprocess_cmd = [
            sys.executable, "-m", "piper_train.preprocess",
            "--language", "en_US",
            "--input-dir", os.path.join(DATASETS_DIR, dataset_name),
            "--output-dir", processed_dir,
            "--dataset-format", "ljspeech",
            "--sample-rate", "22050"
        ]
        
        add_run_log(dataset_name, f"Executing command: {' '.join(preprocess_cmd)}")
        proc = subprocess.Popen(
            preprocess_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=get_subprocess_env()
        )
        
        update_run(dataset_name, process=proc)
        
        # Stream logs
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                add_run_log(dataset_name, line.strip())
                
        if proc.returncode != 0:
            raise RuntimeError(f"Preprocessing subprocess failed with exit code {proc.returncode}")
            
        add_run_log(dataset_name, "Preprocessing successfully finished.")

        # 2. Checkpoint downloading stage
        checkpoint_path = ""
        if base_checkpoint_url:
            update_run(dataset_name, status="downloading_base_model", progress=30)
            
            # Resolve filename
            filename = base_checkpoint_url.split("/")[-1]
            if not filename.endswith(".ckpt"):
                filename = "base_model.ckpt"
                
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, filename)
            if not os.path.exists(checkpoint_path):
                download_file_with_progress(base_checkpoint_url, checkpoint_path, dataset_name)
            else:
                add_run_log(dataset_name, f"Pre-trained base checkpoint already exists at {checkpoint_path}")

        # 3. Fine-Tuning/Training stage
        update_run(dataset_name, status="training", progress=50)
        add_run_log(dataset_name, "Starting Piper Fine-Tuning process...")
        
        train_cmd = [
            sys.executable, "-m", "piper_train",
            "--dataset-dir", processed_dir,
            "--accelerator", device,
            "--devices", "1",
            "--batch-size", str(batch_size),
            "--max_epochs", str(epochs)
        ]
        
        if checkpoint_path:
            train_cmd.extend(["--resume_from_checkpoint", checkpoint_path])
            
        add_run_log(dataset_name, f"Executing command: {' '.join(train_cmd)}")
        
        # Save logs to a training.log file inside processed_dir for audit trails
        log_file_path = os.path.join(processed_dir, "training.log")
        
        proc = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=get_subprocess_env()
        )
        
        update_run(dataset_name, process=proc)
        
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    stripped = line.strip()
                    log_file.write(line)
                    log_file.flush()
                    add_run_log(dataset_name, stripped)
                    
                    # Parse current epoch to display in metadata status
                    epoch_match = re.search(r"Epoch\s+(\d+)", stripped, re.IGNORECASE)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        update_run(dataset_name, current_epoch=current_epoch)
                        # Scale progress dynamically from 50% to 95% based on training epoch percentage
                        train_progress = 50 + int((current_epoch / epochs) * 45)
                        update_run(dataset_name, progress=min(train_progress, 95))
                        
        if proc.returncode != 0:
            # Check if terminated by user
            with _runs_lock:
                was_killed = ACTIVE_RUNS.get(dataset_name, {}).get("status") == "stopping"
            if was_killed:
                add_run_log(dataset_name, "Training terminated by user.")
                update_run(
                    dataset_name,
                    status="stopped",
                    completed_at=datetime.datetime.now().isoformat()
                )
                return
            else:
                raise RuntimeError(f"Training subprocess failed with exit code {proc.returncode}")
                
        add_run_log(dataset_name, "Training process finished successfully!")
        update_run(
            dataset_name,
            status="completed",
            progress=100,
            completed_at=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = str(e)
        add_run_log(dataset_name, f"ERROR occurred: {error_msg}")
        update_run(
            dataset_name,
            status="failed",
            error=error_msg,
            completed_at=datetime.datetime.now().isoformat()
        )
    finally:
        # Clear running process mapping
        update_run(dataset_name, process=None)

def start_training(
    dataset_name: str,
    base_checkpoint_url: str,
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "cpu"
) -> bool:
    """Checks and triggers a thread to run the subprocess pipeline."""
    with _runs_lock:
        # Check if already running
        if dataset_name in ACTIVE_RUNS and ACTIVE_RUNS[dataset_name]["status"] in ["pending", "preprocessing", "downloading_base_model", "training"]:
            return False
            
        # Clean dataset name to avoid folder traversal
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
        
        ACTIVE_RUNS[clean_name] = {
            "dataset_name": clean_name,
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "error": None,
            "created_at": datetime.datetime.now().isoformat(),
            "completed_at": None,
            "logs": [],
            "process": None
        }
        
    thread = threading.Thread(
        target=run_training_pipeline_thread,
        args=(clean_name, base_checkpoint_url, epochs, batch_size, device)
    )
    thread.daemon = True
    thread.start()
    return True

def stop_training(dataset_name: str) -> bool:
    """Terminates the active training process."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    with _runs_lock:
        if clean_name not in ACTIVE_RUNS:
            return False
            
        run = ACTIVE_RUNS[clean_name]
        if run["status"] in ["completed", "failed", "stopped"]:
            return False
            
        # Update status to indicate manual stopping
        run["status"] = "stopping"
        proc: Optional[subprocess.Popen] = run.get("process")
        
    if proc:
        try:
            add_run_log(clean_name, "Terminating training process subprocess...")
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return True

def get_run(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Gets training run status details (excluding raw process pointer)."""
    with _runs_lock:
        run = ACTIVE_RUNS.get(dataset_name)
        if not run:
            return None
        # Copy to return without the process pointer to support JSON serialization
        run_copy = run.copy()
        if "process" in run_copy:
            del run_copy["process"]
        return run_copy

def list_checkpoints(dataset_name: str) -> List[Dict[str, Any]]:
    """Lists generated .ckpt checkpoints inside the training runs folder."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    processed_dir = os.path.join(TRAINING_RUNS_DIR, f"{clean_name}_processed")
    logs_dir = os.path.join(processed_dir, "lightning_logs")
    
    checkpoints = []
    if not os.path.exists(logs_dir):
        return []
        
    for root, _, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".ckpt"):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, processed_dir)
                size = os.path.getsize(path)
                checkpoints.append({
                    "filename": file,
                    "rel_path": rel_path.replace("\\", "/"),
                    "size_bytes": size,
                    "created_at": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                })
    return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

def export_onnx_subprocess(
    dataset_name: str,
    checkpoint_rel_path: str,
    model_name: str
):
    """Orchestrates checkpoint ONNX conversions inside a background thread."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    clean_model_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
    
    processed_dir = os.path.join(TRAINING_RUNS_DIR, f"{clean_name}_processed")
    ckpt_path = os.path.join(processed_dir, checkpoint_rel_path)
    
    from kb_tts.download import VOICES_DIR
    os.makedirs(VOICES_DIR, exist_ok=True)
    
    onnx_dest = os.path.join(VOICES_DIR, f"{clean_model_name}.onnx")
    
    try:
        add_run_log(clean_name, f"Exporting checkpoint '{checkpoint_rel_path}' to ONNX '{clean_model_name}'...")
        
        export_cmd = [
            sys.executable, "-m", "piper_train.export_onnx",
            "--checkpoint", ckpt_path,
            "--output-file", onnx_dest
        ]
        
        add_run_log(clean_name, f"Executing export command: {' '.join(export_cmd)}")
        res = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=get_subprocess_env())
        
        if res.returncode == 0:
            add_run_log(clean_name, f"ONNX model exported successfully to: {onnx_dest}")
        else:
            raise RuntimeError(f"Export utility failed with error: {res.stderr}")
            
    except Exception as e:
        add_run_log(clean_name, f"Failed to export ONNX: {e}")
        # Clear files on failure to avoid corruption
        if os.path.exists(onnx_dest):
            os.remove(onnx_dest)
        json_dest = f"{onnx_dest}.json"
        if os.path.exists(json_dest):
            os.remove(json_dest)

def trigger_onnx_export(
    dataset_name: str,
    checkpoint_rel_path: str,
    model_name: str
) -> bool:
    """Launches the Popen execution thread for ONNX conversion."""
    # Validate files exist
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', dataset_name)
    processed_dir = os.path.join(TRAINING_RUNS_DIR, f"{clean_name}_processed")
    ckpt_path = os.path.join(processed_dir, checkpoint_rel_path)
    
    if not os.path.exists(ckpt_path):
        return False
        
    thread = threading.Thread(
        target=export_onnx_subprocess,
        args=(clean_name, checkpoint_rel_path, model_name)
    )
    thread.daemon = True
    thread.start()
    return True
