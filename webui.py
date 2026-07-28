from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if os.name == "nt":
    cache_root = Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm"))
    cache_paths = {
        "TRITON_HOME": cache_root / "triton-home",
        "TRITON_CACHE_DIR": cache_root / "triton-cache",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "inductor-cache",
        "HF_HOME": cache_root / "hf-cache",
        "TEMP": cache_root / "temp",
        "TMP": cache_root / "temp",
    }
    for key, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)

import numpy as np
import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
from piper_web import configure_piper_callbacks, piper_training, router as piper_router


def configure_torch_cpu_threads() -> int:
    override = os.environ.get("VOXCPM_CPU_THREADS", "").strip()
    if override:
        thread_count = max(1, int(override))
    else:
        try:
            import psutil

            thread_count = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except ImportError:
            thread_count = os.cpu_count() or 1
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits changing inter-op threads before parallel work.
        pass
    return thread_count


TORCH_CPU_THREADS = configure_torch_cpu_threads()


ROOT = Path(__file__).resolve().parent
WEB_PAGE = ROOT / "web.html"
LORA_PAGE = ROOT / "lora.html"
OUTPUT_ROOT = ROOT / "outputs" / "web"
LORA_ROOT = ROOT / "lora"
MODEL_PATHS = {
    "voxcpm2": ROOT / "pretrained_models" / "VoxCPM2",
    "voxcpm1.5": ROOT / "pretrained_models" / "VoxCPM1.5",
    "voxcpm-0.5b": ROOT / "pretrained_models" / "VoxCPM-0.5B",
}
DENOISER_PATH = ROOT / "pretrained_models" / "ZipEnhancer"
TRAINING_DATASET_ROOT = Path(r"D:\音频素材")
DEFAULT_TRAINING_DATASET = TRAINING_DATASET_ROOT / "爱弥斯语音训练集" / "train" / "wavs"
TRAINING_DATASET_REGISTRY = Path(os.environ.get("VOXCPM_CACHE_DIR", str(ROOT / ".cache"))) / "training-datasets.json"
TRAINING_DATASET_REGISTRY_LOCK = threading.RLock()
BATCH_OUTPUT_DIRECTORY_REGISTRY = Path(os.environ.get("VOXCPM_CACHE_DIR", str(ROOT / ".cache"))) / "batch-output-directories.json"
BATCH_OUTPUT_DIRECTORY_REGISTRY_LOCK = threading.RLock()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
VOXCPM2_MIN_GPU_MEMORY = 7 * 1024**3
HYBRID_MAX_GENERATION_LENGTH = 1024
HYBRID_DEVICES = {"hybrid", "hybrid-max"}
MAX_INFERENCE_SEGMENTS = 100
MAX_BATCH_TASKS = 100
LORA_2B_MIN_GPU_MEMORY = 8 * 1024**3


@dataclass(frozen=True)
class ModelConfig:
    model_key: str
    device: str
    optimize: bool
    load_denoiser: bool


class ModelRuntime:
    def __init__(self) -> None:
        self._model: VoxCPM | None = None
        self._config: ModelConfig | None = None
        self._identity: tuple[str, str, bool] | None = None
        self._lora_id: str | None = None
        self._lora_signature: str | None = None
        self._prompt_cache: dict | None = None
        self._prompt_cache_key: tuple | None = None
        self._lock = threading.Lock()

    @property
    def config(self) -> ModelConfig | None:
        return self._config

    def _release(self) -> None:
        self._model = None
        self._config = None
        self._identity = None
        self._lora_id = None
        self._lora_signature = None
        self._prompt_cache = None
        self._prompt_cache_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_prompt_cache(self) -> None:
        self._prompt_cache = None
        self._prompt_cache_key = None

    @staticmethod
    def _load_lora_weights(model: VoxCPM, checkpoint: dict) -> None:
        loaded_keys, skipped_keys = model.load_lora(checkpoint["path"])
        if not loaded_keys or skipped_keys:
            raise RuntimeError(
                f"Incompatible LoRA checkpoint: loaded {len(loaded_keys)} parameters, "
                f"skipped {len(skipped_keys)}"
            )

    def get(self, config: ModelConfig, lora_checkpoint: dict | None = None) -> VoxCPM:
        identity = (config.model_key, config.device, config.optimize)
        denoiser_device = "gpu" if config.device == "hybrid" else "cpu"
        if self._model is not None and self._identity == identity:
            requested_lora_id = lora_checkpoint["id"] if lora_checkpoint else None
            requested_signature = lora_checkpoint["signature"] if lora_checkpoint else None
            if lora_checkpoint is None and self._lora_signature is not None:
                if self._lora_id is not None:
                    self._model.set_lora_enabled(False)
                    self._lora_id = None
                    self._clear_prompt_cache()
            elif lora_checkpoint is not None and self._lora_signature == requested_signature:
                if self._lora_id != requested_lora_id:
                    if config.optimize:
                        # reduce-overhead compilation captures the initial LoRA state.
                        # Rebuild so the requested weights are loaded before warm-up.
                        self._release()
                    else:
                        try:
                            self._load_lora_weights(self._model, lora_checkpoint)
                        except Exception:
                            self._release()
                            raise
                        self._lora_id = requested_lora_id
                        self._clear_prompt_cache()
                if self._model is not None:
                    self._model.set_lora_enabled(True)
            elif lora_checkpoint is not None:
                self._release()

        if self._model is not None and self._identity == identity:
            if config.load_denoiser:
                self._model.ensure_denoiser(str(DENOISER_PATH), device=denoiser_device)
            self._config = config
            return self._model

        self._release()
        model_path = MODEL_PATHS[config.model_key]
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Model is not installed: {model_path}")

        if config.load_denoiser:
            if not DENOISER_PATH.is_dir():
                raise FileNotFoundError(f"Denoiser is not installed: {DENOISER_PATH}")

        try:
            lora_config = LoRAConfig(**lora_checkpoint["lora_config"]) if lora_checkpoint else None
            self._model = VoxCPM(
                voxcpm_model_path=str(model_path),
                zipenhancer_model_path=None,
                enable_denoiser=False,
                optimize=config.optimize,
                device=config.device,
                denoiser_device="cpu",
                lora_config=lora_config,
                lora_weights_path=lora_checkpoint["path"] if lora_checkpoint else None,
            )
            if lora_checkpoint:
                self._lora_id = lora_checkpoint["id"]
                self._lora_signature = lora_checkpoint["signature"]
            if config.load_denoiser:
                self._model.ensure_denoiser(str(DENOISER_PATH), device=denoiser_device)
        except Exception:
            self._release()
            raise
        self._config = config
        self._identity = identity
        return self._model

    @property
    def prompt_cache_ready(self) -> bool:
        return self._prompt_cache is not None

    @property
    def lora_id(self) -> str | None:
        return self._lora_id

    def generate_many(
        self,
        config: ModelConfig,
        texts: list[str],
        prompt_cache_key: tuple | None = None,
        prompt_build_kwargs: dict | None = None,
        lora_checkpoint: dict | None = None,
        **kwargs,
    ):
        with self._lock:
            try:
                model = self.get(config, lora_checkpoint)
                prompt_cache = None
                cache_hit = False
                if prompt_build_kwargs:
                    if (
                        config.device == "hybrid"
                        and self._prompt_cache is not None
                        and self._prompt_cache_key == prompt_cache_key
                    ):
                        prompt_cache = self._prompt_cache
                        cache_hit = True
                    else:
                        prompt_cache = model.build_prompt_cache(**prompt_build_kwargs)
                        if config.device == "hybrid":
                            self._prompt_cache = prompt_cache
                            self._prompt_cache_key = prompt_cache_key

                results = []
                for text in texts:
                    segment_started = time.perf_counter()
                    wav = model.generate(text=text, prompt_cache=prompt_cache, **kwargs)
                    successful_seed = getattr(model.tts_model, "last_successful_seed", kwargs.get("seed"))
                    results.append((wav, successful_seed, time.perf_counter() - segment_started))
                return results, model.tts_model.sample_rate, cache_hit
            except torch.OutOfMemoryError:
                self._release()
                raise

    def release(self) -> None:
        with self._lock:
            self._release()


runtime = ModelRuntime()


def read_lab_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode transcript: {path.name}")


def inspect_lora_dataset(directory: str) -> dict:
    dataset_dir = Path(directory).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    wav_paths = sorted(dataset_dir.glob("*.wav"))
    if not wav_paths:
        raise ValueError("Dataset directory contains no WAV files")

    records = []
    missing_labs = []
    empty_labs = []
    sample_rates: dict[int, int] = {}
    channels: dict[int, int] = {}
    total_duration = 0.0
    for wav_path in wav_paths:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.is_file():
            missing_labs.append(wav_path.name)
            continue
        text = read_lab_text(lab_path)
        if not text:
            empty_labs.append(lab_path.name)
            continue
        info = sf.info(wav_path)
        duration = float(info.duration)
        total_duration += duration
        sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
        channels[info.channels] = channels.get(info.channels, 0) + 1
        records.append(
            {
                "audio": str(wav_path),
                "text": text,
                "duration": round(duration, 6),
            }
        )

    if missing_labs:
        raise ValueError(f"{len(missing_labs)} WAV files have no matching .lab transcript")
    if empty_labs:
        raise ValueError(f"{len(empty_labs)} transcript files are empty")
    if not records:
        raise ValueError("Dataset has no usable audio/transcript pairs")

    durations = sorted(record["duration"] for record in records)
    return {
        "directory": str(dataset_dir),
        "records": records,
        "file_count": len(records),
        "total_minutes": round(total_duration / 60, 3),
        "min_duration": round(durations[0], 3),
        "median_duration": round(durations[len(durations) // 2], 3),
        "max_duration": round(durations[-1], 3),
        "sample_rates": sample_rates,
        "channels": channels,
    }


def safe_lora_job_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip()).strip(". ")
    if not value:
        value = datetime.now().strftime("lora-%y%m%d-%H%M%S")
    return value[:80]


def write_lora_manifest(dataset: dict, destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="\n") as manifest:
        for record in dataset["records"]:
            manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def model_training_sample_rate(model_path: Path) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    return int(config.get("audio_vae_config", {}).get("sample_rate", 16000))


def calculate_lora_training_schedule(
    sample_count: int,
    batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
    save_every_epochs: int,
) -> dict:
    effective_batch_size = batch_size * grad_accum_steps
    optimizer_steps = math.ceil(sample_count * num_epochs / effective_batch_size)
    save_interval_steps = min(
        optimizer_steps,
        math.ceil(sample_count * save_every_epochs / effective_batch_size),
    )
    return {
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch_size,
        "optimizer_steps": max(1, optimizer_steps),
        "save_every_epochs": save_every_epochs,
        "save_interval_steps": max(1, save_interval_steps),
    }


class LoRATrainingRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=2500)
        self._status = "idle"
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._job_name: str | None = None
        self._model_key: str | None = None
        self._returncode: int | None = None

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line.rstrip("\r\n"))

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, config_path: Path, job_name: str, model_key: str) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A LoRA training job is already running")
            pause_request_path = LORA_ROOT / job_name / "checkpoints" / ".pause-request"
            pause_request_path.unlink(missing_ok=True)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            python_path = ROOT / ".venv" / "Scripts" / "python.exe"
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                [str(python_path), "scripts/train_voxcpm_finetune.py", "--config_path", str(config_path)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._logs.clear()
            self._status = "running"
            self._started_at = time.time()
            self._finished_at = None
            self._job_name = job_name
            self._model_key = model_key
            self._returncode = None
            process = self._process

        threading.Thread(target=self._read_process, args=(process,), daemon=True).start()

    def _read_process(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(line)
        returncode = process.wait()
        with self._lock:
            self._returncode = returncode
            self._finished_at = time.time()
            if self._status == "pausing" and returncode == 0:
                pause_request_path = LORA_ROOT / (self._job_name or "") / "checkpoints" / ".pause-request"
                pause_acknowledged = not pause_request_path.exists()
                pause_request_path.unlink(missing_ok=True)
                self._status = "paused" if pause_acknowledged else "completed"
            else:
                self._status = "completed" if returncode == 0 else "failed"

    def pause(self) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return False
            if self._status == "pausing":
                return True
            if not self._job_name:
                return False
            pause_request_path = LORA_ROOT / self._job_name / "checkpoints" / ".pause-request"
            pause_request_path.parent.mkdir(parents=True, exist_ok=True)
            pause_request_path.write_text(str(time.time()), encoding="ascii")
            self._status = "pausing"
            self._logs.append("Pause requested; waiting for the current batch to finish.")
            return True

    def stop(self) -> bool:
        """Compatibility alias for clients that still use the old endpoint."""
        return self.pause()

    def snapshot(self) -> dict:
        with self._lock:
            now = self._finished_at or time.time()
            elapsed = now - self._started_at if self._started_at else 0.0
            return {
                "status": self._status,
                "running": self._process is not None and self._process.poll() is None,
                "pause_supported": True,
                "job_name": self._job_name,
                "model_key": self._model_key,
                "returncode": self._returncode,
                "elapsed_seconds": round(elapsed, 1),
                "logs": "\n".join(self._logs),
            }


lora_training = LoRATrainingRuntime()
app = FastAPI(title="VoxCPM Local Inference", docs_url="/api/docs")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
configure_piper_callbacks(runtime.release, lambda: lora_training.running)
app.include_router(piper_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_PAGE)


@app.get("/lora")
def lora_index() -> FileResponse:
    return FileResponse(LORA_PAGE)


@app.get("/api/status")
def status() -> dict:
    loaded = runtime.config
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available()
        else None,
        "voxcpm2_gpu_supported": torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory >= VOXCPM2_MIN_GPU_MEMORY,
        "models": {key: path.is_dir() for key, path in MODEL_PATHS.items()},
        "denoiser": DENOISER_PATH.is_dir(),
        "loaded": loaded.__dict__ if loaded else None,
        "input_cache_ready": runtime.prompt_cache_ready,
        "active_lora_id": runtime.lora_id,
        "lora_checkpoints": list_lora_checkpoints(),
        "lora_training": lora_training.snapshot(),
    }


def infer_lora_model_key(base_model: str) -> str | None:
    normalized = Path(base_model.replace("\\", "/")).name.lower()
    if normalized == "voxcpm-0.5b":
        return "voxcpm-0.5b"
    if normalized == "voxcpm1.5":
        return "voxcpm1.5"
    if normalized == "voxcpm2":
        return "voxcpm2"
    return None


def list_lora_checkpoints() -> list[dict]:
    checkpoints = []
    if not LORA_ROOT.is_dir():
        return checkpoints
    for folder in LORA_ROOT.glob("*/checkpoints/step_*"):
        if not folder.is_dir():
            continue
        weights_path = next(
            (path for path in (folder / "lora_weights.safetensors", folder / "lora_weights.ckpt") if path.is_file()),
            None,
        )
        config_path = folder / "lora_config.json"
        if weights_path is None or not config_path.is_file():
            continue
        try:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            lora_config = saved_config["lora_config"]
            model_key = infer_lora_model_key(saved_config["base_model"])
            LoRAConfig(**lora_config)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if model_key is None:
            continue
        state_path = folder / "training_state.json"
        step = None
        if state_path.is_file():
            try:
                step = int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
        job_name = folder.parents[1].name
        checkpoint_id = folder.relative_to(LORA_ROOT).as_posix()
        checkpoints.append(
            {
                "id": checkpoint_id,
                "job_name": job_name,
                "name": folder.name,
                "display_name": f"{job_name} · {folder.name}",
                "step": step,
                "path": str(folder),
                "model_key": model_key,
                "rank": lora_config["r"],
                "alpha": lora_config["alpha"],
                "lora_config": lora_config,
                "signature": json.dumps(lora_config, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                "modified_at": datetime.fromtimestamp(weights_path.stat().st_mtime).astimezone().isoformat(
                    timespec="seconds"
                ),
            }
        )
    return sorted(checkpoints, key=lambda item: item["modified_at"], reverse=True)


def resolve_lora_checkpoint(lora_id: str, model_key: str) -> dict | None:
    selected_id = lora_id.strip()
    if not selected_id:
        return None
    checkpoint = next((item for item in list_lora_checkpoints() if item["id"] == selected_id), None)
    if checkpoint is None:
        raise HTTPException(status_code=400, detail="LoRA checkpoint does not exist")
    if checkpoint["model_key"] != model_key:
        raise HTTPException(
            status_code=400,
            detail=f"LoRA checkpoint requires {checkpoint['model_key']}, but {model_key} is selected",
        )
    return checkpoint


@app.get("/api/lora/checkpoints")
def lora_checkpoints() -> dict:
    return {"checkpoints": list_lora_checkpoints(), "active_lora_id": runtime.lora_id}


@app.get("/api/lora/status")
def lora_status() -> dict:
    gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    return {
        **lora_training.snapshot(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(gpu_memory / 1024**3, 1),
        "models": {
            key: {
                "installed": path.is_dir(),
                "path": str(path),
                "supported_on_current_gpu": torch.cuda.is_available()
                and (key != "voxcpm2" or gpu_memory >= LORA_2B_MIN_GPU_MEMORY),
            }
            for key, path in MODEL_PATHS.items()
        },
        "default_dataset": str(DEFAULT_TRAINING_DATASET),
        "checkpoints": list_lora_checkpoints(),
    }


@app.post("/api/lora/dataset")
async def lora_dataset(dataset_dir: str = Form(...)) -> dict:
    try:
        dataset = await asyncio.to_thread(inspect_lora_dataset, dataset_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {key: value for key, value in dataset.items() if key != "records"}


@app.post("/api/lora/train")
async def start_lora_training(
    dataset_dir: str = Form(...),
    model_key: str = Form("voxcpm-0.5b"),
    output_name: str = Form(""),
    learning_rate: float = Form(0.0001),
    num_epochs: int = Form(20),
    batch_size: int = Form(1),
    grad_accum_steps: int = Form(16),
    lora_rank: int = Form(4),
    lora_alpha: int = Form(8),
    save_every_epochs: int = Form(5),
    enable_lm: bool = Form(False),
    enable_dit: bool = Form(True),
    enable_proj: bool = Form(False),
    low_vram: bool = Form(True),
) -> dict:
    if lora_training.running:
        raise HTTPException(status_code=409, detail="A LoRA training job is already running")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="LoRA training requires CUDA")
    if model_key not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown model")
    model_path = MODEL_PATHS[model_key]
    if not (model_path / "config.json").is_file():
        raise HTTPException(status_code=400, detail=f"Model is not installed: {model_path}")
    if model_key == "voxcpm2" and torch.cuda.get_device_properties(0).total_memory < LORA_2B_MIN_GPU_MEMORY:
        raise HTTPException(status_code=507, detail="VoxCPM2 LoRA training requires more than 4 GB VRAM")
    if not enable_lm and not enable_dit and not enable_proj:
        raise HTTPException(status_code=400, detail="At least one LoRA target must be enabled")
    if not 1 <= batch_size <= 32 or not 1 <= grad_accum_steps <= 256:
        raise HTTPException(status_code=400, detail="Batch or gradient accumulation value is invalid")
    if not 1 <= lora_rank <= 256 or not 1 <= lora_alpha <= 1024:
        raise HTTPException(status_code=400, detail="LoRA rank or alpha is invalid")
    if not 1 <= num_epochs <= 1000 or not 1 <= save_every_epochs <= num_epochs:
        raise HTTPException(status_code=400, detail="Training epochs or checkpoint interval is invalid")
    if not 1e-7 <= learning_rate <= 0.1:
        raise HTTPException(status_code=400, detail="Learning rate is invalid")

    try:
        dataset = await asyncio.to_thread(inspect_lora_dataset, dataset_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if batch_size > dataset["file_count"]:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed the number of training samples")
    schedule = calculate_lora_training_schedule(
        dataset["file_count"],
        batch_size,
        grad_accum_steps,
        num_epochs,
        save_every_epochs,
    )
    num_iters = schedule["optimizer_steps"]
    save_interval = schedule["save_interval_steps"]
    if num_iters > 100000:
        raise HTTPException(status_code=400, detail="Calculated optimizer steps exceed 100000")

    job_name = safe_lora_job_name(output_name)
    job_dir = LORA_ROOT / job_name
    checkpoints_dir = job_dir / "checkpoints"
    logs_dir = job_dir / "logs"
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "train.jsonl"
    config_path = job_dir / "train_config.yaml"
    write_lora_manifest(dataset, manifest_path)

    config = {
        "pretrained_path": str(model_path),
        "train_manifest": str(manifest_path),
        "val_manifest": None,
        "sample_rate": model_training_sample_rate(model_path),
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_workers": 1,
        "num_iters": num_iters,
        "log_interval": 1,
        "valid_interval": num_iters,
        "save_interval": save_interval,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": min(30, max(1, num_iters // 10)),
        "max_steps": num_iters,
        "max_batch_tokens": 0,
        "max_grad_norm": 1.0,
        "save_path": str(checkpoints_dir),
        "tensorboard": str(logs_dir),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "low_vram": low_vram,
        "audio_encoder_device": "cpu" if low_vram else "cuda",
        "lora": {
            "enable_lm": enable_lm,
            "enable_dit": enable_dit,
            "enable_proj": enable_proj,
            "r": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.0,
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    await asyncio.to_thread(runtime.release)
    try:
        lora_training.start(config_path, job_name, model_key)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {exc}") from exc
    return {
        "status": "running",
        "job_name": job_name,
        "job_dir": str(job_dir),
        "model_key": model_key,
        "dataset": {key: value for key, value in dataset.items() if key != "records"},
        "schedule": schedule,
    }


@app.post("/api/lora/pause")
def pause_lora_training() -> dict:
    if not lora_training.pause():
        raise HTTPException(status_code=409, detail="No LoRA training job is running")
    return {"status": "pausing"}


@app.post("/api/lora/stop")
def stop_lora_training() -> dict:
    """Backward-compatible cooperative pause; this endpoint never kills training."""
    return pause_lora_training()


@app.get("/api/audio/{filename}")
def audio(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = OUTPUT_ROOT / safe_name
    if safe_name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=safe_name)


@app.get("/api/training-datasets")
def training_datasets() -> dict:
    datasets = list_training_datasets()
    return {
        "datasets": datasets,
        "default": next((item["path"] for item in datasets if item["default"]), datasets[0]["path"] if datasets else None),
    }


@app.post("/api/training-datasets")
async def create_training_dataset_endpoint(name: str = Form(...)) -> dict:
    try:
        return await asyncio.to_thread(create_training_dataset, name)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="同名训练集已经存在") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/training-datasets/browse")
async def browse_training_dataset(initial_path: str = Form("")) -> dict:
    try:
        selected = await asyncio.to_thread(choose_training_dataset_directory, initial_path)
        if selected is None:
            return {"cancelled": True}
        dataset = await asyncio.to_thread(register_training_dataset, selected)
        return {"cancelled": False, "path": dataset["path"], "dataset": dataset}
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/batch-output-directories")
def batch_output_directories() -> dict:
    directories = [str(path) for path in _read_registered_batch_output_directories()]
    return {"directories": directories, "default": directories[-1] if directories else None}


@app.post("/api/batch-output-directories/browse")
async def browse_batch_output_directory(initial_path: str = Form("")) -> dict:
    try:
        selected = await asyncio.to_thread(choose_batch_output_directory, initial_path)
        if selected is None:
            return {"cancelled": True}
        path = await asyncio.to_thread(register_batch_output_directory, selected)
        return {"cancelled": False, "path": str(path)}
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/audio/postprocess")
async def postprocess_generated_audio(
    filename: str = Form(...),
    preset: str = Form("clean"),
    highpass_enabled: bool = Form(True),
    highpass_hz: float = Form(70.0),
    mud_enabled: bool = Form(True),
    mud_gain_db: float = Form(-1.5),
    presence_enabled: bool = Form(True),
    presence_gain_db: float = Form(1.0),
    air_enabled: bool = Form(True),
    air_gain_db: float = Form(0.5),
    compressor_enabled: bool = Form(True),
    compressor_threshold_db: float = Form(-18.0),
    compressor_ratio: float = Form(2.0),
    loudness_enabled: bool = Form(True),
    target_lufs: float = Form(-16.0),
    limiter_enabled: bool = Form(True),
    limiter_db: float = Form(-1.0),
    reverb_enabled: bool = Form(False),
    reverb_wet: float = Form(0.0),
    pitch_enabled: bool = Form(False),
    pitch_semitones: float = Form(0.0),
) -> dict:
    settings = {
        "preset": preset.strip()[:40] or "custom",
        "highpass_enabled": highpass_enabled,
        "highpass_hz": highpass_hz,
        "mud_enabled": mud_enabled,
        "mud_gain_db": mud_gain_db,
        "presence_enabled": presence_enabled,
        "presence_gain_db": presence_gain_db,
        "air_enabled": air_enabled,
        "air_gain_db": air_gain_db,
        "compressor_enabled": compressor_enabled,
        "compressor_threshold_db": compressor_threshold_db,
        "compressor_ratio": compressor_ratio,
        "loudness_enabled": loudness_enabled,
        "target_lufs": target_lufs,
        "limiter_enabled": limiter_enabled,
        "limiter_db": limiter_db,
        "reverb_enabled": reverb_enabled,
        "reverb_wet": reverb_wet,
        "pitch_enabled": pitch_enabled,
        "pitch_semitones": pitch_semitones,
    }
    try:
        source_path = _validated_output_audio(filename)
        return await asyncio.to_thread(postprocess_audio, source_path, settings)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/training-datasets/move")
async def move_generated_audio(filename: str = Form(...), dataset_dir: str = Form(...)) -> dict:
    try:
        return await asyncio.to_thread(move_audio_to_training_dataset, filename, dataset_dir)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="本条已经存在于训练集") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def save_upload(upload: UploadFile, directory: Path) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported reference audio format")

    destination = directory / f"reference{suffix}"
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Reference audio exceeds 50 MB")
            output.write(chunk)
    await upload.close()
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_content_sha256(path: Path) -> str:
    audio_data, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    digest = hashlib.sha256()
    digest.update(struct.pack("<III", sample_rate, audio_data.shape[0], audio_data.shape[1]))
    digest.update(audio_data.tobytes(order="C"))
    return digest.hexdigest()


def _riff_info_field(tag: bytes, value: str) -> bytes:
    data = value.encode("utf-8") + b"\0"
    padding = b"\0" if len(data) % 2 else b""
    return tag + struct.pack("<I", len(data)) + data + padding


def write_wav_metadata(path: Path, metadata: dict) -> None:
    comment = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))
    info = b"INFO" + b"".join(
        (
            _riff_info_field(b"INAM", path.stem),
            _riff_info_field(b"ISFT", "VoxCPM WebUI"),
            _riff_info_field(b"ICRD", metadata["created_at"]),
            _riff_info_field(b"ICMT", comment),
        )
    )
    chunk = b"LIST" + struct.pack("<I", len(info)) + info
    if len(info) % 2:
        chunk += b"\0"

    with path.open("r+b") as wav_file:
        header = wav_file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")
        wav_file.seek(0, os.SEEK_END)
        wav_file.write(chunk)
        riff_size = wav_file.tell() - 8
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", riff_size))


def read_wav_metadata(path: Path) -> dict:
    with path.open("rb") as wav_file:
        header = wav_file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")
        while chunk_header := wav_file.read(8):
            if len(chunk_header) != 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_data = wav_file.read(chunk_size)
            if chunk_size % 2:
                wav_file.read(1)
            if chunk_id != b"LIST" or not chunk_data.startswith(b"INFO"):
                continue
            offset = 4
            while offset + 8 <= len(chunk_data):
                field_id, field_size = struct.unpack("<4sI", chunk_data[offset : offset + 8])
                offset += 8
                field_data = chunk_data[offset : offset + field_size]
                offset += field_size + (field_size % 2)
                if field_id == b"ICMT":
                    try:
                        return json.loads(field_data.rstrip(b"\0").decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid VoxCPM metadata in {path.name}") from exc
    raise ValueError(f"VoxCPM generation metadata not found: {path.name}")


def create_training_dataset(name: str) -> dict:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("请输入训练集名称")
    if len(normalized_name) > 80:
        raise ValueError("训练集名称不能超过 80 个字符")
    if normalized_name in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', normalized_name):
        raise ValueError("训练集名称包含 Windows 不允许的字符")
    if normalized_name.endswith("."):
        raise ValueError("训练集名称不能以句点结尾")
    reserved_names = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
    if normalized_name.split(".", 1)[0].upper() in reserved_names:
        raise ValueError("训练集名称是 Windows 保留名称")

    root = TRAINING_DATASET_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = root / normalized_name
    if dataset_root.parent != root:
        raise ValueError("训练集名称无效")
    dataset_root.mkdir(exist_ok=False)
    wav_directory = dataset_root / "train" / "wavs"
    wav_directory.mkdir(parents=True)
    return {
        "name": normalized_name,
        "path": str(wav_directory.resolve()),
        "file_count": 0,
        "default": False,
    }


def resolve_training_dataset_directory(directory: str | Path) -> Path:
    selected = Path(directory).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"训练集目录不存在: {selected}")
    if any(selected.glob("*.wav")):
        return selected
    for relative_path in (Path("train") / "wavs", Path("wavs")):
        candidate = selected / relative_path
        if candidate.is_dir():
            return candidate.resolve()
    return selected


def _read_registered_training_datasets() -> list[Path]:
    with TRAINING_DATASET_REGISTRY_LOCK:
        if not TRAINING_DATASET_REGISTRY.is_file():
            return []
        try:
            values = json.loads(TRAINING_DATASET_REGISTRY.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
    if not isinstance(values, list):
        return []
    paths = []
    for value in values:
        try:
            path = Path(str(value)).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if path.is_dir():
            paths.append(path)
    return paths


def register_training_dataset(directory: str | Path) -> dict:
    dataset_dir = resolve_training_dataset_directory(directory)
    with TRAINING_DATASET_REGISTRY_LOCK:
        registered = _read_registered_training_datasets()
        unique_paths = {str(path).lower(): path for path in registered}
        unique_paths[str(dataset_dir).lower()] = dataset_dir
        TRAINING_DATASET_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = TRAINING_DATASET_REGISTRY.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps([str(path) for path in unique_paths.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(TRAINING_DATASET_REGISTRY)
    dataset_name = dataset_dir.parents[1].name if dataset_dir.name.lower() == "wavs" and len(dataset_dir.parents) > 1 else dataset_dir.name
    return {
        "name": dataset_name,
        "path": str(dataset_dir),
        "file_count": sum(1 for _ in dataset_dir.glob("*.wav")),
        "default": dataset_dir == DEFAULT_TRAINING_DATASET.resolve(),
    }


def _choose_windows_directory(initial_path: str, default_path: Path, description: str) -> str | None:
    if os.name != "nt":
        raise RuntimeError("文件夹选择器当前仅支持 Windows；请在 Windows 本地 WebUI 中使用")
    initial = Path(initial_path).expanduser() if initial_path.strip() else default_path
    if not initial.is_dir():
        initial = default_path if default_path.is_dir() else ROOT
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $env:VOXCPM_FOLDER_DIALOG_DESCRIPTION
$dialog.ShowNewFolderButton = $true
if ($env:VOXCPM_INITIAL_FOLDER) { $dialog.SelectedPath = $env:VOXCPM_INITIAL_FOLDER }
try {
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write($dialog.SelectedPath)
    }
}
finally { $dialog.Dispose() }
"""
    env = os.environ.copy()
    env["VOXCPM_INITIAL_FOLDER"] = str(initial.resolve())
    env["VOXCPM_FOLDER_DIALOG_DESCRIPTION"] = description
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "Windows 文件夹选择器启动失败"
        raise RuntimeError(detail)
    selected = completed.stdout.strip()
    return selected or None


def choose_training_dataset_directory(initial_path: str = "") -> str | None:
    default_path = DEFAULT_TRAINING_DATASET if DEFAULT_TRAINING_DATASET.is_dir() else TRAINING_DATASET_ROOT
    return _choose_windows_directory(initial_path, default_path, "选择包含 WAV 和同名 LAB 的训练集文件夹")


def _read_registered_batch_output_directories() -> list[Path]:
    with BATCH_OUTPUT_DIRECTORY_REGISTRY_LOCK:
        if not BATCH_OUTPUT_DIRECTORY_REGISTRY.is_file():
            return []
        try:
            values = json.loads(BATCH_OUTPUT_DIRECTORY_REGISTRY.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
    if not isinstance(values, list):
        return []
    directories = []
    for value in values:
        try:
            directory = Path(str(value)).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if directory.is_dir():
            directories.append(directory)
    return directories


def register_batch_output_directory(directory: str | Path) -> Path:
    selected = Path(directory).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"批量保存目录不存在: {selected}")
    with BATCH_OUTPUT_DIRECTORY_REGISTRY_LOCK:
        registered = _read_registered_batch_output_directories()
        unique = {str(path).lower(): path for path in registered}
        unique.pop(str(selected).lower(), None)
        unique[str(selected).lower()] = selected
        BATCH_OUTPUT_DIRECTORY_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = BATCH_OUTPUT_DIRECTORY_REGISTRY.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps([str(path) for path in unique.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(BATCH_OUTPUT_DIRECTORY_REGISTRY)
    return selected


def choose_batch_output_directory(initial_path: str = "") -> str | None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return _choose_windows_directory(initial_path, OUTPUT_ROOT, "选择批量音频与可选 LAB 训练对的保存文件夹")


def resolve_registered_batch_output_directory(directory: str) -> Path:
    requested = Path(directory).expanduser().resolve()
    registered = {str(path).lower(): path for path in _read_registered_batch_output_directories()}
    try:
        return registered[str(requested).lower()]
    except KeyError as exc:
        raise ValueError("批量保存目录未通过文件夹选择器登记") from exc


def list_training_datasets() -> list[dict]:
    candidates = {DEFAULT_TRAINING_DATASET.resolve()}
    candidates.update(_read_registered_training_datasets())
    audio_root = TRAINING_DATASET_ROOT
    if audio_root.is_dir():
        candidates.update(path.resolve() for path in audio_root.glob("*/train/wavs") if path.is_dir())
    for manifest in LORA_ROOT.glob("*/train.jsonl"):
        try:
            first_line = next(line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
            audio_path = Path(json.loads(first_line)["audio"]).resolve()
            if audio_path.parent.is_dir():
                candidates.add(audio_path.parent)
        except (StopIteration, KeyError, OSError, json.JSONDecodeError):
            continue

    datasets = []
    for path in sorted(candidates, key=lambda item: str(item).lower()):
        if not path.is_dir():
            continue
        dataset_name = path.parents[1].name if path.name.lower() == "wavs" and len(path.parents) > 1 else path.name
        datasets.append(
            {
                "name": dataset_name,
                "path": str(path),
                "file_count": sum(1 for _ in path.glob("*.wav")),
                "default": path == DEFAULT_TRAINING_DATASET.resolve(),
            }
        )
    return datasets


def _validated_output_audio(filename: str) -> Path:
    safe_name = Path(filename).name
    path = OUTPUT_ROOT / safe_name
    if safe_name != filename or path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError("Audio file not found")
    return path


def _validate_postprocess_settings(settings: dict) -> None:
    bounds = {
        "highpass_hz": (0.0, 240.0),
        "mud_gain_db": (-12.0, 6.0),
        "presence_gain_db": (-12.0, 12.0),
        "air_gain_db": (-12.0, 12.0),
        "compressor_threshold_db": (-48.0, 0.0),
        "compressor_ratio": (1.0, 12.0),
        "target_lufs": (-30.0, -8.0),
        "limiter_db": (-8.0, -0.1),
        "reverb_wet": (0.0, 0.5),
        "pitch_semitones": (-12.0, 12.0),
    }
    for key, (minimum, maximum) in bounds.items():
        value = settings[key]
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Post-process parameter {key} is out of range")


def postprocess_audio(source_path: Path, settings: dict) -> dict:
    from pedalboard import (
        Compressor,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        PeakFilter,
        Pedalboard,
        PitchShift,
        Reverb,
    )
    import pyloudnorm as pyln

    _validate_postprocess_settings(settings)
    settings_json = json.dumps(settings, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    settings_hash = hashlib.sha256(settings_json.encode("ascii")).hexdigest()[:5]
    output_path = source_path.with_name(f"{source_path.stem}-af-{settings_hash}.wav")
    if output_path.is_file():
        metadata = read_wav_metadata(output_path)
        info = sf.info(output_path)
        return {
            "audio_url": f"/api/audio/{output_path.name}",
            "filename": output_path.name,
            "text": metadata.get("text", ""),
            "duration": round(float(info.duration), 3),
            "postprocess_hash": settings_hash,
            "cached": True,
            "model": metadata.get("model_key"),
            "lora_name": metadata.get("lora_name"),
            "device": metadata.get("device"),
        }

    audio_data, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    channels_first = np.ascontiguousarray(audio_data.T)
    channels_first -= channels_first.mean(axis=1, keepdims=True)
    fade_samples = min(int(sample_rate * 0.008), channels_first.shape[1] // 2)
    if fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        channels_first[:, :fade_samples] *= fade
        channels_first[:, -fade_samples:] *= fade[::-1]

    effects = []
    if settings.get("highpass_enabled", True) and settings["highpass_hz"] > 0:
        effects.append(HighpassFilter(cutoff_frequency_hz=settings["highpass_hz"]))
    if settings.get("mud_enabled", True) and settings["mud_gain_db"]:
        effects.append(PeakFilter(cutoff_frequency_hz=250.0, gain_db=settings["mud_gain_db"], q=0.8))
    if settings.get("presence_enabled", True) and settings["presence_gain_db"]:
        effects.append(PeakFilter(cutoff_frequency_hz=3000.0, gain_db=settings["presence_gain_db"], q=0.8))
    if settings.get("air_enabled", True) and settings["air_gain_db"]:
        effects.append(HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=settings["air_gain_db"], q=0.7))
    if settings.get("pitch_enabled", True) and settings["pitch_semitones"]:
        effects.append(PitchShift(semitones=settings["pitch_semitones"]))
    if settings.get("compressor_enabled", True) and settings["compressor_ratio"] > 1.0:
        effects.append(
            Compressor(
                threshold_db=settings["compressor_threshold_db"],
                ratio=settings["compressor_ratio"],
                attack_ms=10.0,
                release_ms=100.0,
            )
        )
    if settings.get("reverb_enabled", True) and settings["reverb_wet"] > 0:
        effects.append(
            Reverb(
                room_size=0.35,
                damping=0.6,
                wet_level=settings["reverb_wet"],
                dry_level=1.0 - settings["reverb_wet"],
                width=1.0,
            )
        )
    processed = Pedalboard(effects)(channels_first, sample_rate) if effects else channels_first

    gain_db = 0.0
    final_effects = []
    if settings.get("loudness_enabled", True):
        mono_for_meter = processed.mean(axis=0)
        duration = mono_for_meter.shape[0] / sample_rate
        if duration >= 0.4:
            current_loudness = pyln.Meter(sample_rate).integrated_loudness(mono_for_meter)
        else:
            rms = float(np.sqrt(np.mean(np.square(mono_for_meter), dtype=np.float64)))
            current_loudness = 20.0 * math.log10(max(rms, 1e-9))
        gain_db = 0.0 if not math.isfinite(current_loudness) else settings["target_lufs"] - current_loudness
        gain_db = max(-24.0, min(24.0, gain_db))
        final_effects.append(Gain(gain_db=gain_db))
    if settings.get("limiter_enabled", True):
        final_effects.append(Limiter(threshold_db=settings["limiter_db"], release_ms=100.0))
    if final_effects:
        processed = Pedalboard(final_effects)(processed, sample_rate)
    processed = np.clip(processed, -1.0, 1.0).T

    original_metadata = read_wav_metadata(source_path)
    processed_at = datetime.now().astimezone()
    metadata = {
        **original_metadata,
        "schema": "voxcpm-generation-v1",
        "created_at": processed_at.isoformat(timespec="seconds"),
        "filename": output_path.name,
        "source_filename": source_path.name,
        "postprocess_hash": settings_hash,
        "postprocess": settings,
        "postprocess_gain_db": round(gain_db, 4),
    }
    sf.write(output_path, processed, sample_rate, subtype="PCM_16")
    write_wav_metadata(output_path, metadata)
    return {
        "audio_url": f"/api/audio/{output_path.name}",
        "filename": output_path.name,
        "text": metadata.get("text", ""),
        "duration": round(processed.shape[0] / sample_rate, 3),
        "postprocess_hash": settings_hash,
        "cached": False,
        "model": metadata.get("model_key"),
        "lora_name": metadata.get("lora_name"),
        "device": metadata.get("device"),
    }


def move_audio_to_training_dataset(filename: str, dataset_dir: str) -> dict:
    source_path = _validated_output_audio(filename)
    allowed_datasets = {str(Path(item["path"]).resolve()).lower(): Path(item["path"]).resolve() for item in list_training_datasets()}
    requested_key = str(Path(dataset_dir).expanduser().resolve()).lower()
    if requested_key not in allowed_datasets:
        raise ValueError("Training dataset is not available")
    destination_dir = allowed_datasets[requested_key]
    source_hash = audio_content_sha256(source_path)
    for existing_path in destination_dir.glob("*.wav"):
        if audio_content_sha256(existing_path) == source_hash:
            raise FileExistsError("本条已经存在于训练集")

    metadata = read_wav_metadata(source_path)
    transcript = str(metadata.get("text", "")).strip()
    if not transcript:
        raise ValueError("Audio metadata does not contain training text")

    destination_path = destination_dir / source_path.name
    counter = 2
    while destination_path.exists() or destination_path.with_suffix(".lab").exists():
        destination_path = destination_dir / f"{source_path.stem}-{counter:02d}.wav"
        counter += 1
    lab_path = destination_path.with_suffix(".lab")
    temporary_lab = destination_dir / f".{lab_path.name}.tmp"
    temporary_lab.write_text(transcript, encoding="utf-8")
    try:
        shutil.move(str(source_path), str(destination_path))
        temporary_lab.replace(lab_path)
    except Exception:
        temporary_lab.unlink(missing_ok=True)
        if destination_path.is_file() and not source_path.exists():
            shutil.move(str(destination_path), str(source_path))
        raise
    return {
        "filename": destination_path.name,
        "lab_filename": lab_path.name,
        "dataset": str(destination_dir),
        "text": transcript,
    }


def timestamped_output_path(created_at: datetime) -> Path:
    stem = created_at.strftime("voxcpm-%y%m%d-%H%M%S")
    candidate = OUTPUT_ROOT / f"{stem}.wav"
    counter = 2
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"{stem}-{counter:02d}.wav"
        counter += 1
    return candidate


def batch_export_destination(source_path: Path, output_directory: Path, create_training_pair: bool) -> Path:
    if source_path.parent.resolve() == output_directory.resolve():
        return source_path
    candidate = output_directory / source_path.name
    counter = 2
    while candidate.exists() or (create_training_pair and candidate.with_suffix(".lab").exists()):
        candidate = output_directory / f"{source_path.stem}-{counter:02d}.wav"
        counter += 1
    return candidate


def export_batch_audio(
    source_path: Path,
    destination_path: Path,
    transcript: str,
    create_training_pair: bool,
) -> dict:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        temporary_audio = destination_path.with_name(f".{destination_path.name}.tmp")
        try:
            shutil.copy2(source_path, temporary_audio)
            temporary_audio.replace(destination_path)
        except Exception:
            temporary_audio.unlink(missing_ok=True)
            raise

    lab_path = None
    if create_training_pair:
        lab_path = destination_path.with_suffix(".lab")
        temporary_lab = lab_path.with_name(f".{lab_path.name}.tmp")
        try:
            temporary_lab.write_text(transcript.strip(), encoding="utf-8")
            temporary_lab.replace(lab_path)
        except Exception:
            temporary_lab.unlink(missing_ok=True)
            raise
    return {
        "exported_path": str(destination_path),
        "lab_path": str(lab_path) if lab_path else None,
    }


def split_sentence_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentence_endings = set("。！？!?；;…")
    closing_marks = set("”’\"')）】》」』]}〉")
    parts = []
    start = 0
    index = 0
    while index < len(normalized):
        character = normalized[index]
        boundary = character in sentence_endings
        if character == ".":
            previous_is_digit = index > 0 and normalized[index - 1].isdigit()
            next_is_digit = index + 1 < len(normalized) and normalized[index + 1].isdigit()
            lookahead = index + 1
            while lookahead < len(normalized) and normalized[lookahead] in closing_marks:
                lookahead += 1
            next_character = normalized[lookahead] if lookahead < len(normalized) else ""
            boundary = not (previous_is_digit and next_is_digit) and (
                not next_character or next_character.isspace() or not next_character.isascii()
            )
        if not boundary:
            index += 1
            continue

        end = index + 1
        while end < len(normalized) and normalized[end] in sentence_endings:
            end += 1
        while end < len(normalized) and normalized[end] in closing_marks:
            end += 1
        part = normalized[start:end].strip()
        if part:
            parts.append(part)
        start = end
        index = end

    remainder = normalized[start:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def build_text_tasks(text: str, batch_mode: str) -> list[dict]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    task_sources = [normalized] if batch_mode == "ordinary" else normalized.split("\n")
    tasks = []
    for source in task_sources:
        task_text = re.sub(r"\s+", " ", source).strip()
        segments = split_sentence_text(source)
        if segments:
            tasks.append({"text": task_text, "segments": segments})
    return tasks


def merge_text_task_results(tasks: list[dict], generation_results: list[tuple]) -> list[dict]:
    expected_count = sum(len(task["segments"]) for task in tasks)
    if len(generation_results) != expected_count:
        raise ValueError(f"Expected {expected_count} generation results, received {len(generation_results)}")

    merged_results = []
    result_index = 0
    for task in tasks:
        task_results = generation_results[result_index : result_index + len(task["segments"])]
        result_index += len(task_results)
        waveforms = [np.asarray(result[0]).reshape(-1) for result in task_results]
        merged_results.append(
            {
                "text": task["text"],
                "segments": task["segments"],
                "wav": waveforms[0] if len(waveforms) == 1 else np.concatenate(waveforms, axis=0),
                "successful_seeds": [result[1] for result in task_results],
                "segment_processing_seconds": [round(float(result[2]), 3) for result in task_results],
                "processing_seconds": round(sum(float(result[2]) for result in task_results), 3),
            }
        )
    return merged_results


def build_prompt_kwargs(mode: str, reference_path: Path | None, prompt_text: str, denoise: bool) -> dict | None:
    if reference_path is None:
        return None
    if mode == "reference":
        return {
            "reference_wav_path": str(reference_path),
            "denoise": denoise,
        }
    if mode == "ultimate":
        # Continuation already conditions on both the transcript and prompt
        # audio. Passing the same file again as reference audio duplicates the
        # conditioning and can make the model speak the prompt transcript.
        normalized_prompt = prompt_text.strip()
        if normalized_prompt and normalized_prompt[-1] not in "。！？!?；;，,、：:\n":
            normalized_prompt += "。"
        return {
            "prompt_wav_path": str(reference_path),
            "prompt_text": normalized_prompt,
            "denoise": denoise,
        }
    return None


def validate_request(
    model_key: str,
    mode: str,
    text: str,
    control: str,
    prompt_text: str,
    batch_mode: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
) -> None:
    if model_key not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown model")
    if mode not in {"design", "reference", "ultimate"}:
        raise HTTPException(status_code=400, detail="Unknown generation mode")
    if batch_mode not in {"ordinary", "batch"}:
        raise HTTPException(status_code=400, detail="Unknown text processing mode")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if control.strip() and model_key != "voxcpm2":
        raise HTTPException(status_code=400, detail="Voice control requires VoxCPM2")
    if control.strip() and mode == "ultimate":
        raise HTTPException(
            status_code=400,
            detail="Voice control cannot be combined with ultimate cloning",
        )
    if model_key != "voxcpm2" and mode == "reference":
        raise HTTPException(status_code=400, detail="Reference-only cloning requires VoxCPM2")
    if mode == "ultimate" and not prompt_text.strip():
        raise HTTPException(status_code=400, detail="Reference transcript is required in ultimate mode")
    if not 0.1 <= cfg_value <= 10:
        raise HTTPException(status_code=400, detail="CFG must be between 0.1 and 10")
    if not 1 <= inference_timesteps <= 100:
        raise HTTPException(status_code=400, detail="Inference steps must be between 1 and 100")
    if not 1 <= min_len <= max_len <= 4096:
        raise HTTPException(status_code=400, detail="Length range is invalid")


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    model_key: str = Form("voxcpm2"),
    lora_id: str = Form(""),
    mode: str = Form("design"),
    control: str = Form(""),
    prompt_text: str = Form(""),
    batch_mode: str = Form("ordinary"),
    batch_output_dir: str = Form(""),
    create_training_pairs: bool = Form(False),
    device: str = Form("cuda"),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    min_len: int = Form(2),
    max_len: int = Form(4096),
    seed: int = Form(42),
    normalize: bool = Form(False),
    denoise: bool = Form(True),
    optimize: bool = Form(True),
    reference_audio: UploadFile | None = File(None),
) -> dict:
    if lora_training.running:
        raise HTTPException(status_code=409, detail="LoRA training is using the GPU; stop training before inference")
    if piper_training.running:
        raise HTTPException(status_code=409, detail="Piper training is using the GPU; stop training before inference")
    request_started = time.perf_counter()
    created_at = datetime.now().astimezone()
    validate_request(
        model_key,
        mode,
        text,
        control,
        prompt_text,
        batch_mode,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
    )
    if batch_mode != "batch" and (batch_output_dir.strip() or create_training_pairs):
        raise HTTPException(status_code=400, detail="批量保存选项仅可用于批量推理模式")
    if create_training_pairs and not batch_output_dir.strip():
        raise HTTPException(status_code=400, detail="同步生成训练对前请先选择批量保存目录")
    try:
        batch_output_directory = (
            resolve_registered_batch_output_directory(batch_output_dir.strip())
            if batch_output_dir.strip()
            else None
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    lora_checkpoint = resolve_lora_checkpoint(lora_id, model_key)
    if device not in {"cuda", "cpu", *HYBRID_DEVICES}:
        raise HTTPException(status_code=400, detail="Device must be cuda, cpu, hybrid, or hybrid-max")
    if device in {"cuda", *HYBRID_DEVICES} and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available")
    if device in HYBRID_DEVICES and model_key != "voxcpm2":
        raise HTTPException(status_code=400, detail="Hybrid inference requires VoxCPM2")
    if device == "hybrid-max" and max_len > HYBRID_MAX_GENERATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Hybrid-max generation length cannot exceed {HYBRID_MAX_GENERATION_LENGTH}",
        )
    if (
        device == "cuda"
        and model_key == "voxcpm2"
        and torch.cuda.get_device_properties(0).total_memory < VOXCPM2_MIN_GPU_MEMORY
    ):
        raise HTTPException(
            status_code=507,
            detail="VoxCPM2 requires about 8 GB VRAM; this GPU has 4 GB. Select CPU or VoxCPM-0.5B.",
        )
    if mode != "design" and reference_audio is None:
        raise HTTPException(status_code=400, detail="Reference audio is required")

    source_text = text.strip()
    text_tasks = build_text_tasks(source_text, batch_mode)
    if len(text_tasks) > MAX_BATCH_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch text contains more than {MAX_BATCH_TASKS} non-empty lines",
        )
    segments = [segment for task in text_tasks for segment in task["segments"]]
    if len(segments) > MAX_INFERENCE_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Text contains more than {MAX_INFERENCE_SEGMENTS} inference segments",
        )
    final_texts = [f"({control.strip()}){segment}" if control.strip() else segment for segment in segments]
    # reduce-overhead compilation produces near-silent output with LoRALinear.
    config = ModelConfig(
        model_key,
        device,
        optimize and device in {"cuda", "hybrid"} and lora_checkpoint is None,
        denoise and reference_audio is not None,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_filename = Path(reference_audio.filename or "reference.wav").name if reference_audio else None
    reference_hash = None

    with tempfile.TemporaryDirectory(prefix="voxcpm_web_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        reference_path = await save_upload(reference_audio, temp_dir) if reference_audio else None
        if reference_path is not None:
            reference_hash = file_sha256(reference_path)
        kwargs = {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "min_len": min_len,
            "max_len": max_len,
            "normalize": normalize,
            "seed": seed,
        }
        if mode == "ultimate":
            # A failed continuation is expensive on the CPU-side autoregressive
            # path. One pace-aware attempt followed by isolated-reference fallback
            # is both faster and prevents prompt transcript leakage.
            kwargs["retry_badcase_max_times"] = 1
        prompt_build_kwargs = build_prompt_kwargs(mode, reference_path, prompt_text, denoise)

        prompt_cache_key = (
            model_key,
            lora_checkpoint["id"] if lora_checkpoint else None,
            mode,
            reference_hash,
            prompt_text.strip() if mode == "ultimate" else "",
            bool(denoise and reference_path is not None),
        ) if reference_path is not None else None

        try:
            generation_results, sample_rate, input_cache_hit = await asyncio.to_thread(
                runtime.generate_many,
                config,
                final_texts,
                prompt_cache_key,
                prompt_build_kwargs,
                lora_checkpoint,
                **kwargs,
            )
        except torch.OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=f"CUDA out of memory: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    request_inference_seconds = round(time.perf_counter() - request_started, 3)
    input_cache_created = device == "hybrid" and reference_path is not None and not input_cache_hit
    task_results = merge_text_task_results(text_tasks, generation_results)
    outputs = []
    task_count = len(task_results)
    segment_count = len(generation_results)
    final_text_index = 0
    for task_index, task_result in enumerate(task_results, start=1):
        task_created_at = datetime.now().astimezone() if task_index > 1 else created_at
        output_path = timestamped_output_path(task_created_at)
        filename = output_path.name
        wav = task_result["wav"]
        sf.write(output_path, wav, sample_rate)
        duration = round(len(wav) / sample_rate, 3)
        task_final_texts = final_texts[final_text_index : final_text_index + len(task_result["segments"])]
        final_text_index += len(task_final_texts)
        successful_seeds = task_result["successful_seeds"]
        export_path = (
            batch_export_destination(output_path, batch_output_directory, create_training_pairs)
            if batch_output_directory is not None
            else None
        )
        metadata = {
            "schema": "voxcpm-generation-v1",
            "created_at": task_created_at.isoformat(timespec="seconds"),
            "filename": filename,
            "text": task_result["text"],
            "source_text": source_text,
            "final_text": " ".join(task_final_texts),
            "batch_mode": batch_mode,
            "batch_output_dir": str(batch_output_directory) if batch_output_directory else None,
            "create_training_pairs": create_training_pairs,
            "exported_path": str(export_path) if export_path else None,
            "task_index": task_index,
            "task_count": task_count,
            "segments": task_result["segments"],
            "segment_processing_seconds": task_result["segment_processing_seconds"],
            "segment_count": len(task_result["segments"]),
            "total_segment_count": segment_count,
            "model_key": model_key,
            "lora_id": lora_checkpoint["id"] if lora_checkpoint else None,
            "lora_name": lora_checkpoint["display_name"] if lora_checkpoint else None,
            "lora_path": lora_checkpoint["path"] if lora_checkpoint else None,
            "lora_config": lora_checkpoint["lora_config"] if lora_checkpoint else None,
            "mode": mode,
            "control": control.strip(),
            "prompt_text": prompt_text.strip(),
            "device": device,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "min_len": min_len,
            "max_len": max_len,
            "requested_seed": seed,
            "successful_seed": successful_seeds[0] if len(set(successful_seeds)) == 1 else None,
            "successful_seeds": successful_seeds,
            "normalize": normalize,
            "denoise": denoise and reference_filename is not None,
            "optimize_requested": optimize,
            "optimize_effective": config.optimize,
            "input_cache_hit": input_cache_hit,
            "input_cache_created": input_cache_created,
            "reference_filename": reference_filename,
            "reference_sha256": reference_hash,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "processing_seconds": request_inference_seconds,
            "task_processing_seconds": task_result["processing_seconds"],
            "torch_version": torch.__version__,
        }
        write_wav_metadata(output_path, metadata)
        try:
            export_result = (
                export_batch_audio(output_path, export_path, task_result["text"], create_training_pairs)
                if export_path is not None
                else {"exported_path": None, "lab_path": None}
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"批量音频保存失败: {exc}") from exc
        outputs.append(
            {
                "audio_url": f"/api/audio/{filename}",
                "filename": filename,
                "text": task_result["text"],
                "task_index": task_index,
                "task_count": task_count,
                "sentence_count": len(task_result["segments"]),
                "duration": duration,
                "processing_seconds": task_result["processing_seconds"],
                "successful_seed": metadata["successful_seed"],
                "successful_seeds": successful_seeds,
                **export_result,
            }
        )

    processing_seconds = round(time.perf_counter() - request_started, 3)
    first_output = outputs[0]
    return {
        **first_output,
        "outputs": outputs,
        "sample_rate": sample_rate,
        "model": model_key,
        "lora_id": lora_checkpoint["id"] if lora_checkpoint else None,
        "lora_name": lora_checkpoint["display_name"] if lora_checkpoint else None,
        "device": device,
        "batch_mode": batch_mode,
        "batch_output_dir": str(batch_output_directory) if batch_output_directory else None,
        "training_pairs_created": bool(batch_output_directory and create_training_pairs),
        "task_count": task_count,
        "segment_count": segment_count,
        "processing_seconds": processing_seconds,
        "input_cache_hit": input_cache_hit,
        "input_cache_created": input_cache_created,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM local web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8810)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
