from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parent
VOXCPM_CACHE_ROOT = Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm"))
DISTILL_PAGE = ROOT / "distill.html"
EXPORT_PAGE = ROOT / "export.html"
OPTIMIZER_PAGE = ROOT / "optimizer.html"
PIPER_ROOT = ROOT / "piper"
PIPER_MODELS_ROOT = PIPER_ROOT / "models"
PIPER_RUNS_ROOT = PIPER_ROOT / "runs"
PIPER_DOWNLOAD_ROOT = PIPER_ROOT / "downloads"
PIPER_OUTPUT_ROOT = ROOT / "outputs" / "piper"
OPTIMIZER_HEAD_ROOT = PIPER_ROOT / "optimization-heads"
OPTIMIZER_DOWNLOAD_ROOT = PIPER_ROOT / "optimizer-downloads"
MELO_BASE_ROOT = PIPER_ROOT / "melo-bases"
MELO_BASE_DIR = MELO_BASE_ROOT / "MeloTTS-Chinese"
MELO_BASE_CHECKPOINT = MELO_BASE_DIR / "checkpoint.pth"
MELO_SOURCE_ROOT = ROOT / "third_party" / "MeloTTS"
MELO_EXPORT_CACHE_ROOT = VOXCPM_CACHE_ROOT / "melo-exports"
STUDENT_PREVIEW_CACHE_ROOT = VOXCPM_CACHE_ROOT / "student-previews"
ONNX_TEMP_ROOT = Path(os.environ.get("VOXCPM_ONNX_TEMP_DIR", r"C:\tmp\voxcpm-onnx" if os.name == "nt" else "/tmp/voxcpm-onnx"))
PIPER_ESPEAK_DATA = VOXCPM_CACHE_ROOT / "piper-espeak-data"
PIPER_RESOURCE_ROOT = VOXCPM_CACHE_ROOT / "piper-resources"
PIPER_BERT_TOKENIZER = PIPER_RESOURCE_ROOT / "bert-base-chinese"
SHERPA_RUNTIME_ROOT = Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm")) / "sherpa-models"
DEFAULT_TRAINING_DATASET = Path(r"D:\音频素材\爱弥斯语音训练集\train\wavs")
DEFAULT_PIPER_CHECKPOINT_DIR = "pretrained-zh_CN-huayan-medium"
STUDENT_MANIFEST_NAME = "voxcpm-model.json"
STUDENT_EXPORT_PRECISIONS = ("fp32", "fp16", "int8")
ONNX_CONVERSION_LOCK = threading.Lock()

# Native Melo ONNX keeps the original BERT frontend.  Set this before the
# frontend imports transformers so a standalone piper_web process uses the
# same local cache as the main WebUI.
os.environ.setdefault("HF_HOME", str(VOXCPM_CACHE_ROOT / "hf-cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

for directory in (
    PIPER_MODELS_ROOT,
    PIPER_RUNS_ROOT,
    PIPER_DOWNLOAD_ROOT,
    PIPER_OUTPUT_ROOT,
    MELO_BASE_ROOT,
    OPTIMIZER_HEAD_ROOT,
    OPTIMIZER_DOWNLOAD_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)


def prepare_piper_espeak_data() -> Path:
    if (PIPER_ESPEAK_DATA / "phontab").is_file():
        return PIPER_ESPEAK_DATA
    import piper

    source = Path(piper.__file__).resolve().parent / "espeak-ng-data"
    if not (source / "phontab").is_file():
        raise RuntimeError("piper-tts 安装中缺少 espeak-ng-data")
    shutil.copytree(source, PIPER_ESPEAK_DATA, dirs_exist_ok=True)
    return PIPER_ESPEAK_DATA


def prepare_piper_chinese_phonemizer() -> None:
    global _g2pw_local_tokenizer_configured
    if _g2pw_local_tokenizer_configured:
        return
    required = ("config.json", "tokenizer_config.json", "vocab.txt")
    if not all((PIPER_BERT_TOKENIZER / name).is_file() for name in required):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="bert-base-chinese",
            allow_patterns=list(required),
            local_dir=PIPER_BERT_TOKENIZER,
        )

    from g2pw import G2PWConverter
    from piper import phonemize_chinese

    tokenizer_dir = str(PIPER_BERT_TOKENIZER)

    def local_g2pw_converter(*args, **kwargs):
        kwargs.setdefault("model_source", tokenizer_dir)
        return G2PWConverter(*args, **kwargs)

    phonemize_chinese.G2PWConverter = local_g2pw_converter
    _g2pw_local_tokenizer_configured = True


QUALITY_PROFILES = {
    "x_low": {
        "label": "x_low · 16kHz · 约 20-35MB",
        "sample_rate": 16000,
        "inter_channels": 96,
        "hidden_channels": 96,
        "filter_channels": 384,
        "n_layers": 4,
        "upsample_initial_channel": 128,
    },
    "low": {
        "label": "low · 16kHz · 约 50-70MB",
        "sample_rate": 16000,
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_layers": 6,
        "upsample_initial_channel": 256,
    },
    "medium": {
        "label": "medium · 22.05kHz · 约 60-80MB",
        "sample_rate": 22050,
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_layers": 6,
        "upsample_initial_channel": 256,
    },
}


def safe_piper_job_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value.strip()).strip("._")
    if not name:
        name = datetime.now().strftime("piper-%y%m%d-%H%M%S")
    return name[:80]


def inspect_piper_dataset(directory: str) -> dict:
    dataset_dir = Path(directory).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"训练集目录不存在: {dataset_dir}")

    wav_paths = sorted(dataset_dir.glob("*.wav"))
    if len(wav_paths) < 2:
        raise ValueError("Piper 训练至少需要 2 条 WAV 音频")

    records = []
    sample_rates: dict[int, int] = {}
    durations = []
    missing_labs = []
    for wav_path in wav_paths:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.is_file():
            missing_labs.append(lab_path.name)
            continue
        text = lab_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"文本文件为空: {lab_path.name}")
        info = sf.info(wav_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"无效音频: {wav_path.name}")
        duration = float(info.duration)
        sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
        durations.append(duration)
        records.append({"audio": str(wav_path), "text": text})

    if missing_labs:
        preview = ", ".join(missing_labs[:5])
        suffix = "..." if len(missing_labs) > 5 else ""
        raise ValueError(f"缺少 {len(missing_labs)} 个同名 .lab: {preview}{suffix}")
    if len(records) < 2:
        raise ValueError("有效训练对不足 2 条")

    return {
        "directory": str(dataset_dir),
        "file_count": len(records),
        "total_minutes": round(sum(durations) / 60.0, 3),
        "min_duration": round(min(durations), 3),
        "max_duration": round(max(durations), 3),
        "sample_rates": sample_rates,
        "records": records,
    }


def write_piper_manifest(dataset: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.writer(manifest, delimiter="|", lineterminator="\n")
        for record in dataset["records"]:
            writer.writerow([Path(record["audio"]).name, record["text"]])


def _artifact_id(kind: str, path: Path) -> str:
    relative = path.resolve().relative_to(PIPER_ROOT.resolve()).as_posix()
    return hashlib.sha256(f"{kind}:{relative}".encode("utf-8")).hexdigest()[:16]


def _find_voice_config(path: Path) -> Path | None:
    if path.suffix.lower() == ".onnx":
        candidates = (Path(f"{path}.json"), path.with_suffix(".json"), path.parent / "config.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    for parent in (path.parent, *path.parents):
        if parent == PIPER_ROOT.parent:
            break
        for filename in ("voice.json", "config.json"):
            candidate = parent / filename
            if candidate.is_file():
                return candidate
        if parent == PIPER_ROOT:
            break
    return None


def _read_student_manifest(path: Path) -> dict | None:
    manifest_path = path.parent / STUDENT_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"学生模型清单无效: {manifest_path}") from exc
    if manifest.get("model") != path.name or not manifest.get("engine"):
        return None
    return manifest


def _manifest_resource(model_path: Path, relative_path: str) -> Path:
    root = model_path.parent.resolve()
    resource = (root / relative_path).resolve()
    if resource != root and root not in resource.parents:
        raise ValueError("学生模型清单包含越界路径")
    return resource


def _sherpa_runtime_dir(model_path: Path) -> Path:
    digest = hashlib.sha256(str(model_path.parent.resolve()).encode("utf-8")).hexdigest()[:12]
    return SHERPA_RUNTIME_ROOT / digest


def prepare_sherpa_runtime_model(model_path: Path, manifest: dict) -> Path:
    runtime_dir = _sherpa_runtime_dir(model_path)
    marker_path = runtime_dir / ".source.json"
    manifest_path = model_path.parent / STUDENT_MANIFEST_NAME
    signature = {
        "source": str(model_path.parent.resolve()),
        "model_size": model_path.stat().st_size,
        "model_mtime_ns": model_path.stat().st_mtime_ns,
        "manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
    }
    current_signature = None
    if marker_path.is_file():
        try:
            current_signature = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    runtime_model = runtime_dir / manifest["model"]
    if current_signature == signature and runtime_model.is_file():
        return runtime_model

    runtime_dir.mkdir(parents=True, exist_ok=True)
    for relative_name in manifest.get("bundle_files", []):
        source = _manifest_resource(model_path, relative_name)
        destination = runtime_dir / relative_name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"双语模型资源缺失: {source}")
    marker_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_model


def _read_voice_summary(config_path: Path | None, artifact_path: Path | None = None) -> dict:
    if config_path is None:
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    language = config.get("language") or {}
    quality = (config.get("audio") or {}).get("quality")
    if not quality and artifact_path is not None:
        path_text = "/".join(artifact_path.parts).lower()
        quality = next(
            (
                value
                for value in QUALITY_PROFILES
                if re.search(rf"(?:^|[-_/]){re.escape(value)}(?:$|[-_/.])", path_text)
            ),
            None,
        )
    return {
        "sample_rate": (config.get("audio") or {}).get("sample_rate"),
        "quality": quality,
        "language": language.get("code") or (config.get("espeak") or {}).get("voice"),
    }


def _artifact_precision(kind: str, path: Path, manifest: dict | None, quality: str | None) -> str:
    if kind != "onnx":
        return "fp32"
    values = " ".join((path.name, str(quality or ""), str((manifest or {}).get("precision", "")))).lower()
    if "int8" in values or "qint8" in values:
        return "int8"
    if "fp16" in values or "float16" in values:
        return "fp16"
    return "fp32"


def _artifact_export_precisions(kind: str, precision: str) -> list[str]:
    if kind in ("checkpoint", "melo_checkpoint"):
        return list(STUDENT_EXPORT_PRECISIONS)
    if precision == "fp32":
        return list(STUDENT_EXPORT_PRECISIONS)
    return [precision]


def _find_melo_config(path: Path) -> Path | None:
    for directory in (path.parent, path.parent.parent):
        candidate = directory / "config.json"
        if candidate.is_file():
            try:
                config = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "train" in config and "model" in config and "symbols" in config:
                return candidate
    return None


def melo_base_status() -> dict:
    installed = MELO_BASE_CHECKPOINT.is_file() and (MELO_BASE_DIR / "config.json").is_file()
    return {
        "id": "official-melotts-chinese",
        "name": "myshell-ai/MeloTTS-Chinese 官方基座",
        "installed": installed,
        "checkpoint": str(MELO_BASE_CHECKPOINT) if installed else None,
        "size_mb": round(MELO_BASE_CHECKPOINT.stat().st_size / 1024**2, 2) if installed else 0,
        "sample_rate": 44100,
        "language": "中文 + English",
        "license": "MIT",
    }


def list_piper_artifacts() -> list[dict]:
    paths: list[tuple[str, Path]] = []
    for root in (PIPER_MODELS_ROOT, PIPER_RUNS_ROOT):
        paths.extend(("onnx", path) for path in root.rglob("*.onnx") if path.is_file())
    paths.extend(("checkpoint", path) for path in PIPER_RUNS_ROOT.rglob("*.ckpt") if path.is_file())
    paths.extend(("melo_checkpoint", path) for path in PIPER_RUNS_ROOT.rglob("G_*.pth") if path.is_file())

    seen = set()
    artifacts = []
    for kind, path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        manifest = _read_student_manifest(path) if kind == "onnx" else None
        config_path = _find_melo_config(path) if kind == "melo_checkpoint" else _find_voice_config(path) if manifest is None else None
        stat = path.stat()
        relative = path.relative_to(PIPER_ROOT)
        summary = (
            {
                "sample_rate": manifest.get("sample_rate"),
                "quality": manifest.get("quality"),
                "language": manifest.get("language"),
            }
            if manifest
            else {
                "sample_rate": 44100,
                "quality": "MeloTTS FP32",
                "language": "zh_CN+en_US",
            }
            if kind == "melo_checkpoint"
            else _read_voice_summary(config_path, path)
        )
        engine = "sherpa_onnx" if kind == "melo_checkpoint" else manifest.get("engine", "piper") if manifest else "piper"
        precision = _artifact_precision(kind, path, manifest, summary.get("quality"))
        recommended = kind == "checkpoint" and DEFAULT_PIPER_CHECKPOINT_DIR in relative.parts
        display_name = path.stem
        if kind == "melo_checkpoint" and path.parent.parent != PIPER_RUNS_ROOT:
            display_name = f"{path.parent.parent.name} · {path.stem}"
        artifacts.append(
            {
                "id": _artifact_id(kind, path),
                "kind": kind,
                "name": manifest.get("display_name", path.stem) if manifest else display_name,
                "relative_path": str(relative),
                "size_mb": round(stat.st_size / 1024**2, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "has_config": config_path is not None or manifest is not None,
                "previewable": (kind == "onnx" and (config_path is not None or manifest is not None)) or (kind == "melo_checkpoint" and config_path is not None),
                "downloadable": kind in ("checkpoint", "melo_checkpoint") or (kind == "onnx" and (config_path is not None or manifest is not None)),
                "engine": engine,
                "architecture": "melotts" if engine in ("sherpa_onnx", "melo_onnx_native") else "piper",
                "engine_label": "MeloTTS" if kind == "melo_checkpoint" else manifest.get("engine_label", "Piper") if manifest else "Piper",
                "precision": precision,
                "export_precisions": _artifact_export_precisions(kind, precision),
                "license": manifest.get("license") if manifest else None,
                "recommended": recommended,
                **summary,
            }
        )
    artifacts.sort(key=lambda item: item["modified_at"], reverse=True)
    return artifacts


def resolve_piper_artifact(artifact_id: str, expected_kind: str | None = None) -> tuple[dict, Path]:
    for artifact in list_piper_artifacts():
        if artifact["id"] != artifact_id:
            continue
        if expected_kind and artifact["kind"] != expected_kind:
            raise ValueError(f"需要 {expected_kind} 文件")
        path = PIPER_ROOT / artifact["relative_path"]
        if not path.is_file():
            break
        return artifact, path
    raise ValueError("Piper 模型或检查点不存在")


def _optimizer_head_id(path: Path) -> str:
    relative = path.resolve().relative_to(OPTIMIZER_HEAD_ROOT.resolve()).as_posix()
    return hashlib.sha256(f"optimizer-head:{relative}".encode("utf-8")).hexdigest()[:16]


def list_optimizer_heads() -> list[dict]:
    heads = []
    for manifest_path in OPTIMIZER_HEAD_ROOT.glob("*/optimizer-head.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_name = Path(str(manifest.get("model", ""))).name
        model_path = manifest_path.parent / model_name
        if not model_name or not model_path.is_file():
            continue
        stat = model_path.stat()
        heads.append(
            {
                "id": _optimizer_head_id(manifest_path.parent),
                "name": manifest.get("display_name") or manifest_path.parent.name,
                "architecture": manifest.get("architecture", "tiny_crn_hybrid"),
                "precision": manifest.get("precision", "fp32"),
                "size_mb": round(stat.st_size / 1024**2, 3),
                "parameter_count": manifest.get("parameter_count"),
                "sample_rate": manifest.get("sample_rate"),
                "validation_loss": manifest.get("validation_loss"),
                "best_epoch": manifest.get("best_epoch"),
                "source_model": manifest.get("source_model"),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "model": model_name,
            }
        )
    heads.sort(key=lambda item: item["modified_at"], reverse=True)
    return heads


def resolve_optimizer_head(head_id: str) -> tuple[dict, Path, Path]:
    for head in list_optimizer_heads():
        if head["id"] != head_id:
            continue
        for manifest_path in OPTIMIZER_HEAD_ROOT.glob("*/optimizer-head.json"):
            if _optimizer_head_id(manifest_path.parent) == head_id:
                model_path = manifest_path.parent / head["model"]
                return head, model_path, manifest_path
    raise ValueError("优化头不存在")


class PiperVoiceRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voice = None
        self._identity: tuple | None = None

    def release(self) -> None:
        with self._lock:
            self._voice = None
            self._identity = None

    def synthesize(self, model_path: Path, config_path: Path, text: str, output_path: Path, settings: dict) -> None:
        from piper.config import SynthesisConfig
        from piper.voice import PiperVoice

        identity = (
            str(model_path.resolve()),
            model_path.stat().st_mtime_ns,
            config_path.stat().st_mtime_ns,
        )
        with self._lock:
            if self._voice is None or self._identity != identity:
                try:
                    voice_config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    voice_config = {}
                if voice_config.get("phoneme_type") == "pinyin":
                    prepare_piper_chinese_phonemizer()
                self._voice = PiperVoice.load(
                    model_path,
                    config_path=config_path,
                    use_cuda=False,
                    espeak_data_dir=prepare_piper_espeak_data(),
                    download_dir=PIPER_RESOURCE_ROOT,
                )
                self._identity = identity
            synthesis_config = SynthesisConfig(
                speaker_id=settings.get("speaker_id"),
                length_scale=settings["length_scale"],
                noise_scale=settings["noise_scale"],
                noise_w_scale=settings["noise_w_scale"],
                normalize_audio=True,
                volume=settings["volume"],
            )
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._voice.config.sample_rate)
                self._voice.synthesize_wav(text, wav_file, syn_config=synthesis_config, set_wav_format=False)


class SherpaVoiceRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tts = None
        self._identity: tuple | None = None

    def release(self) -> None:
        with self._lock:
            self._tts = None
            self._identity = None

    def synthesize(self, model_path: Path, manifest: dict, text: str, output_path: Path, settings: dict) -> None:
        global _sherpa_ort_dll
        if os.name == "nt" and _sherpa_ort_dll is None:
            import ctypes
            import onnxruntime

            runtime_dll = Path(onnxruntime.__file__).resolve().parent / "capi" / "onnxruntime.dll"
            if not runtime_dll.is_file():
                raise RuntimeError(f"ONNX Runtime DLL 不存在: {runtime_dll}")
            _sherpa_ort_dll = ctypes.WinDLL(str(runtime_dll))

        import sherpa_onnx

        source_model_path = model_path
        model_path = prepare_sherpa_runtime_model(source_model_path, manifest)
        tokens = _manifest_resource(model_path, manifest["tokens"])
        lexicon = _manifest_resource(model_path, manifest["lexicon"])
        dict_dir = _manifest_resource(model_path, manifest["dict_dir"])
        rule_paths = [_manifest_resource(model_path, item) for item in manifest.get("rule_fsts", [])]
        required = [tokens, lexicon, dict_dir, *rule_paths]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ValueError(f"双语模型资源缺失: {missing[0]}")

        identity = (
            str(source_model_path.resolve()),
            source_model_path.stat().st_mtime_ns,
            settings["noise_scale"],
            settings["noise_w_scale"],
            settings["length_scale"],
        )
        with self._lock:
            if self._tts is None or self._identity != identity:
                vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    lexicon=str(lexicon),
                    tokens=str(tokens),
                    dict_dir=str(dict_dir),
                    noise_scale=settings["noise_scale"],
                    noise_scale_w=settings["noise_w_scale"],
                    length_scale=settings["length_scale"],
                )
                model = sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=max(1, min(4, os.cpu_count() or 1)))
                config = sherpa_onnx.OfflineTtsConfig(
                    model=model,
                    rule_fsts=",".join(str(path) for path in rule_paths),
                    max_num_sentences=1,
                )
                self._tts = sherpa_onnx.OfflineTts(config)
                self._identity = identity

            speed = 1 / max(0.2, float(settings.get("length_scale", 1.0)))
            generated = self._tts.generate(text, sid=int(manifest.get("speaker_id", 0)), speed=speed)
            samples = np.asarray(generated.samples, dtype=np.float32) * float(settings["volume"])
            sf.write(output_path, np.clip(samples, -1.0, 1.0), int(generated.sample_rate), subtype="PCM_16")


class NativeMeloVoiceRuntime:
    """MeloTTS-native ONNX runtime that preserves the original text frontend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = None
        self._identity: tuple | None = None
        self._hps = None
        self._symbol_to_id: dict[str, int] | None = None
        self._language = "ZH_MIX_EN"
        self._sample_rate = 44100

    def release(self, release_frontend: bool = True) -> None:
        with self._lock:
            had_session = self._session is not None
            self._session = None
            self._identity = None
            self._hps = None
            self._symbol_to_id = None
        if release_frontend and (had_session or self._frontend_is_cached()):
            self._release_frontend_models()

    @staticmethod
    def _frontend_is_cached() -> bool:
        import sys

        for module_name in ("melo.text.chinese_bert", "melo.text.japanese_bert"):
            module = sys.modules.get(module_name)
            models = getattr(module, "models", None) if module is not None else None
            if isinstance(models, dict) and models:
                return True
        return False

    @staticmethod
    def _release_frontend_models() -> None:
        """Drop optional BERT weights when another GPU workload needs room."""
        import sys

        for module_name in ("melo.text.chinese_bert", "melo.text.japanese_bert"):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name in ("models", "tokenizers"):
                cache = getattr(module, name, None)
                if isinstance(cache, dict):
                    cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, model_path: Path, manifest: dict):
        import sys

        if str(MELO_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(MELO_SOURCE_ROOT))
        import onnxruntime as ort
        from melo.utils import get_hparams_from_file

        config_relative = manifest.get("config", "config.json")
        config_path = _manifest_resource(model_path, config_relative)
        identity = (
            str(model_path.resolve()),
            model_path.stat().st_mtime_ns,
            str(config_path.resolve()),
            config_path.stat().st_mtime_ns,
        )
        with self._lock:
            if self._session is None or self._identity != identity:
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                self._hps = get_hparams_from_file(str(config_path))
                symbols = list(self._hps.symbols)
                self._symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
                self._language = str(manifest.get("frontend_language", "ZH_MIX_EN"))
                self._sample_rate = int(manifest.get("sample_rate") or self._hps.data.sampling_rate)
                self._identity = identity
            return self._session, self._hps, self._symbol_to_id, self._language, self._sample_rate

    def synthesize(self, model_path: Path, manifest: dict, text: str, output_path: Path, settings: dict) -> None:
        import sys

        if str(MELO_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(MELO_SOURCE_ROOT))
        from melo.split_utils import split_sentence
        from melo.utils import get_text_for_tts_infer

        session, hps, symbol_to_id, language, sample_rate = self._load(model_path, manifest)
        pieces = split_sentence(text, language_str=language)
        if not pieces:
            raise ValueError("MeloTTS 文本前端没有生成可推理的句子")
        speed = 1 / max(0.2, float(settings.get("length_scale", 1.0)))
        noise_scale = float(settings.get("noise_scale", 0.6))
        noise_scale_w = float(settings.get("noise_w_scale", 0.8))
        sdp_ratio = float(settings.get("sdp_ratio", manifest.get("sdp_ratio", 0.2)))
        speaker_id = int(manifest.get("speaker_id", 1))
        frontend_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        audio_segments = []
        with self._lock:
            for piece in pieces:
                if language in ("EN", "ZH_MIX_EN"):
                    piece = re.sub(r"([a-z])([A-Z])", r"\1 \2", piece)
                bert, ja_bert, phones, tones, language_ids = get_text_for_tts_infer(
                    piece,
                    language,
                    hps,
                    frontend_device,
                    symbol_to_id,
                )
                inputs = {
                    "x": phones.unsqueeze(0).numpy().astype(np.int64),
                    "x_lengths": np.asarray([phones.numel()], dtype=np.int64),
                    "tones": tones.unsqueeze(0).numpy().astype(np.int64),
                    "language": language_ids.unsqueeze(0).numpy().astype(np.int64),
                    "bert": bert.unsqueeze(0).numpy().astype(np.float32),
                    "ja_bert": ja_bert.unsqueeze(0).numpy().astype(np.float32),
                    "sid": np.asarray([speaker_id], dtype=np.int64),
                    "noise_scale": np.asarray([noise_scale], dtype=np.float32),
                    "length_scale": np.asarray([float(1 / speed)], dtype=np.float32),
                    "noise_scale_w": np.asarray([noise_scale_w], dtype=np.float32),
                    "sdp_ratio": np.asarray([sdp_ratio], dtype=np.float32),
                }
                audio = np.asarray(session.run(["y"], inputs)[0], dtype=np.float32)[0, 0]
                audio_segments.append(audio)
                audio_segments.append(np.zeros(int(sample_rate * 0.05 / speed), dtype=np.float32))
        samples = np.concatenate(audio_segments).astype(np.float32)
        samples *= float(settings.get("volume", 1.0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_16")


def _latest_checkpoint(job_dir: Path) -> Path | None:
    checkpoints = [path for path in job_dir.rglob("*.ckpt") if path.is_file()]
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns) if checkpoints else None


def export_piper_checkpoint(checkpoint_path: Path, output_path: Path, config_path: Path) -> Path:
    python_path = ROOT / ".venv" / "Scripts" / "python.exe"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        "-m",
        "piper.train.export_onnx",
        "--checkpoint",
        str(checkpoint_path),
        "--output-file",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stdout + "\n" + completed.stderr).strip()[-4000:])
    shutil.copy2(config_path, Path(f"{output_path}.json"))
    return output_path


def convert_onnx_precision(source_path: Path, output_path: Path, precision: str) -> Path:
    precision = str(precision).strip().lower()
    if precision not in STUDENT_EXPORT_PRECISIONS:
        raise ValueError("导出精度仅支持 FP32、FP16 或 INT8")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("导出路径不能覆盖源 ONNX")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{time.time_ns():x}.tmp.onnx")
    temporary_path.unlink(missing_ok=True)
    try:
        if precision == "fp32":
            shutil.copy2(source_path, temporary_path)
        elif precision == "fp16":
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = onnx.load(source_path)
            converted = convert_float_to_float16(model, keep_io_types=True)
            _sort_onnx_graph(converted.graph)
            onnx.checker.check_model(converted)
            onnx.save(converted, temporary_path)
        else:
            import onnx
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_cache = STUDENT_PREVIEW_CACHE_ROOT / "onnx-quant" / hashlib.sha256(
                f"{source_path.resolve()}:{source_path.stat().st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:16]
            quant_source = quant_cache / "model.onnx"
            quant_cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, quant_source)
            ONNX_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            with ONNX_CONVERSION_LOCK:
                previous_tempdir = tempfile.tempdir
                tempfile.tempdir = str(ONNX_TEMP_ROOT)
                try:
                    quantize_dynamic(
                        str(quant_source),
                        str(temporary_path),
                        weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm", "Gather"],
                    )
                finally:
                    tempfile.tempdir = previous_tempdir
                    shutil.rmtree(quant_cache, ignore_errors=True)
            onnx.checker.check_model(onnx.load(temporary_path))
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _sort_onnx_graph(graph) -> None:
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute.GRAPH:
                _sort_onnx_graph(attribute.g)
            elif attribute.type == attribute.GRAPHS:
                for child in attribute.graphs:
                    _sort_onnx_graph(child)
    remaining = list(graph.node)
    produced = {name for node in remaining for name in node.output if name}
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(
        name
        for node in remaining
        for name in node.input
        if name and name not in produced
    )
    ordered = []
    while remaining:
        ready = [node for node in remaining if all(not name or name in available for name in node.input)]
        if not ready:
            raise RuntimeError("FP16 ONNX 转换后无法建立有效的节点拓扑")
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
            remaining.remove(node)
    del graph.node[:]
    graph.node.extend(ordered)


def export_piper_checkpoint_precision(
    checkpoint_path: Path,
    output_path: Path,
    config_path: Path,
    precision: str,
) -> Path:
    precision = str(precision).strip().lower()
    if precision not in STUDENT_EXPORT_PRECISIONS:
        raise ValueError("导出精度仅支持 FP32、FP16 或 INT8")
    if precision == "fp32":
        return export_piper_checkpoint(checkpoint_path, output_path, config_path)
    cache_dir = STUDENT_PREVIEW_CACHE_ROOT / "exports" / hashlib.sha256(
        f"{checkpoint_path.resolve()}:{checkpoint_path.stat().st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    fp32_path = cache_dir / "model.fp32.onnx"
    if not fp32_path.is_file():
        export_piper_checkpoint(checkpoint_path, fp32_path, config_path)
    convert_onnx_precision(fp32_path, output_path, precision)
    shutil.copy2(config_path, Path(f"{output_path}.json"))
    return output_path


def preview_melo_checkpoint(checkpoint_path: Path, config_path: Path, text: str, output_path: Path, speed: float) -> Path:
    python_path = ROOT / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HOME", str(Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm")) / "hf-cache"))
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    command = [
        str(python_path),
        "-X",
        "utf8",
        "scripts/preview_melo_checkpoint.py",
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--text",
        text,
        "--output",
        str(output_path),
        "--speed",
        str(speed),
        "--device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError((completed.stdout + "\n" + completed.stderr).strip()[-5000:])
    return output_path


def export_melo_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    config_path: Path,
    precision: str = "int8",
) -> Path:
    precision = str(precision).strip().lower()
    if precision not in STUDENT_EXPORT_PRECISIONS:
        raise ValueError("导出精度仅支持 FP32、FP16 或 INT8")
    try:
        melo_config = json.loads(config_path.read_text(encoding="utf-8"))
        sample_rate = int(melo_config["data"]["sampling_rate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("MeloTTS 检查点缺少有效的采样率配置") from exc
    python_path = ROOT / ".venv" / "Scripts" / "python.exe"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model.{precision}.onnx"
    cache_key = hashlib.sha256(
        f"{checkpoint_path.resolve()}:{checkpoint_path.stat().st_mtime_ns}:{config_path.stat().st_mtime_ns}:{precision}:native-v1".encode("utf-8")
    ).hexdigest()[:16]
    runtime_dir = MELO_EXPORT_CACHE_ROOT / cache_key
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_output = runtime_dir / f"model.{precision}.onnx"
    command = [
        str(python_path),
        "-X",
        "utf8",
        "scripts/export_melo_onnx.py",
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--output",
        str(runtime_output),
        "--precision",
        precision,
        "--runtime",
        "native",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stdout + "\n" + completed.stderr).strip()[-5000:])
    shutil.copy2(runtime_output, output_path)
    shutil.rmtree(runtime_dir, ignore_errors=True)

    bundled_config = output_dir / "config.json"
    shutil.copy2(config_path, bundled_config)
    bundle_files = [output_path.name, bundled_config.name]
    license_path = MELO_SOURCE_ROOT / "LICENSE"
    if license_path.is_file():
        shutil.copy2(license_path, output_dir / license_path.name)
        bundle_files.append(license_path.name)
    bundle_files.append(STUDENT_MANIFEST_NAME)
    manifest = {
        "engine": "melo_onnx_native",
        "engine_label": "MeloTTS Native ONNX",
        "display_name": f"{output_dir.name} · Native · {precision.upper()}",
        "model": output_path.name,
        "config": bundled_config.name,
        "frontend_language": "ZH_MIX_EN",
        "sample_rate": sample_rate,
        "quality": f"{precision}-finetuned",
        "precision": precision,
        "language": "zh_CN+en_US",
        "license": "MIT",
        "speaker_id": 1,
        "noise_scale": 0.6,
        "noise_scale_w": 0.8,
        "sdp_ratio": 0.2,
        "runtime_requirement": "VoxCPM embedded MeloTTS frontend",
        "source_checkpoint": str(checkpoint_path.relative_to(PIPER_ROOT)),
        "bundle_files": bundle_files,
    }
    (output_dir / STUDENT_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def export_existing_onnx_precision(artifact: dict, source_path: Path, precision: str) -> Path:
    precision = str(precision).strip().lower()
    if precision not in artifact.get("export_precisions", []):
        raise ValueError(f"{artifact['precision'].upper()} ONNX 不能转换为 {precision.upper()}")
    if precision == artifact["precision"]:
        return source_path
    output_name = safe_piper_job_name(f"{source_path.parent.name}-{source_path.stem}-{precision}")
    output_dir = PIPER_MODELS_ROOT / output_name
    manifest = _read_student_manifest(source_path)
    if manifest is not None:
        output_path = output_dir / f"model.{precision}.onnx"
        convert_onnx_precision(source_path, output_path, precision)
        bundle_files = []
        for relative_name in manifest.get("bundle_files", []):
            if relative_name in (manifest.get("model"), STUDENT_MANIFEST_NAME):
                continue
            source = _manifest_resource(source_path, relative_name)
            destination = output_dir / relative_name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            bundle_files.append(relative_name)
        bundle_files = [output_path.name, *bundle_files, STUDENT_MANIFEST_NAME]
        exported_manifest = {
            **manifest,
            "display_name": f"{artifact['name']} · {precision.upper()}",
            "model": output_path.name,
            "quality": precision,
            "precision": precision,
            "source_model": artifact["relative_path"],
            "bundle_files": bundle_files,
        }
        (output_dir / STUDENT_MANIFEST_NAME).write_text(
            json.dumps(exported_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    config_path = _find_voice_config(source_path)
    if config_path is None:
        raise ValueError("Piper ONNX 缺少 JSON 配置")
    output_path = output_dir / f"{safe_piper_job_name(source_path.stem)}.{precision}.onnx"
    convert_onnx_precision(source_path, output_path, precision)
    shutil.copy2(config_path, Path(f"{output_path}.json"))
    return output_path


def preview_piper_checkpoint(
    artifact: dict,
    checkpoint_path: Path,
    text: str,
    settings: dict,
) -> dict:
    config_path = _find_voice_config(checkpoint_path)
    if config_path is None:
        raise ValueError("Piper 检查点目录缺少 voice.json/config.json")
    model_cache = STUDENT_PREVIEW_CACHE_ROOT / artifact["id"]
    model_path = model_cache / "model.fp32.onnx"
    signature_path = model_cache / "source.json"
    signature = {
        "checkpoint": str(checkpoint_path.resolve()),
        "size": checkpoint_path.stat().st_size,
        "mtime_ns": checkpoint_path.stat().st_mtime_ns,
    }
    cached_signature = None
    if signature_path.is_file():
        try:
            cached_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if cached_signature != signature or not model_path.is_file():
        model_cache.mkdir(parents=True, exist_ok=True)
        export_piper_checkpoint(checkpoint_path, model_path, config_path)
        signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(
        json.dumps({"source": signature, "text": text, **settings}, sort_keys=True, ensure_ascii=True).encode("ascii")
    ).hexdigest()[:8]
    output_path = PIPER_OUTPUT_ROOT / f"piper-ckpt-{datetime.now():%y%m%d}-{digest}.wav"
    cached = output_path.is_file()
    if not cached:
        piper_voice_runtime.synthesize(model_path, Path(f"{model_path}.json"), text, output_path, settings)
    info = sf.info(output_path)
    return {
        "filename": output_path.name,
        "audio_url": f"/api/piper/audio/{output_path.name}",
        "duration": round(float(info.duration), 3),
        "cached": cached,
        "model": artifact["name"],
    }


class PiperTrainingRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=3000)
        self._status = "idle"
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._job_name: str | None = None
        self._job_dir: Path | None = None
        self._auto_export: str | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def _append(self, line: str) -> None:
        with self._lock:
            self._logs.append(line.rstrip("\r\n"))

    def start(self, config_path: Path, job_name: str, job_dir: Path) -> None:
        prepare_piper_espeak_data()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Piper 训练任务已经在运行")
            python_path = ROOT / ".venv" / "Scripts" / "python.exe"
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            self._process = subprocess.Popen(
                [str(python_path), "scripts/train_piper_local.py", "fit", "--config", str(config_path)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._logs.clear()
            self._logs.append(f"Piper training started: {job_name}")
            self._status = "running"
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._job_name = job_name
            self._job_dir = job_dir
            self._auto_export = None
            process = self._process
        threading.Thread(target=self._read_process, args=(process,), daemon=True).start()

    def _read_process(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append(line)
        returncode = process.wait()
        with self._lock:
            stopping = self._status == "stopping"
            self._returncode = returncode
            self._finished_at = time.time()
            self._status = "stopped" if stopping else "completed" if returncode == 0 else "failed"

        if returncode == 0 and self._job_dir is not None and self._job_name is not None:
            checkpoint = _latest_checkpoint(self._job_dir)
            config_path = self._job_dir / "voice.json"
            if checkpoint and config_path.is_file():
                self._append(f"Exporting latest checkpoint: {checkpoint.name}")
                with self._lock:
                    self._status = "exporting"
                output_path = PIPER_MODELS_ROOT / self._job_name / f"{self._job_name}-latest.onnx"
                try:
                    export_piper_checkpoint(checkpoint, output_path, config_path)
                    self._auto_export = str(output_path)
                    self._append(f"Exported ONNX: {output_path}")
                except Exception as exc:
                    self._append(f"Automatic ONNX export failed: {exc}")
                finally:
                    with self._lock:
                        self._status = "completed"

    def stop(self) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return False
            self._status = "stopping"
            self._logs.append("Stopping Piper training; saved checkpoints will be kept.")
            self._process.terminate()
            return True

    def snapshot(self) -> dict:
        with self._lock:
            now = self._finished_at or time.time()
            elapsed = now - self._started_at if self._started_at else 0.0
            return {
                "status": self._status,
                "running": self._process is not None and self._process.poll() is None,
                "job_name": self._job_name,
                "returncode": self._returncode,
                "elapsed_seconds": round(elapsed, 1),
                "logs": "\n".join(self._logs),
                "auto_export": self._auto_export,
                "started_at": self._started_at or 0,
                "engine": "piper",
            }


class MeloTrainingRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=4000)
        self._status = "idle"
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._job_name: str | None = None
        self._job_dir: Path | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def _append(self, line: str) -> None:
        with self._lock:
            self._logs.append(line.rstrip("\r\n"))

    def start(self, job_config: Path, job_name: str, job_dir: Path) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("MeloTTS 训练任务已经在运行")
            python_path = ROOT / ".venv" / "Scripts" / "python.exe"
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env.setdefault("HF_HOME", str(Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm")) / "hf-cache"))
            env["HF_HUB_DISABLE_XET"] = "1"
            env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self._process = subprocess.Popen(
                [str(python_path), "-X", "utf8", "scripts/train_melo_local.py", str(job_config)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._logs.clear()
            self._logs.append(f"MeloTTS training started: {job_name}")
            self._status = "running"
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._job_name = job_name
            self._job_dir = job_dir
            process = self._process
        threading.Thread(target=self._read_process, args=(process,), daemon=True).start()

    def _read_process(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append(line)
        returncode = process.wait()
        with self._lock:
            stopping = self._status == "stopping"
            self._returncode = returncode
            self._finished_at = time.time()
            self._status = "stopped" if stopping else "completed" if returncode == 0 else "failed"

    def stop(self) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return False
            process = self._process
            self._status = "stopping"
            self._logs.append("Stopping MeloTTS training; saved checkpoints and input cache will be kept.")
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
        return True

    def snapshot(self) -> dict:
        with self._lock:
            now = self._finished_at or time.time()
            elapsed = now - self._started_at if self._started_at else 0.0
            return {
                "status": self._status,
                "running": self._process is not None and self._process.poll() is None,
                "job_name": self._job_name,
                "returncode": self._returncode,
                "elapsed_seconds": round(elapsed, 1),
                "logs": "\n".join(self._logs),
                "auto_export": None,
                "started_at": self._started_at or 0,
                "engine": "sherpa_onnx",
            }


class OptimizerHeadTrainingRuntime:
    _progress_pattern = re.compile(
        r"^\[Head\]\[(?P<phase>[^\]]+)\]\s+(?P<current>\d+)/(?P<total>\d+)\s*(?P<detail>.*)$"
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=5000)
        self._status = "idle"
        self._phase = "idle"
        self._current = 0
        self._total = 0
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._job_name: str | None = None
        self._job_dir: Path | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, job_config: Path, job_name: str, job_dir: Path) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("优化头训练任务已经在运行")
            python_path = ROOT / ".venv" / "Scripts" / "python.exe"
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env.setdefault("HF_HOME", str(Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm")) / "hf-cache"))
            self._process = subprocess.Popen(
                [str(python_path), "-X", "utf8", "scripts/train_optimizer_head.py", str(job_config)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._logs.clear()
            self._logs.append(f"优化头任务已启动: {job_name}")
            self._status = "running"
            self._phase = "prepare"
            self._current = 0
            self._total = 0
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._job_name = job_name
            self._job_dir = job_dir
            process = self._process
        threading.Thread(target=self._read_process, args=(process,), daemon=True).start()

    def _read_process(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            clean_line = line.rstrip("\r\n")
            with self._lock:
                self._logs.append(clean_line)
                match = self._progress_pattern.match(clean_line)
                if match:
                    self._phase = match.group("phase")
                    self._current = int(match.group("current"))
                    self._total = int(match.group("total"))
        returncode = process.wait()
        with self._lock:
            stopping = self._status == "stopping"
            self._returncode = returncode
            self._finished_at = time.time()
            self._status = "stopped" if stopping else "completed" if returncode == 0 else "failed"
            if returncode == 0:
                self._phase = "complete"

    def stop(self) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return False
            process = self._process
            self._status = "stopping"
            self._logs.append("正在停止优化头任务；已生成的学生音频和对齐缓存会保留。")
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
        return True

    def snapshot(self) -> dict:
        with self._lock:
            now = self._finished_at or time.time()
            elapsed = now - self._started_at if self._started_at else 0.0
            phase_progress = self._current / self._total if self._total else 0.0
            if self._phase == "synthesize":
                progress = 0.6 * max(0.0, (self._current - 0.5) / max(1, self._total))
            elif self._phase in ("prepare", "align"):
                progress = 0.6 * phase_progress
            elif self._phase == "train":
                progress = 0.6 + 0.32 * phase_progress
            elif self._phase == "validate":
                progress = min(0.92, 0.6 + 0.32 * phase_progress)
            elif self._phase == "export":
                progress = 0.92 + 0.08 * phase_progress
            elif self._phase == "complete":
                progress = 1.0
            else:
                progress = 0.0
            output = None
            request_config = None
            if self._job_dir is not None:
                manifest_path = self._job_dir / "optimizer-head.json"
                if manifest_path.is_file():
                    try:
                        output = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        output = None
                config_path = self._job_dir / "job.json"
                if config_path.is_file():
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        request_config = {
                            key: config.get(key)
                            for key in (
                                "architecture",
                                "epochs",
                                "batch_size",
                                "learning_rate",
                                "training_precision",
                                "export_precision",
                                "validation_split",
                                "chunk_frames",
                                "dataset_dir",
                            )
                        }
                        request_config["model_id"] = (config.get("source_artifact") or {}).get("id")
                    except (OSError, json.JSONDecodeError):
                        request_config = None
            return {
                "status": self._status,
                "running": self._process is not None and self._process.poll() is None,
                "job_name": self._job_name,
                "returncode": self._returncode,
                "elapsed_seconds": round(elapsed, 1),
                "logs": "\n".join(self._logs),
                "phase": self._phase,
                "current": self._current,
                "total": self._total,
                "progress": round(min(1.0, max(0.0, progress)), 4),
                "output": output,
                "request_config": request_config,
                "started_at": self._started_at or 0,
            }


piper_voice_runtime = PiperVoiceRuntime()
sherpa_voice_runtime = SherpaVoiceRuntime()
melo_native_voice_runtime = NativeMeloVoiceRuntime()
piper_training = PiperTrainingRuntime()
melo_training = MeloTrainingRuntime()
optimizer_head_training = OptimizerHeadTrainingRuntime()
_release_inference: Callable[[], None] = lambda: None
_other_training_running: Callable[[], bool] = lambda: False
_sherpa_ort_dll = None
_g2pw_local_tokenizer_configured = False


def student_training_running() -> bool:
    return piper_training.running or melo_training.running or optimizer_head_training.running


def configure_piper_callbacks(release_inference: Callable[[], None], other_training_running: Callable[[], bool]) -> None:
    global _release_inference, _other_training_running
    _release_inference = release_inference
    _other_training_running = other_training_running


def release_native_melo_frontend() -> None:
    """Free the native Melo BERT frontend before loading a large VoxCPM model."""
    melo_native_voice_runtime.release()


router = APIRouter()


@router.get("/distill")
def distill_index() -> FileResponse:
    return FileResponse(DISTILL_PAGE)


@router.get("/export")
def export_index() -> FileResponse:
    return FileResponse(EXPORT_PAGE)


@router.get("/optimizer")
def optimizer_index() -> FileResponse:
    return FileResponse(OPTIMIZER_PAGE)


@router.get("/api/piper/status")
def piper_status() -> dict:
    artifacts = list_piper_artifacts()
    training_states = [piper_training.snapshot(), melo_training.snapshot()]
    running_state = next((state for state in training_states if state["running"]), None)
    active_state = running_state or max(training_states, key=lambda state: state["started_at"])
    return {
        **active_state,
        "active_engine": active_state["engine"],
        "training_states": {state["engine"]: state for state in training_states},
        "runtime_installed": True,
        "training_installed": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available()
        else 0,
        "default_dataset": str(DEFAULT_TRAINING_DATASET),
        "quality_profiles": QUALITY_PROFILES,
        "melo_base": melo_base_status(),
        "student_engines": [
            {"id": "piper", "label": "Piper", "trainable": True},
            {"id": "sherpa_onnx", "label": "MeloTTS 中英双语", "trainable": True},
        ],
        "models": [item for item in artifacts if item["kind"] == "onnx"],
        "checkpoints": [item for item in artifacts if item["kind"] == "checkpoint"],
        "melo_checkpoints": [item for item in artifacts if item["kind"] == "melo_checkpoint"],
        "optimizer_running": optimizer_head_training.running,
    }


@router.get("/api/export/artifacts")
def export_artifacts() -> dict:
    artifacts = list_piper_artifacts()
    return {
        "artifacts": artifacts,
        "architectures": [
            {"id": "piper", "label": "Piper"},
            {"id": "melotts", "label": "MeloTTS / Sherpa-ONNX"},
        ],
        "precisions": [
            {"id": "fp32", "label": "FP32", "description": "最高兼容性，体积最大"},
            {"id": "fp16", "label": "FP16", "description": "约半体积，依赖运行端 FP16 算子支持"},
            {"id": "int8", "label": "INT8", "description": "边缘 CPU 推荐，体积与内存最低"},
        ],
        "busy": student_training_running(),
    }


@router.get("/api/optimizer/status")
def optimizer_status() -> dict:
    artifacts = [
        item
        for item in list_piper_artifacts()
        if item["kind"] == "onnx" and item["previewable"] and item["engine"] in ("piper", "sherpa_onnx", "melo_onnx_native")
    ]
    return {
        **optimizer_head_training.snapshot(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available()
        else 0,
        "default_dataset": str(DEFAULT_TRAINING_DATASET),
        "models": artifacts,
        "heads": list_optimizer_heads(),
        "architectures": [
            {
                "id": "tiny_crn_hybrid",
                "label": "Tiny-CRN 混合频谱优化",
                "description": "约 178 万参数，以频谱掩码和有界残差修复沙音并保留源相位。",
            }
        ],
        "training_precisions": ["fp32", "fp16", "bf16"],
        "export_precisions": list(STUDENT_EXPORT_PRECISIONS),
        "blocked_by_student_training": piper_training.running or melo_training.running,
        "blocked_by_lora_training": _other_training_running(),
    }


@router.post("/api/optimizer/dataset")
async def optimizer_dataset(dataset_dir: str = Form(...)) -> dict:
    try:
        dataset = await asyncio.to_thread(inspect_piper_dataset, dataset_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {key: value for key, value in dataset.items() if key != "records"}


@router.post("/api/optimizer/train")
async def start_optimizer_training(
    model_id: str = Form(...),
    dataset_dir: str = Form(...),
    architecture: str = Form("tiny_crn_hybrid"),
    output_name: str = Form(""),
    epochs: int = Form(30),
    batch_size: int = Form(4),
    learning_rate: float = Form(0.0003),
    training_precision: str = Form("fp32"),
    export_precision: str = Form("fp32"),
    validation_split: float = Form(0.1),
    chunk_frames: int = Form(192),
) -> dict:
    if optimizer_head_training.running:
        raise HTTPException(status_code=409, detail="优化头任务已经在运行")
    if piper_training.running or melo_training.running:
        raise HTTPException(status_code=409, detail="学生模型训练正在运行，请先停止或等待完成")
    if _other_training_running():
        raise HTTPException(status_code=409, detail="LoRA 训练正在运行，请先暂停")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="优化头训练需要 CUDA")
    if architecture != "tiny_crn_hybrid":
        raise HTTPException(status_code=400, detail="未知优化头架构")
    if training_precision not in ("fp32", "fp16", "bf16"):
        raise HTTPException(status_code=400, detail="训练精度无效")
    if export_precision not in STUDENT_EXPORT_PRECISIONS:
        raise HTTPException(status_code=400, detail="导出精度无效")
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 32:
        raise HTTPException(status_code=400, detail="训练轮数或批大小无效")
    if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 0.01:
        raise HTTPException(status_code=400, detail="学习率无效")
    if not math.isfinite(validation_split) or not 0.05 <= validation_split <= 0.3:
        raise HTTPException(status_code=400, detail="验证集比例无效")
    if not 64 <= chunk_frames <= 512:
        raise HTTPException(status_code=400, detail="频谱片段长度无效")

    try:
        artifact, model_path = resolve_piper_artifact(model_id, "onnx")
        if artifact["engine"] not in ("piper", "sherpa_onnx", "melo_onnx_native") or not artifact["previewable"]:
            raise ValueError("该 ONNX 缺少批量合成所需的运行配置")
        dataset = await asyncio.to_thread(inspect_piper_dataset, dataset_dir)
        if batch_size > dataset["file_count"]:
            raise ValueError("批大小不能超过训练对数量")
        manifest_path = model_path.parent / STUDENT_MANIFEST_NAME if artifact["engine"] in ("sherpa_onnx", "melo_onnx_native") else None
        config_path = _find_voice_config(model_path) if artifact["engine"] == "piper" else None
        if manifest_path is not None and not manifest_path.is_file():
            raise ValueError("MeloTTS ONNX 缺少学生模型清单")
        if artifact["engine"] == "piper" and config_path is None:
            raise ValueError("Piper ONNX 缺少语音配置")
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base_name = output_name.strip() or f"head-{datetime.now():%y%m%d-%H%M%S}"
    job_name = safe_piper_job_name(base_name)
    job_dir = OPTIMIZER_HEAD_ROOT / job_name
    if job_dir.exists() and any(job_dir.iterdir()):
        raise HTTPException(status_code=409, detail="同名优化头任务已经存在，请更换任务名称")
    job_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int(artifact.get("sample_rate") or 22050)
    n_fft = 1024 if sample_rate >= 32000 else 512
    config = {
        "job_name": job_name,
        "job_dir": str(job_dir.resolve()),
        "dataset_dir": dataset["directory"],
        "model_path": str(model_path.resolve()),
        "model_engine": artifact["engine"],
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "config_path": str(config_path.resolve()) if config_path else None,
        "source_artifact": {key: value for key, value in artifact.items() if key != "export_precisions"},
        "architecture": architecture,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_precision": training_precision,
        "export_precision": export_precision,
        "validation_split": validation_split,
        "chunk_frames": chunk_frames,
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": n_fft // 4,
        "synthesis": {
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w_scale": 0.8,
            "volume": 1.0,
        },
    }
    config_path_out = job_dir / "job.json"
    config_path_out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    await asyncio.to_thread(_release_inference)
    piper_voice_runtime.release()
    sherpa_voice_runtime.release()
    melo_native_voice_runtime.release()
    try:
        optimizer_head_training.start(config_path_out, job_name, job_dir)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"优化头任务启动失败: {exc}") from exc
    return {
        "status": "running",
        "job_name": job_name,
        "job_dir": str(job_dir),
        "model": artifact,
        "dataset": {key: value for key, value in dataset.items() if key != "records"},
    }


@router.post("/api/optimizer/stop")
def stop_optimizer_training() -> dict:
    if not optimizer_head_training.stop():
        raise HTTPException(status_code=409, detail="没有正在运行的优化头任务")
    return {"status": "stopping"}


@router.get("/api/optimizer/download/{head_id}")
def download_optimizer_head(head_id: str) -> FileResponse:
    try:
        head, model_path, manifest_path = resolve_optimizer_head(head_id)
        bundle_path = OPTIMIZER_DOWNLOAD_ROOT / f"{safe_piper_job_name(head['name'])}-{head['precision']}.zip"
        source_signature = max(model_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)
        if not bundle_path.is_file() or bundle_path.stat().st_mtime_ns < source_signature:
            temporary_path = bundle_path.with_name(f".{bundle_path.stem}-{time.time_ns():x}.tmp.zip")
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(model_path, arcname=model_path.name)
                archive.write(manifest_path, arcname=manifest_path.name)
            temporary_path.replace(bundle_path)
        return FileResponse(bundle_path, media_type="application/zip", filename=bundle_path.name)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/export/preview")
async def preview_student_artifact(
    artifact_id: str = Form(...),
    text: str = Form(...),
    length_scale: float = Form(1.0),
    noise_scale: float = Form(0.667),
    noise_w_scale: float = Form(0.8),
    volume: float = Form(1.0),
) -> dict:
    if optimizer_head_training.running:
        raise HTTPException(status_code=409, detail="优化头任务运行时不能试听学生模型")
    clean_text = text.strip()
    if not clean_text or len(clean_text) > 500:
        raise HTTPException(status_code=400, detail="试听文本长度必须为 1-500 个字符")
    values = (length_scale, noise_scale, noise_w_scale, volume)
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=400, detail="试听参数无效")
    if not 0.2 <= length_scale <= 3 or not 0 <= noise_scale <= 2 or not 0 <= noise_w_scale <= 2 or not 0.1 <= volume <= 3:
        raise HTTPException(status_code=400, detail="试听参数超出范围")
    try:
        artifact, artifact_path = resolve_piper_artifact(artifact_id)
        if artifact["kind"] == "onnx":
            return await piper_preview(
                model_id=artifact_id,
                text=clean_text,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
                volume=volume,
            )
        if artifact["kind"] == "melo_checkpoint":
            return await melo_checkpoint_preview(
                checkpoint_id=artifact_id,
                text=clean_text,
                speed=1 / length_scale,
                volume=volume,
            )
        if artifact["kind"] != "checkpoint":
            raise ValueError("该学生资产不支持试听")
        if student_training_running():
            raise HTTPException(status_code=409, detail="训练运行时不能试听检查点")
        settings = {
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w_scale": noise_w_scale,
            "volume": volume,
            "speaker_id": None,
        }
        return await asyncio.to_thread(
            preview_piper_checkpoint,
            artifact,
            artifact_path,
            clean_text,
            settings,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/piper/dataset")
async def piper_dataset(dataset_dir: str = Form(...)) -> dict:
    try:
        dataset = await asyncio.to_thread(inspect_piper_dataset, dataset_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {key: value for key, value in dataset.items() if key != "records"}


@router.post("/api/melo/base/download")
async def download_melo_base() -> dict:
    if student_training_running():
        raise HTTPException(status_code=409, detail="训练运行时不能下载基座")

    def download() -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "myshell-ai/MeloTTS-Chinese",
            allow_patterns=["checkpoint.pth", "config.json", "README.md"],
            local_dir=MELO_BASE_DIR,
            endpoint="https://hf-mirror.com",
        )

    try:
        await asyncio.to_thread(download)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MeloTTS 官方基座下载失败: {exc}") from exc
    return {"base": melo_base_status()}


@router.post("/api/piper/preview")
async def piper_preview(
    model_id: str = Form(...),
    text: str = Form(...),
    length_scale: float = Form(1.0),
    noise_scale: float = Form(0.667),
    noise_w_scale: float = Form(0.8),
    volume: float = Form(1.0),
) -> dict:
    if optimizer_head_training.running:
        raise HTTPException(status_code=409, detail="优化头任务运行时不能试听学生模型")
    clean_text = text.strip()
    if not clean_text or len(clean_text) > 500:
        raise HTTPException(status_code=400, detail="试听文本长度必须为 1-500 个字符")
    values = (length_scale, noise_scale, noise_w_scale, volume)
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=400, detail="试听参数无效")
    if not 0.2 <= length_scale <= 3 or not 0 <= noise_scale <= 2 or not 0 <= noise_w_scale <= 2 or not 0.1 <= volume <= 3:
        raise HTTPException(status_code=400, detail="试听参数超出范围")
    try:
        artifact, model_path = resolve_piper_artifact(model_id, "onnx")
        manifest = _read_student_manifest(model_path)
        config_path = _find_voice_config(model_path) if manifest is None else None
        if manifest is None and config_path is None:
            raise ValueError("模型缺少运行配置")
        if artifact.get("engine") == "melo_onnx_native":
            if student_training_running() or _other_training_running():
                raise ValueError("训练运行时不能试听原生 Melo ONNX")
        settings = {
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w_scale": noise_w_scale,
            "volume": volume,
            "speaker_id": None,
        }
        cache_settings = settings
        if artifact.get("engine") == "melo_onnx_native":
            cache_settings = {
                **settings,
                "noise_scale": 0.6 if math.isclose(noise_scale, 0.667) else noise_scale,
                "noise_w_scale": 0.8 if math.isclose(noise_w_scale, 0.8) else noise_w_scale,
                "sdp_ratio": float(manifest.get("sdp_ratio", 0.2)),
            }
        cache_key = json.dumps(
            {
                "model": artifact["id"],
                "mtime": model_path.stat().st_mtime_ns,
                "manifest_mtime": (model_path.parent / STUDENT_MANIFEST_NAME).stat().st_mtime_ns if manifest else None,
                "text": clean_text,
                **cache_settings,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        digest = hashlib.sha256(cache_key.encode("ascii")).hexdigest()[:8]
        output_path = PIPER_OUTPUT_ROOT / f"piper-{datetime.now():%y%m%d}-{digest}.wav"
        cached = False
        if output_path.is_file():
            try:
                cached = sf.info(output_path).duration > 0
            except (OSError, RuntimeError):
                output_path.unlink(missing_ok=True)
        if not cached:
            try:
                if artifact.get("engine") == "sherpa_onnx":
                    if manifest is None:
                        raise ValueError("双语模型缺少学生模型清单")
                    await asyncio.to_thread(
                        sherpa_voice_runtime.synthesize,
                        model_path,
                        manifest,
                        clean_text,
                        output_path,
                        settings,
                    )
                elif artifact.get("engine") == "melo_onnx_native":
                    if manifest is None:
                        raise ValueError("原生 Melo ONNX 缺少模型清单")
                    await asyncio.to_thread(_release_inference)
                    piper_voice_runtime.release()
                    sherpa_voice_runtime.release()
                    await asyncio.to_thread(
                        melo_native_voice_runtime.synthesize,
                        model_path,
                        manifest,
                        clean_text,
                        output_path,
                        cache_settings,
                    )
                else:
                    await asyncio.to_thread(
                        piper_voice_runtime.synthesize,
                        model_path,
                        config_path,
                        clean_text,
                        output_path,
                        settings,
                    )
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
        info = sf.info(output_path)
        return {
            "filename": output_path.name,
            "audio_url": f"/api/piper/audio/{output_path.name}",
            "duration": round(float(info.duration), 3),
            "cached": cached,
            "model": artifact["name"],
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/piper/audio/{filename}")
def piper_audio(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = PIPER_OUTPUT_ROOT / safe_name
    if filename != safe_name or path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(status_code=404, detail="Piper 试听音频不存在")
    return FileResponse(path, media_type="audio/wav", filename=safe_name)


@router.post("/api/melo/preview")
async def melo_checkpoint_preview(
    checkpoint_id: str = Form(...),
    text: str = Form(...),
    speed: float = Form(1.0),
    volume: float = Form(1.0),
) -> dict:
    clean_text = text.strip()
    if not clean_text or len(clean_text) > 500:
        raise HTTPException(status_code=400, detail="试听文本长度必须为 1-500 个字符")
    if not math.isfinite(speed) or not math.isfinite(volume) or not 0.3 <= speed <= 3 or not 0.1 <= volume <= 3:
        raise HTTPException(status_code=400, detail="MeloTTS 试听语速超出范围")
    if student_training_running():
        raise HTTPException(status_code=409, detail="训练运行时不能试听 MeloTTS 检查点")
    try:
        artifact, checkpoint_path = resolve_piper_artifact(checkpoint_id, "melo_checkpoint")
        config_path = _find_melo_config(checkpoint_path)
        if config_path is None:
            raise ValueError("MeloTTS 检查点缺少完整 config.json")
        digest = hashlib.sha256(
            f"{checkpoint_path}:{checkpoint_path.stat().st_mtime_ns}:{clean_text}:{speed}:{volume}".encode("utf-8")
        ).hexdigest()[:8]
        output_path = PIPER_OUTPUT_ROOT / f"melo-{datetime.now():%y%m%d}-{digest}.wav"
        cached = output_path.is_file()
        if not cached:
            await asyncio.to_thread(_release_inference)
            piper_voice_runtime.release()
            sherpa_voice_runtime.release()
            melo_native_voice_runtime.release()
            await asyncio.to_thread(
                preview_melo_checkpoint,
                checkpoint_path,
                config_path,
                clean_text,
                output_path,
                speed,
            )
            if not math.isclose(volume, 1.0):
                samples, sample_rate = sf.read(output_path, dtype="float32")
                sf.write(output_path, np.clip(samples * volume, -1.0, 1.0), sample_rate, subtype="PCM_16")
        info = sf.info(output_path)
        return {
            "filename": output_path.name,
            "audio_url": f"/api/piper/audio/{output_path.name}",
            "duration": round(float(info.duration), 3),
            "cached": cached,
            "model": artifact["name"],
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/piper/train")
async def start_piper_training(
    dataset_dir: str = Form(...),
    output_name: str = Form(""),
    quality: str = Form("x_low"),
    base_checkpoint_id: str = Form(""),
    num_epochs: int = Form(100),
    learning_rate: float = Form(0.0001),
    batch_size: int = Form(1),
    validation_split: float = Form(0.05),
    save_every_epochs: int = Form(5),
    num_workers: int = Form(1),
    trim_silence: bool = Form(True),
) -> dict:
    if piper_training.running:
        raise HTTPException(status_code=409, detail="Piper 训练任务已经在运行")
    if melo_training.running:
        raise HTTPException(status_code=409, detail="MeloTTS 训练正在运行")
    if optimizer_head_training.running:
        raise HTTPException(status_code=409, detail="优化头训练正在运行")
    if _other_training_running():
        raise HTTPException(status_code=409, detail="LoRA 训练正在运行，请先暂停")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="Piper 微调需要 CUDA")
    if quality not in QUALITY_PROFILES:
        raise HTTPException(status_code=400, detail="未知 Piper 质量配置")
    if not 1 <= num_epochs <= 5000 or not 1 <= save_every_epochs <= num_epochs:
        raise HTTPException(status_code=400, detail="训练轮数或保存间隔无效")
    if not 1 <= batch_size <= 16 or not 0 <= num_workers <= 8:
        raise HTTPException(status_code=400, detail="批次大小或数据线程无效")
    if not 1e-6 <= learning_rate <= 0.01 or not 0.01 <= validation_split <= 0.3:
        raise HTTPException(status_code=400, detail="学习率或验证集比例无效")

    try:
        dataset = await asyncio.to_thread(inspect_piper_dataset, dataset_dir)
        if batch_size > dataset["file_count"]:
            raise ValueError("批次大小不能超过训练集条数")
        warmstart_path = None
        if base_checkpoint_id:
            checkpoint, warmstart_path = resolve_piper_artifact(base_checkpoint_id, "checkpoint")
            checkpoint_quality = checkpoint.get("quality")
            if checkpoint_quality and checkpoint_quality != quality:
                raise ValueError(
                    f"检查点规格为 {checkpoint_quality}，不能用于 {quality}；"
                    "请选择相同规格或从头训练"
                )
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_name = safe_piper_job_name(output_name)
    job_dir = PIPER_RUNS_ROOT / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "metadata.csv"
    config_path = job_dir / "train.yaml"
    voice_config_path = job_dir / "voice.json"
    write_piper_manifest(dataset, manifest_path)

    profile = QUALITY_PROFILES[quality]
    config = {
        "seed_everything": 42,
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "16-mixed",
            "max_epochs": num_epochs,
            "check_val_every_n_epoch": save_every_epochs,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 1,
            "default_root_dir": str(job_dir),
            "enable_progress_bar": True,
        },
        "model": {
            "sample_rate": profile["sample_rate"],
            "num_speakers": 1,
            "resblock": "2",
            "resblock_kernel_sizes": [3, 5, 7],
            "resblock_dilation_sizes": [[1, 2], [2, 6], [3, 12]],
            "upsample_rates": [8, 8, 4],
            "upsample_initial_channel": profile["upsample_initial_channel"],
            "upsample_kernel_sizes": [16, 16, 8],
            "inter_channels": profile["inter_channels"],
            "hidden_channels": profile["hidden_channels"],
            "filter_channels": profile["filter_channels"],
            "n_layers": profile["n_layers"],
            "learning_rate": learning_rate,
            "learning_rate_d": max(1e-6, learning_rate / 2),
            "grad_clip": 1.0,
            "mos_metric": None,
            "warmstart_ckpt": str(warmstart_path) if warmstart_path else None,
        },
        "data": {
            "csv_path": str(manifest_path),
            "audio_dir": dataset["directory"],
            "cache_dir": str(job_dir / "cache"),
            "config_path": str(voice_config_path),
            "voice_name": job_name,
            "espeak_voice": "cmn",
            "phoneme_type": "espeak",
            "num_symbols": 256,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "num_test_examples": min(3, max(1, dataset["file_count"] // 20)),
            "num_workers": num_workers,
            "trim_silence": trim_silence,
        },
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_name": job_name,
                "quality": quality,
                "dataset": dataset["directory"],
                "base_checkpoint_id": base_checkpoint_id or None,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    await asyncio.to_thread(_release_inference)
    piper_voice_runtime.release()
    sherpa_voice_runtime.release()
    melo_native_voice_runtime.release()
    try:
        piper_training.start(config_path, job_name, job_dir)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Piper 训练启动失败: {exc}") from exc
    return {
        "status": "running",
        "job_name": job_name,
        "quality": quality,
        "dataset": {key: value for key, value in dataset.items() if key != "records"},
    }


@router.post("/api/melo/train")
async def start_melo_training(
    dataset_dir: str = Form(...),
    output_name: str = Form(""),
    base_checkpoint_id: str = Form("official-melotts-chinese"),
    language: str = Form("ZH_MIX_EN"),
    num_epochs: int = Form(50),
    learning_rate: float = Form(0.00005),
    batch_size: int = Form(1),
    validation_split: float = Form(0.05),
    save_every_epochs: int = Form(5),
    num_workers: int = Form(1),
    segment_size: int = Form(8192),
    keep_checkpoints: int = Form(5),
    precision: str = Form("fp32"),
) -> dict:
    if melo_training.running:
        raise HTTPException(status_code=409, detail="MeloTTS 训练任务已经在运行")
    if piper_training.running:
        raise HTTPException(status_code=409, detail="Piper 训练正在运行")
    if optimizer_head_training.running:
        raise HTTPException(status_code=409, detail="优化头训练正在运行")
    if _other_training_running():
        raise HTTPException(status_code=409, detail="LoRA 训练正在运行，请先暂停")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="MeloTTS 微调需要 CUDA")
    if not MELO_SOURCE_ROOT.is_dir():
        raise HTTPException(status_code=500, detail="MeloTTS 训练核心未安装")
    if language not in ("ZH", "ZH_MIX_EN"):
        raise HTTPException(status_code=400, detail="MeloTTS 文本前端无效")
    if precision not in ("fp32", "bf16", "fp16"):
        raise HTTPException(status_code=400, detail="MeloTTS 训练精度无效")
    if not 1 <= num_epochs <= 1000 or not 1 <= save_every_epochs <= num_epochs:
        raise HTTPException(status_code=400, detail="训练轮数或保存间隔无效")
    if not 1 <= batch_size <= 4 or not 0 <= num_workers <= 4:
        raise HTTPException(status_code=400, detail="批次大小或数据线程无效")
    if not 1e-6 <= learning_rate <= 0.001 or not 0.01 <= validation_split <= 0.3:
        raise HTTPException(status_code=400, detail="学习率或验证集比例无效")
    if segment_size not in (4096, 8192, 16384) or not 1 <= keep_checkpoints <= 20:
        raise HTTPException(status_code=400, detail="音频片段或检查点保留数量无效")

    try:
        dataset = await asyncio.to_thread(inspect_piper_dataset, dataset_dir)
        if batch_size > dataset["file_count"] - 1:
            raise ValueError("批次大小不能超过扣除验证集后的训练条数")
        if base_checkpoint_id in ("", "official-melotts-chinese"):
            if not melo_base_status()["installed"]:
                raise ValueError("官方 MeloTTS-Chinese 基座尚未下载")
            base_checkpoint = MELO_BASE_CHECKPOINT
            base_label = "official-melotts-chinese"
        else:
            _, base_checkpoint = resolve_piper_artifact(base_checkpoint_id, "melo_checkpoint")
            base_label = base_checkpoint_id
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested_name = safe_piper_job_name(output_name)
    job_name = requested_name if requested_name.startswith("melo-") else f"melo-{requested_name}"
    job_dir = PIPER_RUNS_ROOT / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    job_config_path = job_dir / "melo-job.json"
    job_config = {
        "job_name": job_name,
        "job_dir": str(job_dir.resolve()),
        "dataset_dir": dataset["directory"],
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_id": base_label,
        "speaker_name": "VOXCPM",
        "language": language,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "save_every_epochs": save_every_epochs,
        "num_workers": num_workers,
        "segment_size": segment_size,
        "keep_checkpoints": keep_checkpoints,
        "precision": precision,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    job_config_path.write_text(json.dumps(job_config, ensure_ascii=False, indent=2), encoding="utf-8")

    await asyncio.to_thread(_release_inference)
    piper_voice_runtime.release()
    sherpa_voice_runtime.release()
    melo_native_voice_runtime.release()
    try:
        melo_training.start(job_config_path, job_name, job_dir)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"MeloTTS 训练启动失败: {exc}") from exc
    return {
        "status": "running",
        "job_name": job_name,
        "quality": f"MeloTTS 44.1kHz · {precision.upper()}",
        "dataset": {key: value for key, value in dataset.items() if key != "records"},
    }


@router.post("/api/piper/stop")
def stop_piper_training() -> dict:
    if melo_training.running:
        melo_training.stop()
        return {"status": "stopping", "engine": "sherpa_onnx"}
    if not piper_training.stop():
        raise HTTPException(status_code=409, detail="没有正在运行的学生模型训练")
    return {"status": "stopping", "engine": "piper"}


@router.post("/api/piper/export/{artifact_id}")
async def export_piper_artifact(artifact_id: str) -> dict:
    if student_training_running():
        raise HTTPException(status_code=409, detail="训练运行时不能导出检查点")
    try:
        artifact, checkpoint_path = resolve_piper_artifact(artifact_id)
        if artifact["kind"] == "melo_checkpoint":
            config_path = _find_melo_config(checkpoint_path)
            if config_path is None:
                raise ValueError("MeloTTS 检查点缺少完整 config.json")
            job_name = checkpoint_path.parent.parent.name
            export_name = safe_piper_job_name(f"{job_name}-{checkpoint_path.stem}-native-int8")
            output_dir = PIPER_MODELS_ROOT / export_name
            await asyncio.to_thread(export_melo_checkpoint, checkpoint_path, output_dir, config_path)
            exported = next(
                item
                for item in list_piper_artifacts()
                if item["kind"] == "onnx" and item["relative_path"] == str((output_dir / "model.int8.onnx").relative_to(PIPER_ROOT))
            )
            return {"artifact": exported, "source": artifact}
        if artifact["kind"] != "checkpoint":
            raise ValueError("只能导出 Piper CKPT 或 MeloTTS PTH")
        config_path = _find_voice_config(checkpoint_path)
        if config_path is None:
            raise ValueError("检查点目录缺少 voice.json/config.json")
        job_name = safe_piper_job_name(checkpoint_path.relative_to(PIPER_RUNS_ROOT).parts[0])
        output_path = PIPER_MODELS_ROOT / job_name / f"{job_name}-{checkpoint_path.stem}.onnx"
        await asyncio.to_thread(export_piper_checkpoint, checkpoint_path, output_path, config_path)
        exported = next(item for item in list_piper_artifacts() if item["kind"] == "onnx" and item["relative_path"] == str(output_path.relative_to(PIPER_ROOT)))
        return {"artifact": exported, "source": artifact}
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/export/artifact/{artifact_id}")
async def export_student_artifact(artifact_id: str, precision: str = Form("int8")) -> dict:
    if student_training_running():
        raise HTTPException(status_code=409, detail="训练运行时不能导出学生模型")
    precision = str(precision).strip().lower()
    try:
        artifact, source_path = resolve_piper_artifact(artifact_id)
        if precision not in artifact.get("export_precisions", []):
            raise ValueError(
                f"{artifact['kind']} {artifact['precision'].upper()} 不支持导出为 {precision.upper()}"
            )
        reused = artifact["kind"] == "onnx" and precision == artifact["precision"]
        if artifact["kind"] == "onnx":
            output_path = await asyncio.to_thread(
                export_existing_onnx_precision,
                artifact,
                source_path,
                precision,
            )
        elif artifact["kind"] == "melo_checkpoint":
            config_path = _find_melo_config(source_path)
            if config_path is None:
                raise ValueError("MeloTTS 检查点缺少完整 config.json")
            job_name = source_path.parent.parent.name
            output_dir = PIPER_MODELS_ROOT / safe_piper_job_name(
                f"{job_name}-{source_path.stem}-native-{precision}"
            )
            output_path = await asyncio.to_thread(
                export_melo_checkpoint,
                source_path,
                output_dir,
                config_path,
                precision,
            )
        elif artifact["kind"] == "checkpoint":
            config_path = _find_voice_config(source_path)
            if config_path is None:
                raise ValueError("Piper 检查点目录缺少 voice.json/config.json")
            job_name = safe_piper_job_name(source_path.relative_to(PIPER_RUNS_ROOT).parts[0])
            output_dir = PIPER_MODELS_ROOT / safe_piper_job_name(
                f"{job_name}-{source_path.stem}-{precision}"
            )
            output_path = output_dir / f"{job_name}-{source_path.stem}.{precision}.onnx"
            await asyncio.to_thread(
                export_piper_checkpoint_precision,
                source_path,
                output_path,
                config_path,
                precision,
            )
        else:
            raise ValueError("该资产类型不能导出 ONNX")

        output_resolved = output_path.resolve()
        exported = next(
            item
            for item in list_piper_artifacts()
            if item["kind"] == "onnx"
            and (PIPER_ROOT / item["relative_path"]).resolve() == output_resolved
        )
        return {
            "artifact": exported,
            "source": artifact,
            "precision": precision,
            "reused": reused,
            "download_url": f"/api/piper/download/{exported['id']}",
        }
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/piper/artifacts/{artifact_id}")
def delete_piper_artifact(artifact_id: str) -> dict:
    if student_training_running():
        raise HTTPException(status_code=409, detail="训练运行时不能删除模型或检查点")
    try:
        artifact, path = resolve_piper_artifact(artifact_id)
        manifest = _read_student_manifest(path) if artifact["kind"] == "onnx" else None
        if manifest is not None:
            model_dir = path.parent.resolve()
            if model_dir.parent != PIPER_MODELS_ROOT.resolve():
                raise ValueError("只能删除模型目录中的学生模型")
            runtime_dir = _sherpa_runtime_dir(path)
            shutil.rmtree(model_dir)
            if runtime_dir.is_dir():
                shutil.rmtree(runtime_dir)
        elif artifact["kind"] == "melo_checkpoint":
            checkpoint_path = path.resolve()
            if PIPER_RUNS_ROOT.resolve() not in checkpoint_path.parents:
                raise ValueError("只能删除训练目录中的 MeloTTS 检查点")
            match = re.fullmatch(r"G_(\d+)\.pth", path.name)
            path.unlink()
            if match:
                for prefix in ("D", "DUR"):
                    path.with_name(f"{prefix}_{match.group(1)}.pth").unlink(missing_ok=True)
        else:
            config_path = _find_voice_config(path) if artifact["kind"] == "onnx" else None
            path.unlink()
            if config_path is not None and config_path.is_file() and len(list(config_path.parent.glob("*.onnx"))) == 0:
                config_path.unlink()
        piper_voice_runtime.release()
        sherpa_voice_runtime.release()
        melo_native_voice_runtime.release()
        return {"deleted": artifact}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/piper/download/{artifact_id}")
def download_piper_artifact(artifact_id: str) -> FileResponse:
    try:
        artifact, path = resolve_piper_artifact(artifact_id)
        if artifact["kind"] in ("checkpoint", "melo_checkpoint"):
            return FileResponse(path, media_type="application/octet-stream", filename=path.name)
        manifest = _read_student_manifest(path)
        if manifest is not None:
            manifest_path = path.parent / STUDENT_MANIFEST_NAME
            bundle_hash = hashlib.sha256(
                f"{path.stat().st_mtime_ns}:{path.stat().st_size}:{manifest_path.stat().st_mtime_ns}".encode("ascii")
            ).hexdigest()[:10]
            bundle_path = PIPER_DOWNLOAD_ROOT / f"{safe_piper_job_name(path.parent.name)}-{bundle_hash}.zip"
            if not bundle_path.is_file():
                with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for relative_name in manifest.get("bundle_files", []):
                        resource = _manifest_resource(path, relative_name)
                        if resource.is_dir():
                            for child in resource.rglob("*"):
                                if child.is_file():
                                    archive.write(child, arcname=child.relative_to(path.parent).as_posix())
                        elif resource.is_file():
                            archive.write(resource, arcname=resource.relative_to(path.parent).as_posix())
                        else:
                            raise ValueError(f"模型下载资源缺失: {relative_name}")
            return FileResponse(
                bundle_path,
                media_type="application/zip",
                filename=f"{path.parent.name}-sherpa-onnx.zip",
            )
        config_path = _find_voice_config(path)
        if config_path is None:
            raise ValueError("ONNX 模型缺少 JSON 配置")
        bundle_hash = hashlib.sha256(
            f"{path.stat().st_mtime_ns}:{path.stat().st_size}:{config_path.stat().st_mtime_ns}".encode("ascii")
        ).hexdigest()[:10]
        bundle_path = PIPER_DOWNLOAD_ROOT / f"{safe_piper_job_name(path.stem)}-{bundle_hash}.zip"
        if not bundle_path.is_file():
            with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.write(path, arcname=f"{path.stem}.onnx")
                archive.write(config_path, arcname=f"{path.stem}.onnx.json")
        return FileResponse(bundle_path, media_type="application/zip", filename=f"{path.stem}-piper.zip")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
