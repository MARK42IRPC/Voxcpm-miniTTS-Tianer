from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def emit(phase: str, current: int, total: int, message: str) -> None:
    print(f"[Head][{phase}] {current}/{total} {message}", flush=True)


def load_job(path: Path) -> dict:
    job = json.loads(path.read_text(encoding="utf-8"))
    required = ("job_name", "job_dir", "dataset_dir", "model_path", "model_engine")
    missing = [key for key in required if not job.get(key)]
    if missing:
        raise ValueError(f"Optimizer-head job is missing: {', '.join(missing)}")
    return job


def source_records(dataset_dir: Path) -> list[tuple[Path, str]]:
    records = []
    for wav_path in sorted(dataset_dir.glob("*.wav")):
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.is_file():
            raise ValueError(f"Missing transcript: {lab_path.name}")
        text = lab_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"Empty transcript: {lab_path.name}")
        records.append((wav_path, text))
    if len(records) < 2:
        raise ValueError("Optimizer-head training requires at least 2 WAV/LAB pairs")
    return records


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def synthesis_key(job: dict, text: str) -> str:
    model_path = Path(job["model_path"])
    value = {
        "model": file_signature(model_path),
        "engine": job["model_engine"],
        "manifest": file_signature(Path(job["manifest_path"])) if job.get("manifest_path") else None,
        "config": file_signature(Path(job["config_path"])) if job.get("config_path") else None,
        "text": text,
        "settings": job.get("synthesis", {}),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()


def synthesize_student(job: dict, text: str, output_path: Path) -> None:
    from voxcpm.web import piper_web

    model_path = Path(job["model_path"])
    settings = {
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w_scale": 0.8,
        "volume": 1.0,
        "speaker_id": None,
        **job.get("synthesis", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if job["model_engine"] == "sherpa_onnx":
        manifest = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
        piper_web.sherpa_voice_runtime.synthesize(model_path, manifest, text, output_path, settings)
    elif job["model_engine"] == "melo_onnx_native":
        manifest = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
        piper_web.melo_native_voice_runtime.synthesize(
            model_path,
            manifest,
            text,
            output_path,
            {**settings, "noise_scale": 0.6, "noise_w_scale": 0.8, "sdp_ratio": manifest.get("sdp_ratio", 0.2)},
        )
    elif job["model_engine"] == "piper":
        piper_web.piper_voice_runtime.synthesize(
            model_path,
            Path(job["config_path"]),
            text,
            output_path,
            settings,
        )
    else:
        raise ValueError(f"Unsupported student ONNX engine: {job['model_engine']}")


def _normalize_alignment_features(features: np.ndarray) -> np.ndarray:
    centered = features - np.mean(features, axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=0, keepdims=True)
    return centered / np.maximum(scale, 1e-5)


def align_spectra(
    student_path: Path,
    clean_path: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    student, _ = librosa.load(student_path, sr=sample_rate, mono=True)
    clean, _ = librosa.load(clean_path, sr=sample_rate, mono=True)
    if student.size < n_fft or clean.size < n_fft:
        raise ValueError(f"Audio is too short for spectral alignment: {clean_path.name}")

    student = np.asarray(student, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    student_rms = float(np.sqrt(np.mean(np.square(student)) + 1e-8))
    clean_rms = float(np.sqrt(np.mean(np.square(clean)) + 1e-8))
    clean = np.clip(clean * min(4.0, student_rms / max(clean_rms, 1e-5)), -1.0, 1.0)

    student_stft = librosa.stft(student, n_fft=n_fft, hop_length=hop_length, window="hann")
    clean_stft = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length, window="hann")
    student_mag = np.abs(student_stft).astype(np.float32)
    clean_mag = np.abs(clean_stft).astype(np.float32)

    alignment_hop = max(hop_length, round(sample_rate * 0.02))
    student_mel = librosa.feature.melspectrogram(
        y=student,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=alignment_hop,
        n_mels=64,
        fmin=40,
        fmax=sample_rate / 2,
        power=2.0,
    )
    clean_mel = librosa.feature.melspectrogram(
        y=clean,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=alignment_hop,
        n_mels=64,
        fmin=40,
        fmax=sample_rate / 2,
        power=2.0,
    )
    student_features = _normalize_alignment_features(librosa.power_to_db(student_mel + 1e-8))
    clean_features = _normalize_alignment_features(librosa.power_to_db(clean_mel + 1e-8))
    _, path = librosa.sequence.dtw(
        X=clean_features,
        Y=student_features,
        metric="euclidean",
        backtrack=True,
    )
    path = path[::-1]
    student_align_frames = student_features.shape[1]
    mapped = [[] for _ in range(student_align_frames)]
    for clean_index, student_index in path:
        if 0 <= student_index < student_align_frames:
            mapped[int(student_index)].append(int(clean_index))
    known_student = np.asarray([index for index, values in enumerate(mapped) if values], dtype=np.float32)
    known_clean = np.asarray([np.median(mapped[index]) for index in known_student.astype(int)], dtype=np.float32)
    if known_student.size < 2:
        raise ValueError(f"Unable to align student and clean spectra: {clean_path.name}")

    student_frames = np.arange(student_mag.shape[1], dtype=np.float32)
    student_alignment_positions = student_frames * hop_length / alignment_hop
    clean_alignment_positions = np.interp(student_alignment_positions, known_student, known_clean)
    clean_frame_positions = np.clip(
        clean_alignment_positions * alignment_hop / hop_length,
        0,
        clean_mag.shape[1] - 1,
    )
    clean_axis = np.arange(clean_mag.shape[1], dtype=np.float32)
    aligned_clean = np.empty_like(student_mag)
    for frequency in range(clean_mag.shape[0]):
        aligned_clean[frequency] = np.interp(clean_frame_positions, clean_axis, clean_mag[frequency])

    input_log = np.log1p(student_mag).astype(np.float32)
    target_log = np.log1p(np.maximum(aligned_clean, 0)).astype(np.float32)
    metadata = {
        "student_seconds": round(student.size / sample_rate, 4),
        "clean_seconds": round(clean.size / sample_rate, 4),
        "student_frames": int(student_mag.shape[1]),
        "clean_frames": int(clean_mag.shape[1]),
        "dtw_cost_frames": int(path.shape[0]),
    }
    return input_log, target_log, metadata


def build_aligned_dataset(job: dict) -> list[Path]:
    job_dir = Path(job["job_dir"])
    student_dir = job_dir / "student-audio"
    aligned_dir = job_dir / "aligned-cache"
    student_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    records = source_records(Path(job["dataset_dir"]))
    sample_rate = int(job["sample_rate"])
    n_fft = int(job["n_fft"])
    hop_length = int(job["hop_length"])
    outputs = []

    emit("prepare", 0, len(records), "preparing aligned training pairs")
    for index, (clean_path, text) in enumerate(records, 1):
        stem = f"pair-{index:06d}"
        student_path = student_dir / f"{stem}.wav"
        student_meta_path = student_dir / f"{stem}.json"
        synth_key = synthesis_key(job, text)
        student_cached = False
        if student_path.is_file() and student_meta_path.is_file():
            try:
                student_cached = json.loads(student_meta_path.read_text(encoding="utf-8")).get("key") == synth_key
            except (OSError, json.JSONDecodeError):
                student_cached = False
        if not student_cached:
            student_path.unlink(missing_ok=True)
            emit("synthesize", index, len(records), clean_path.name)
            synthesize_student(job, text, student_path)
            student_meta_path.write_text(
                json.dumps({"key": synth_key, "text": text}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            emit("synthesize", index, len(records), f"{clean_path.name} (cache)")

        aligned_path = aligned_dir / f"{stem}.npz"
        aligned_meta_path = aligned_dir / f"{stem}.json"
        align_key = hashlib.sha256(
            f"{synth_key}|{file_signature(clean_path)}|{sample_rate}|{n_fft}|{hop_length}".encode("utf-8")
        ).hexdigest()
        aligned_cached = False
        if aligned_path.is_file() and aligned_meta_path.is_file():
            try:
                aligned_cached = json.loads(aligned_meta_path.read_text(encoding="utf-8")).get("key") == align_key
            except (OSError, json.JSONDecodeError):
                aligned_cached = False
        if not aligned_cached:
            emit("align", index, len(records), clean_path.name)
            input_log, target_log, metadata = align_spectra(
                student_path,
                clean_path,
                sample_rate,
                n_fft,
                hop_length,
            )
            np.savez_compressed(
                aligned_path,
                input_log=input_log.astype(np.float16),
                target_log=target_log.astype(np.float16),
            )
            aligned_meta_path.write_text(
                json.dumps(
                    {"key": align_key, "source": str(clean_path), "text": text, **metadata},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            emit("align", index, len(records), f"{clean_path.name} (cache)")
        outputs.append(aligned_path)
    return outputs


class SpectralChunkDataset(Dataset):
    def __init__(self, files: list[Path], chunk_frames: int) -> None:
        self.files = files
        self.chunk_frames = chunk_frames
        self.index: list[tuple[int, int]] = []
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        for file_index, path in enumerate(files):
            with np.load(path) as values:
                frame_count = int(values["input_log"].shape[1])
            starts = list(range(0, max(1, frame_count - chunk_frames + 1), chunk_frames))
            final_start = max(0, frame_count - chunk_frames)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            self.index.extend((file_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, file_index: int) -> tuple[np.ndarray, np.ndarray]:
        if file_index in self.cache:
            values = self.cache.pop(file_index)
            self.cache[file_index] = values
            return values
        with np.load(self.files[file_index]) as archive:
            values = (
                archive["input_log"].astype(np.float32),
                archive["target_log"].astype(np.float32),
            )
        self.cache[file_index] = values
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return values

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_index, start = self.index[index]
        input_log, target_log = self._load(file_index)
        end = start + self.chunk_frames
        x = input_log[:, start:end]
        y = target_log[:, start:end]
        if x.shape[1] < self.chunk_frames:
            padding = self.chunk_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, padding)), mode="edge")
            y = np.pad(y, ((0, 0), (0, padding)), mode="edge")
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride_frequency: int = 1) -> None:
        super().__init__()
        groups = max(1, min(8, output_channels // 8))
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=(stride_frequency, 1), padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=(1, dilation),
            dilation=(1, dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm(self.pointwise(self.depthwise(value))))
        return value + residual


class TinyCrnHybrid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(1, 32, 2)
        self.encoder2 = ConvBlock(32, 64, 2)
        self.encoder3 = ConvBlock(64, 128, 2)
        self.encoder4 = ConvBlock(128, 192, 2)
        self.temporal = nn.Sequential(
            *(TemporalResidualBlock(192, dilation) for dilation in (1, 2, 4, 8, 4, 2))
        )
        self.decoder3 = ConvBlock(192 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)
        self.output = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _join(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="nearest")
        return torch.cat((value, skip), dim=1)

    def forward(self, input_log_magnitude: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(input_log_magnitude)
        second = self.encoder2(first)
        third = self.encoder3(second)
        value = self.temporal(self.encoder4(third))
        value = self.decoder3(self._join(value, third))
        value = self.decoder2(self._join(value, second))
        value = self.decoder1(self._join(value, first))
        value = F.interpolate(value, size=input_log_magnitude.shape[-2:], mode="nearest")
        controls = self.output(value)
        mask = 0.5 + torch.sigmoid(controls[:, :1])
        residual = 0.35 * torch.tanh(controls[:, 1:2])
        return torch.clamp(input_log_magnitude * mask + residual, min=0.0)


def spectral_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    frequency_weight = 1.0 + 2.0 * torch.linspace(
        0,
        1,
        predicted.shape[-2],
        device=predicted.device,
        dtype=predicted.dtype,
    ).square()[None, None, :, None]
    log_loss = torch.mean(torch.abs(predicted - target) * frequency_weight)
    predicted_linear = torch.expm1(torch.clamp(predicted, max=8.0))
    target_linear = torch.expm1(torch.clamp(target, max=8.0))
    scale = torch.mean(target_linear, dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    linear_loss = torch.mean(torch.abs(predicted_linear - target_linear) / scale)
    temporal_loss = torch.mean(
        torch.abs((predicted[..., 1:] - predicted[..., :-1]) - (target[..., 1:] - target[..., :-1]))
    )
    return log_loss + 0.1 * linear_loss + 0.15 * temporal_loss


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for input_log, target_log in loader:
            input_log = input_log.to(device)
            target_log = target_log.to(device)
            loss = spectral_loss(model(input_log), target_log)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss became non-finite")
            total += float(loss.item())
            count += 1
    return total / max(1, count)


def train_model(job: dict, aligned_files: list[Path]) -> tuple[Path, list[Path]]:
    random.Random(52).shuffle(aligned_files)
    validation_count = max(1, min(len(aligned_files) - 1, round(len(aligned_files) * job["validation_split"])))
    validation_files = aligned_files[:validation_count]
    training_files = aligned_files[validation_count:]
    train_dataset = SpectralChunkDataset(training_files, int(job["chunk_frames"]))
    validation_dataset = SpectralChunkDataset(validation_files, int(job["chunk_frames"]))
    batch_size = min(int(job["batch_size"]), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCrnHybrid().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[Head] model parameters: {parameter_count} ({parameter_count * 4 / 1024**2:.2f} MiB FP32)", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(job["learning_rate"]), weight_decay=1e-4)
    precision = job.get("training_precision", "fp32")
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_autocast = device.type == "cuda" and precision in ("fp16", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    best_loss = math.inf
    job_dir = Path(job["job_dir"])
    checkpoint_path = job_dir / "best-head.pt"
    total_steps = int(job["epochs"]) * len(train_loader)
    global_step = 0

    for epoch in range(1, int(job["epochs"]) + 1):
        model.train()
        for input_log, target_log in train_loader:
            global_step += 1
            input_log = input_log.to(device, non_blocking=True)
            target_log = target_log.to(device, non_blocking=True)
            if random.random() < 0.1:
                identity_count = max(1, input_log.shape[0] // 4)
                input_log = input_log.clone()
                input_log[:identity_count] = target_log[:identity_count]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss = spectral_loss(model(input_log), target_log)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Training loss became non-finite at step {global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Gradient norm became non-finite at step {global_step}")
            scaler.step(optimizer)
            scaler.update()
            emit("train", global_step, total_steps, f"epoch={epoch} loss={float(loss.item()):.6f}")

        validation_loss = evaluate(model, validation_loader, device)
        emit("validate", epoch, int(job["epochs"]), f"loss={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "validation_loss": best_loss,
                    "epoch": epoch,
                    "architecture": job["architecture"],
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
            print(f"[Head] saved best checkpoint at epoch {epoch}", flush=True)

    return checkpoint_path, training_files


def sort_onnx_graph(graph) -> None:
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute.GRAPH:
                sort_onnx_graph(attribute.g)
            elif attribute.type == attribute.GRAPHS:
                for child in attribute.graphs:
                    sort_onnx_graph(child)
    remaining = list(graph.node)
    produced = {name for node in remaining for name in node.output if name}
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(name for node in remaining for name in node.input if name and name not in produced)
    ordered = []
    while remaining:
        ready = [node for node in remaining if all(not name or name in available for name in node.input)]
        if not ready:
            raise RuntimeError("FP16 optimizer-head conversion produced an invalid graph topology")
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
            remaining.remove(node)
    del graph.node[:]
    graph.node.extend(ordered)


class HeadCalibrationReader:
    def __init__(self, files: list[Path], chunk_frames: int, limit: int = 32) -> None:
        self.samples = []
        for path in files:
            with np.load(path) as archive:
                value = archive["input_log"].astype(np.float32)
            for start in range(0, value.shape[1], chunk_frames):
                chunk = value[:, start : start + chunk_frames]
                if chunk.shape[1] < chunk_frames:
                    chunk = np.pad(chunk, ((0, 0), (0, chunk_frames - chunk.shape[1])), mode="edge")
                self.samples.append({"log_magnitude": chunk[None, None]})
                if len(self.samples) >= limit:
                    break
            if len(self.samples) >= limit:
                break
        self.iterator = iter(self.samples)

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self) -> None:
        self.iterator = iter(self.samples)


def export_models(job: dict, checkpoint_path: Path, calibration_files: list[Path]) -> Path:
    import onnx

    job_dir = Path(job["job_dir"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TinyCrnHybrid().eval()
    model.load_state_dict(checkpoint["model"])
    frequency_bins = int(job["n_fft"]) // 2 + 1
    dummy = torch.zeros(1, 1, frequency_bins, int(job["chunk_frames"]), dtype=torch.float32)
    fp32_path = job_dir / "optimizer-head.fp32.onnx"
    emit("export", 1, 3, "exporting FP32 ONNX")
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["log_magnitude"],
        output_names=["enhanced_log_magnitude"],
        dynamic_axes={"log_magnitude": {3: "frames"}, "enhanced_log_magnitude": {3: "frames"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(fp32_path))

    precision = job["export_precision"]
    output_path = fp32_path
    if precision == "fp16":
        from onnxruntime.transformers.float16 import convert_float_to_float16

        emit("export", 2, 3, "converting FP16 ONNX")
        converted = convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
        sort_onnx_graph(converted.graph)
        onnx.checker.check_model(converted)
        output_path = job_dir / "optimizer-head.fp16.onnx"
        onnx.save(converted, output_path)
    elif precision == "int8":
        import onnxruntime as ort
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        emit("export", 2, 3, "calibrating INT8 ONNX")
        temporary_root = Path(
            os.environ.get("VOXCPM_ONNX_TEMP_DIR", r"C:\tmp\voxcpm-onnx" if os.name == "nt" else "/tmp/voxcpm-onnx")
        )
        runtime_dir = temporary_root / f"head-{hashlib.sha256(str(job_dir).encode('utf-8')).hexdigest()[:12]}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_fp32 = runtime_dir / "model.fp32.onnx"
        runtime_int8 = runtime_dir / "model.int8.onnx"
        shutil.copy2(fp32_path, runtime_fp32)
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(temporary_root)
        try:
            quantize_static(
                str(runtime_fp32),
                str(runtime_int8),
                HeadCalibrationReader(calibration_files, int(job["chunk_frames"])),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["Conv"],
            )
            ort.InferenceSession(str(runtime_int8), providers=["CPUExecutionProvider"])
            output_path = job_dir / "optimizer-head.int8.onnx"
            shutil.copy2(runtime_int8, output_path)
        finally:
            tempfile.tempdir = previous_tempdir
            shutil.rmtree(runtime_dir, ignore_errors=True)
    emit("export", 3, 3, f"ready: {output_path.name}")

    manifest = {
        "format": "voxcpm-optimizer-head-v1",
        "architecture": job["architecture"],
        "display_name": job["job_name"],
        "model": output_path.name,
        "fp32_model": fp32_path.name,
        "precision": precision,
        "parameter_count": checkpoint["parameter_count"],
        "size_mb": round(output_path.stat().st_size / 1024**2, 3),
        "sample_rate": job["sample_rate"],
        "n_fft": job["n_fft"],
        "hop_length": job["hop_length"],
        "window": "hann",
        "input": {"name": "log_magnitude", "formula": "log1p(abs(STFT(waveform)))"},
        "output": {"name": "enhanced_log_magnitude", "formula": "expm1(output) with source phase"},
        "source_model": job["source_artifact"],
        "dataset_dir": job["dataset_dir"],
        "validation_loss": checkpoint["validation_loss"],
        "best_epoch": checkpoint["epoch"],
    }
    (job_dir / "optimizer-head.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    job = load_job(args.job_config.resolve())
    aligned_files = build_aligned_dataset(job)
    if args.prepare_only:
        emit("complete", len(aligned_files), len(aligned_files), "aligned cache ready")
        return
    checkpoint_path, calibration_files = train_model(job, aligned_files)
    output_path = export_models(job, checkpoint_path, calibration_files)
    emit("complete", 1, 1, str(output_path))


if __name__ == "__main__":
    main()
