from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


STUDENT_MANIFEST_NAME = "voxcpm-model.json"


def _remove_job_directory(job_dir: Path, name: str) -> bool:
    job_root = job_dir.resolve()
    target = (job_dir / name).resolve()
    if target.parent != job_root:
        raise ValueError(f"Refusing to remove training cache outside job directory: {target}")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    print(f"[Piper Plus] Released training cache: {name}", flush=True)
    return True


def cleanup_training_cache(job_dir: Path, *names: str) -> list[str]:
    removed = []
    for name in names or ("ljspeech-input", "dataset"):
        try:
            if name == "dataset":
                source_config = job_dir / "dataset" / "config.json"
                preserved_config = job_dir / "config.json"
                if source_config.is_file():
                    shutil.copy2(source_config, preserved_config)
            if _remove_job_directory(job_dir, name):
                removed.append(name)
        except OSError as exc:
            print(f"[Piper Plus] Training cache cleanup failed for {name}: {exc}", flush=True)
    return removed


def stage_ljspeech_input(records: list[dict], input_dir: Path) -> None:
    wav_dir = input_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, record in enumerate(records):
        source = Path(record["audio"])
        stem = f"sample-{index:06d}"
        destination = wav_dir / f"{stem}{source.suffix.lower()}"
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        text = str(record["text"]).replace("\n", " ").strip()
        lines.append(f"{stem}|{text}|{text}")
    (input_dir / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path) -> None:
    print("[Piper Plus] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def build_preprocess_command(
    python: str,
    input_dir: Path,
    dataset_dir: Path,
    max_workers: int = 1,
) -> list[str]:
    return [
        python,
        "-m",
        "piper_train.preprocess",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(dataset_dir),
        "--language",
        "ja-en-zh-es-fr-pt",
        "--dataset-format",
        "ljspeech",
        "--sample-rate",
        "22050",
        "--single-speaker",
        "--phoneme-type",
        "multilingual",
        "--max-workers",
        str(max_workers),
    ]


def build_training_command(python: str, dataset_dir: Path, job_dir: Path, job: dict) -> list[str]:
    command = [
        python,
        "-m",
        "piper_train",
        "--dataset-dir",
        str(dataset_dir),
        "--prosody-dim",
        "16",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--precision",
        "32-true",
        "--max_epochs",
        str(job["num_epochs"]),
        "--batch-size",
        str(job["batch_size"]),
        "--checkpoint-epochs",
        str(job["save_every_epochs"]),
        "--base_lr",
        str(job["learning_rate"]),
        "--disable_auto_lr_scaling",
        "--ema-decay",
        "0.9995",
        "--max-phoneme-ids",
        "400",
        "--no-wavlm",
        "--default_root_dir",
        str(job_dir),
        "--validation-split",
        str(job["validation_split"]),
        "--num-test-examples",
        "0",
        "--num-workers",
        str(job.get("num_workers", 1)),
    ]
    if job.get("resume_mode") == "checkpoint":
        command.extend(["--resume_from_checkpoint", str(job["base_checkpoint"])])
    else:
        command.extend(["--resume-from-multispeaker-checkpoint", str(job["base_checkpoint"])])
    return command


def latest_checkpoint(job_dir: Path) -> Path:
    checkpoints = [path for path in job_dir.rglob("*.ckpt") if path.is_file()]
    if not checkpoints:
        raise RuntimeError("Piper Plus 训练完成，但没有生成 CKPT 检查点")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def export_student_model(python: str, job_dir: Path, dataset_dir: Path, job: dict) -> Path:
    checkpoint = latest_checkpoint(job_dir)
    model_dir = Path(job["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "model.fp16.onnx"
    run(
        [python, "-m", "piper_train.export_onnx", str(checkpoint), str(output_path)],
        job_dir,
    )

    source_config = dataset_dir / "config.json"
    if not source_config.is_file():
        raise RuntimeError("Piper Plus 预处理目录缺少 config.json")
    bundled_config = model_dir / "config.json"
    shutil.copy2(source_config, bundled_config)
    bundle_files = [output_path.name, bundled_config.name]
    license_path = Path(job.get("source_root", "")) / "LICENSE.md"
    if license_path.is_file():
        shutil.copy2(license_path, model_dir / license_path.name)
        bundle_files.append(license_path.name)
    bundle_files.append(STUDENT_MANIFEST_NAME)
    manifest = {
        "engine": "piper_plus",
        "engine_label": "Piper Plus",
        "display_name": f"{job['job_name']} · FP16",
        "model": output_path.name,
        "config": bundled_config.name,
        "sample_rate": 22050,
        "quality": "fp16-finetuned",
        "precision": "fp16",
        "language": "ja+en+zh+es+fr+pt",
        "license": "MIT",
        "speaker_id": 0,
        "runtime_requirement": "piper-plus>=1.13.0",
        "source_checkpoint": str(checkpoint),
        "bundle_files": bundle_files,
    }
    (model_dir / STUDENT_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Piper Plus] Exported student model: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and fine-tune a Piper Plus multilingual student")
    parser.add_argument("job_config", type=Path)
    args = parser.parse_args()

    if importlib.util.find_spec("piper_train") is None:
        raise RuntimeError("Piper Plus 训练依赖未安装。请先执行 uv sync 安装 piper-tts-plus。")

    job = json.loads(args.job_config.read_text(encoding="utf-8"))
    job_dir = Path(job["job_dir"])
    input_dir = job_dir / "ljspeech-input"
    dataset_dir = job_dir / "dataset"

    python = sys.executable
    try:
        if not (dataset_dir / "dataset.jsonl").is_file():
            stage_ljspeech_input(job["records"], input_dir)
            run(build_preprocess_command(python, input_dir, dataset_dir, int(job.get("num_workers", 1))), job_dir)
        cleanup_training_cache(job_dir, "ljspeech-input")
        run(build_training_command(python, dataset_dir, job_dir, job), job_dir)
        export_student_model(python, job_dir, dataset_dir, job)
    finally:
        cleanup_training_cache(job_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
