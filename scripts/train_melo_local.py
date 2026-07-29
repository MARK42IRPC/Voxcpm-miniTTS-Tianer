from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
MELO_ROOT = ROOT / "third_party" / "MeloTTS"
MELO_PACKAGE = MELO_ROOT / "melo"


def cleanup_training_cache(job_dir: Path) -> list[str]:
    job_root = job_dir.resolve()
    removed = []
    audio_dir = (job_dir / "audio").resolve()
    if audio_dir.parent != job_root:
        raise ValueError(f"Refusing to remove training cache outside job directory: {audio_dir}")
    try:
        if audio_dir.is_dir():
            shutil.rmtree(audio_dir)
            removed.append("audio")
            print("[Melo] Released training cache: audio", flush=True)
        event_count = 0
        for event_path in job_dir.rglob("events.out.tfevents.*"):
            if not event_path.is_file():
                continue
            resolved = event_path.resolve()
            if job_root not in resolved.parents:
                raise ValueError(f"Refusing to remove TensorBoard event outside job directory: {resolved}")
            event_path.unlink()
            event_count += 1
        if event_count:
            removed.append(f"tensorboard-events:{event_count}")
            print(f"[Melo] Released TensorBoard event files: {event_count}", flush=True)
    except OSError as exc:
        print(f"[Melo] Training cache cleanup failed: {exc}", flush=True)
    return removed


def _load_job(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("job_name", "job_dir", "dataset_dir", "base_checkpoint")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Melo job config is missing: {', '.join(missing)}")
    return data


def _source_records(dataset_dir: Path, filtered_manifest: Path | None = None) -> list[tuple[Path, str]]:
    records = []
    if filtered_manifest is not None:
        if not filtered_manifest.is_file():
            raise ValueError(f"Filtered dataset manifest not found: {filtered_manifest}")
        dataset_root = dataset_dir.resolve()
        for line_number, line in enumerate(filtered_manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            wav_path = Path(str(record.get("audio", ""))).resolve()
            text = str(record.get("text", "")).strip()
            if wav_path.parent != dataset_root or not wav_path.is_file() or not text:
                raise ValueError(f"Invalid filtered dataset record at line {line_number}")
            records.append((wav_path, text))
    else:
        for wav_path in sorted(dataset_dir.glob("*.wav")):
            lab_path = wav_path.with_suffix(".lab")
            if not lab_path.is_file():
                raise ValueError(f"Missing transcript: {lab_path.name}")
            text = lab_path.read_text(encoding="utf-8-sig").strip()
            if not text:
                raise ValueError(f"Empty transcript: {lab_path.name}")
            records.append((wav_path, text))
    if len(records) < 2:
        raise ValueError("MeloTTS training requires at least 2 WAV/LAB pairs")
    return records


def _cache_key(source: Path, text: str, language: str) -> str:
    stat = source.stat()
    value = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{language}|{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def preprocess(job_path: Path) -> None:
    job = _load_job(job_path)
    sys.path.insert(0, str(MELO_ROOT))
    from melo.text.cleaner import clean_text_bert

    job_dir = Path(job["job_dir"])
    dataset_dir = Path(job["dataset_dir"])
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_value = str(job.get("filtered_manifest", "")).strip()
    records = _source_records(dataset_dir, Path(manifest_value) if manifest_value else None)
    language = job.get("language", "ZH")
    speaker_name = job.get("speaker_name", "VOXCPM")
    sampling_rate = 44100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cleaned_records: list[str] = []
    cache_hits = 0

    print(f"[Melo] preprocessing {len(records)} pairs on {device}", flush=True)
    for index, (source, text) in enumerate(records, 1):
        stem = f"melo-{index:06d}"
        target = audio_dir / f"{stem}.wav"
        bert_path = audio_dir / f"{stem}.bert.pt"
        record_path = audio_dir / f"{stem}.json"
        key = _cache_key(source, text, language)
        cached = False
        if target.is_file() and bert_path.is_file() and record_path.is_file():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                cached = record.get("cache_key") == key and record.get("cleaned_line")
            except (OSError, json.JSONDecodeError):
                cached = False

        if cached:
            cache_hits += 1
            cleaned_line = record["cleaned_line"]
        else:
            samples, _ = librosa.load(source, sr=sampling_rate, mono=True)
            sf.write(target, samples, sampling_rate, subtype="PCM_16")
            norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)
            if not phones or len(phones) != len(tones) or len(phones) != sum(word2ph):
                raise ValueError(f"Invalid phoneme alignment: {source.name}")
            torch.save(bert.cpu(), bert_path)
            cleaned_line = "|".join(
                (
                    str(target.resolve()),
                    speaker_name,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join(str(value) for value in tones),
                    " ".join(str(value) for value in word2ph),
                )
            )
            record_path.write_text(
                json.dumps({"cache_key": key, "cleaned_line": cleaned_line}, ensure_ascii=False),
                encoding="utf-8",
            )
        cleaned_records.append(cleaned_line)
        state = "cache" if cached else "built"
        print(f"[Melo] preprocess {index}/{len(records)} {source.name} ({state})", flush=True)

    random.Random(52).shuffle(cleaned_records)
    validation_count = max(1, min(len(cleaned_records) - 1, round(len(cleaned_records) * job["validation_split"])))
    validation_records = cleaned_records[:validation_count]
    training_records = cleaned_records[validation_count:]
    train_path = job_dir / "train.list"
    val_path = job_dir / "val.list"
    train_path.write_text("\n".join(training_records) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(validation_records) + "\n", encoding="utf-8")

    template = json.loads((MELO_PACKAGE / "configs" / "config.json").read_text(encoding="utf-8"))
    base_config_path = Path(job["base_checkpoint"]).with_name("config.json")
    if not base_config_path.is_file():
        candidate = Path(job["base_checkpoint"]).parent.parent / "config.json"
        base_config_path = candidate if candidate.is_file() else base_config_path
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    template["model"].update(base_config["model"])
    template["data"].update(base_config["data"])
    steps_per_epoch = max(1, math.ceil(len(training_records) / job["batch_size"]))
    precision = job.get("precision", "fp32")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unsupported MeloTTS training precision: {precision}")
    template["train"].update(
        {
            "log_interval": 1,
            "eval_interval": max(1, steps_per_epoch * job["save_every_epochs"]),
            "epochs": job["num_epochs"],
            "learning_rate": job["learning_rate"],
            "batch_size": job["batch_size"],
            "fp16_run": precision != "fp32",
            "precision": precision,
            "segment_size": job["segment_size"],
            "num_workers": job["num_workers"],
            "keep_ckpts": job.get("keep_checkpoints", 5),
            "skip_optimizer": False,
        }
    )
    template["data"].update(
        {
            "training_files": str(train_path.resolve()),
            "validation_files": str(val_path.resolve()),
            "sampling_rate": sampling_rate,
            "n_speakers": 256,
            "spk2id": {speaker_name: 1},
            "symbols": base_config["symbols"],
            "cleaned_text": True,
            "max_text_len": 500,
        }
    )
    template["num_languages"] = base_config["num_languages"]
    template["num_tones"] = base_config["num_tones"]
    template["symbols"] = base_config["symbols"]
    train_config = job_dir / "config.json"
    train_config.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Melo] preprocessing complete: {len(training_records)} train, "
        f"{len(validation_records)} validation, {cache_hits} cached",
        flush=True,
    )
    print(f"[Melo] training precision: {precision.upper()}", flush=True)
    del cleaned_records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(job_path: Path) -> None:
    job = _load_job(job_path)
    job_dir = Path(job["job_dir"])
    checkpoint_dir = job_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(MELO_ROOT), env.get("PYTHONPATH", ""))))
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["USE_LIBUV"] = "0"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    command = [
        sys.executable,
        "train.py",
        "--config",
        str((job_dir / "config.json").resolve()),
        "--model",
        job["job_name"],
        "--model-dir",
        str(checkpoint_dir.resolve()),
        "--pretrain_G",
        str(Path(job["base_checkpoint"]).resolve()),
    ]
    print(f"[Melo] starting GPU training: {job['job_name']}", flush=True)
    print(f"[Melo] checkpoints: {checkpoint_dir}", flush=True)
    completed = subprocess.run(command, cwd=MELO_PACKAGE, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_config", type=Path)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    job_path = args.job_config.resolve()
    if args.preprocess_only:
        preprocess(job_path)
        return

    job = _load_job(job_path)
    job_dir = Path(job["job_dir"])
    try:
        preprocess_command = [sys.executable, "-X", "utf8", __file__, str(job_path), "--preprocess-only"]
        subprocess.run(preprocess_command, cwd=ROOT, check=True)
        train(job_path)
    finally:
        cleanup_training_cache(job_dir)


if __name__ == "__main__":
    main()
