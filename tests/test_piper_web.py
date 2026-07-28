import json
import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import piper_web


def test_safe_piper_job_name():
    assert piper_web.safe_piper_job_name(" 爱弥斯:test/01 ") == "爱弥斯_test_01"


def test_voice_summary_infers_quality_from_checkpoint_path(tmp_path):
    config_path = tmp_path / "pretrained-zh_CN-huayan-medium" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"audio": {"sample_rate": 22050}, "espeak": {"voice": "cmn"}}),
        encoding="utf-8",
    )

    summary = piper_web._read_voice_summary(config_path, config_path.parent / "voice.ckpt")

    assert summary == {"sample_rate": 22050, "quality": "medium", "language": "cmn"}


def test_inspect_piper_dataset_requires_complete_wav_lab_pairs(tmp_path):
    for index in range(2):
        sf.write(tmp_path / f"sample-{index}.wav", np.zeros(1600, dtype=np.float32), 16000)
        (tmp_path / f"sample-{index}.lab").write_text(f"测试文本 {index}", encoding="utf-8")

    dataset = piper_web.inspect_piper_dataset(str(tmp_path))

    assert dataset["file_count"] == 2
    assert dataset["sample_rates"] == {16000: 2}
    assert dataset["records"][0]["text"] == "测试文本 0"

    (tmp_path / "sample-1.lab").unlink()
    with pytest.raises(ValueError, match="缺少 1 个同名 .lab"):
        piper_web.inspect_piper_dataset(str(tmp_path))


def test_piper_model_bundle_download_and_restricted_delete(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    downloads = tmp_path / "downloads"
    outputs = tmp_path / "outputs"
    for directory in (models, runs, downloads, outputs):
        directory.mkdir()
    model_dir = models / "test-voice"
    model_dir.mkdir()
    model_path = model_dir / "test-voice.onnx"
    config_path = model_dir / "test-voice.onnx.json"
    model_path.write_bytes(b"fake-onnx")
    config_path.write_text(
        json.dumps({"audio": {"sample_rate": 16000, "quality": "x_low"}, "language": {"code": "zh_CN"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "PIPER_DOWNLOAD_ROOT", downloads)
    monkeypatch.setattr(piper_web, "PIPER_OUTPUT_ROOT", outputs)

    app = FastAPI()
    app.include_router(piper_web.router)
    client = TestClient(app)
    artifact = piper_web.list_piper_artifacts()[0]

    assert artifact["previewable"] is True
    assert artifact["engine"] == "piper"
    assert artifact["sample_rate"] == 16000
    download = client.get(f"/api/piper/download/{artifact['id']}")
    assert download.status_code == 200
    bundle = next(downloads.glob("*.zip"))
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"test-voice.onnx", "test-voice.onnx.json"}

    invalid = client.delete("/api/piper/artifacts/not-a-real-id")
    assert invalid.status_code == 400
    assert model_path.is_file()

    deleted = client.delete(f"/api/piper/artifacts/{artifact['id']}")
    assert deleted.status_code == 200
    assert not model_path.exists()
    assert not config_path.exists()


def test_sherpa_student_manifest_and_bundle(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    downloads = tmp_path / "downloads"
    outputs = tmp_path / "outputs"
    for directory in (models, runs, downloads, outputs):
        directory.mkdir()
    model_dir = models / "bilingual"
    (model_dir / "dict").mkdir(parents=True)
    model_path = model_dir / "model.int8.onnx"
    model_path.write_bytes(b"fake-sherpa-onnx")
    (model_dir / "tokens.txt").write_text("_ 0", encoding="utf-8")
    (model_dir / "lexicon.txt").write_text("hello h e l o", encoding="utf-8")
    (model_dir / "dict" / "jieba.dict.utf8").write_text("test", encoding="utf-8")
    (model_dir / piper_web.STUDENT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "engine": "sherpa_onnx",
                "engine_label": "MeloTTS",
                "display_name": "中英双语 INT8",
                "model": model_path.name,
                "tokens": "tokens.txt",
                "lexicon": "lexicon.txt",
                "dict_dir": "dict",
                "sample_rate": 44100,
                "quality": "int8",
                "language": "zh_CN+en_US",
                "bundle_files": [model_path.name, "tokens.txt", "lexicon.txt", "dict", piper_web.STUDENT_MANIFEST_NAME],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "PIPER_DOWNLOAD_ROOT", downloads)
    monkeypatch.setattr(piper_web, "PIPER_OUTPUT_ROOT", outputs)

    app = FastAPI()
    app.include_router(piper_web.router)
    client = TestClient(app)
    artifact = piper_web.list_piper_artifacts()[0]

    assert artifact["engine"] == "sherpa_onnx"
    assert artifact["name"] == "中英双语 INT8"
    assert artifact["language"] == "zh_CN+en_US"
    download = client.get(f"/api/piper/download/{artifact['id']}")
    assert download.status_code == 200
    with zipfile.ZipFile(next(downloads.glob("*.zip"))) as archive:
        assert set(archive.namelist()) == {
            "model.int8.onnx",
            "tokens.txt",
            "lexicon.txt",
            "dict/jieba.dict.utf8",
            piper_web.STUDENT_MANIFEST_NAME,
        }

    with pytest.raises(ValueError, match="越界路径"):
        piper_web._manifest_resource(model_path, "../outside.txt")


def test_melo_checkpoint_is_classified_separately(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    downloads = tmp_path / "downloads"
    outputs = tmp_path / "outputs"
    checkpoint_dir = runs / "melo-test" / "checkpoints"
    for directory in (models, checkpoint_dir, downloads, outputs):
        directory.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "G_42.pth"
    checkpoint_path.write_bytes(b"melo-generator")
    (checkpoint_dir / "D_42.pth").write_bytes(b"melo-discriminator")
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"train": {}, "model": {}, "data": {}, "symbols": ["_"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "PIPER_DOWNLOAD_ROOT", downloads)
    monkeypatch.setattr(piper_web, "PIPER_OUTPUT_ROOT", outputs)

    artifacts = piper_web.list_piper_artifacts()

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["kind"] == "melo_checkpoint"
    assert artifact["engine"] == "sherpa_onnx"
    assert artifact["engine_label"] == "MeloTTS"
    assert artifact["previewable"] is True
    assert artifact["quality"] == "MeloTTS FP32"
    with pytest.raises(ValueError, match="需要 checkpoint"):
        piper_web.resolve_piper_artifact(artifact["id"], "checkpoint")


def test_melo_base_status_requires_checkpoint_and_config(monkeypatch, tmp_path):
    base_dir = tmp_path / "MeloTTS-Chinese"
    base_dir.mkdir()
    checkpoint = base_dir / "checkpoint.pth"
    monkeypatch.setattr(piper_web, "MELO_BASE_DIR", base_dir)
    monkeypatch.setattr(piper_web, "MELO_BASE_CHECKPOINT", checkpoint)

    assert piper_web.melo_base_status()["installed"] is False
    checkpoint.write_bytes(b"checkpoint")
    assert piper_web.melo_base_status()["installed"] is False
    (base_dir / "config.json").write_text("{}", encoding="utf-8")

    status = piper_web.melo_base_status()
    assert status["installed"] is True
    assert status["checkpoint"] == str(checkpoint)
    assert status["language"] == "中文 + English"


def test_melo_training_persists_precision_and_rejects_unknown_value(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    runs = tmp_path / "runs"
    source = tmp_path / "MeloTTS" / "melo"
    base = tmp_path / "base" / "checkpoint.pth"
    for directory in (dataset, runs, source, base.parent):
        directory.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"checkpoint")
    for index in range(2):
        sf.write(dataset / f"sample-{index}.wav", np.zeros(1600, dtype=np.float32), 16000)
        (dataset / f"sample-{index}.lab").write_text(f"测试文本 {index}", encoding="utf-8")

    started = {}
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "MELO_SOURCE_ROOT", source)
    monkeypatch.setattr(piper_web, "MELO_BASE_CHECKPOINT", base)
    monkeypatch.setattr(piper_web, "melo_base_status", lambda: {"installed": True})
    monkeypatch.setattr(piper_web.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(piper_web, "_release_inference", lambda: None)
    monkeypatch.setattr(piper_web.piper_voice_runtime, "release", lambda: None)
    monkeypatch.setattr(piper_web.sherpa_voice_runtime, "release", lambda: None)
    monkeypatch.setattr(
        piper_web.melo_training,
        "start",
        lambda config, name, directory: started.update(config=config, name=name, directory=directory),
    )
    app = FastAPI()
    app.include_router(piper_web.router)
    client = TestClient(app)

    response = client.post(
        "/api/melo/train",
        data={
            "dataset_dir": str(dataset),
            "output_name": "precision-test",
            "num_epochs": 1,
            "save_every_epochs": 1,
            "precision": "fp32",
        },
    )
    invalid = client.post(
        "/api/melo/train",
        data={"dataset_dir": str(dataset), "precision": "tf32"},
    )

    assert response.status_code == 200
    assert response.json()["quality"].endswith("FP32")
    assert json.loads(Path(started["config"]).read_text(encoding="utf-8"))["precision"] == "fp32"
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "MeloTTS 训练精度无效"


def test_melo_non_finite_guard_stops_nan_and_inf():
    melo_dir = Path(__file__).parents[1] / "third_party" / "MeloTTS" / "melo"
    import_paths = [str(melo_dir.parent), str(melo_dir)]
    sys.path[:0] = import_paths
    try:
        spec = importlib.util.spec_from_file_location("voxcpm_melo_train_test", melo_dir / "train.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for path in import_paths:
            sys.path.remove(path)

    module.assert_finite_training_values(3, loss=torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match="step 4: generator_loss, grad_norm"):
        module.assert_finite_training_values(
            4,
            generator_loss=torch.tensor(float("nan")),
            grad_norm=torch.tensor(float("inf")),
        )
