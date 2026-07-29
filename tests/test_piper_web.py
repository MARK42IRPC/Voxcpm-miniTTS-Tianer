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
    assert artifact["architecture"] == "piper"
    assert artifact["precision"] == "fp32"
    assert artifact["export_precisions"] == ["fp32", "fp16", "int8"]
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
    assert artifact["architecture"] == "melotts"
    assert artifact["precision"] == "int8"
    assert artifact["export_precisions"] == ["int8"]
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


def test_native_melo_student_manifest_and_bundle(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    downloads = tmp_path / "downloads"
    outputs = tmp_path / "outputs"
    for directory in (models, runs, downloads, outputs):
        directory.mkdir()
    model_dir = models / "native-melo"
    model_dir.mkdir()
    model_path = model_dir / "model.fp32.onnx"
    model_path.write_bytes(b"fake-native-melo-onnx")
    (model_dir / "config.json").write_text(
        json.dumps({"train": {}, "model": {}, "data": {}, "symbols": ["_"]}),
        encoding="utf-8",
    )
    (model_dir / piper_web.STUDENT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "engine": "melo_onnx_native",
                "engine_label": "MeloTTS Native ONNX",
                "display_name": "原生质量模型",
                "model": model_path.name,
                "config": "config.json",
                "sample_rate": 44100,
                "quality": "fp32-finetuned",
                "precision": "fp32",
                "language": "zh_CN+en_US",
                "bundle_files": [model_path.name, "config.json", piper_web.STUDENT_MANIFEST_NAME],
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

    assert artifact["engine"] == "melo_onnx_native"
    assert artifact["architecture"] == "melotts"
    assert artifact["engine_label"] == "MeloTTS Native ONNX"
    download = client.get(f"/api/piper/download/{artifact['id']}")
    assert download.status_code == 200
    with zipfile.ZipFile(next(downloads.glob("*.zip"))) as archive:
        assert set(archive.namelist()) == {"model.fp32.onnx", "config.json", piper_web.STUDENT_MANIFEST_NAME}


def test_native_melo_preview_uses_quality_defaults(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    outputs = tmp_path / "outputs"
    for directory in (models, runs, outputs):
        directory.mkdir()
    model_dir = models / "native-melo"
    model_dir.mkdir()
    model_path = model_dir / "model.fp32.onnx"
    model_path.write_bytes(b"fake-native-melo-onnx")
    (model_dir / "config.json").write_text(
        json.dumps({"train": {}, "model": {}, "data": {}, "symbols": ["_"]}),
        encoding="utf-8",
    )
    (model_dir / piper_web.STUDENT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "engine": "melo_onnx_native",
                "engine_label": "MeloTTS Native ONNX",
                "model": model_path.name,
                "config": "config.json",
                "sample_rate": 44100,
                "precision": "fp32",
                "sdp_ratio": 0.2,
                "bundle_files": [model_path.name, "config.json", piper_web.STUDENT_MANIFEST_NAME],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def synthesize(model, manifest, text, output, settings):
        captured.update(model=model, manifest=manifest, text=text, settings=settings)
        sf.write(output, np.zeros(4410, dtype=np.float32), 44100)

    released = []
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "PIPER_OUTPUT_ROOT", outputs)
    monkeypatch.setattr(piper_web, "_release_inference", lambda: released.append(True))
    monkeypatch.setattr(piper_web.melo_native_voice_runtime, "synthesize", synthesize)
    app = FastAPI()
    app.include_router(piper_web.router)
    artifact = piper_web.list_piper_artifacts()[0]

    response = TestClient(app).post(
        "/api/piper/preview",
        data={"model_id": artifact["id"], "text": "原生质量试听。"},
    )

    assert response.status_code == 200
    assert captured["model"] == model_path
    assert captured["text"] == "原生质量试听。"
    assert captured["settings"]["noise_scale"] == 0.6
    assert captured["settings"]["noise_w_scale"] == 0.8
    assert captured["settings"]["sdp_ratio"] == 0.2
    assert released == [True]


def test_native_melo_export_keeps_frontend_and_duration_inputs():
    source = (piper_web.ROOT / "scripts" / "export_melo_onnx.py").read_text(encoding="utf-8")

    for input_name in ("language", "bert", "ja_bert", "sdp_ratio"):
        assert f'"{input_name}"' in source
    assert "sdp_ratio=sdp_ratio" in source
    assert 'parser.add_argument("--runtime", choices=("native", "sherpa"), default="native")' in source


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
    assert artifact["precision"] == "fp32"
    assert artifact["export_precisions"] == ["fp32", "fp16", "int8"]
    with pytest.raises(ValueError, match="需要 checkpoint"):
        piper_web.resolve_piper_artifact(artifact["id"], "checkpoint")


def test_convert_onnx_precision_creates_fp16_and_int8_without_touching_source(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source = tmp_path / "source.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "precision-test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, source)
    original = source.read_bytes()

    fp16 = piper_web.convert_onnx_precision(source, tmp_path / "model.fp16.onnx", "fp16")
    int8 = piper_web.convert_onnx_precision(source, tmp_path / "model.int8.onnx", "int8")

    assert source.read_bytes() == original
    assert fp16.is_file() and int8.is_file()
    assert onnx.load(fp16).graph.initializer[0].data_type == TensorProto.FLOAT16
    assert any(node.op_type in {"MatMulInteger", "DynamicQuantizeLinear"} for node in onnx.load(int8).graph.node)


def test_piper_checkpoint_preview_exports_runtime_model_once(monkeypatch, tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "epoch.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = checkpoint.parent.parent / "voice.json"
    config.write_text("{}", encoding="utf-8")
    preview_cache = tmp_path / "preview-cache"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    exports = []
    syntheses = []

    def fake_export(source, destination, config_path):
        exports.append((source, destination, config_path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"onnx")
        Path(f"{destination}.json").write_text("{}", encoding="utf-8")
        return destination

    def fake_synthesize(model, config_path, text, output, settings):
        syntheses.append((model, text))
        sf.write(output, np.zeros(800, dtype=np.float32), 16000)

    monkeypatch.setattr(piper_web, "STUDENT_PREVIEW_CACHE_ROOT", preview_cache)
    monkeypatch.setattr(piper_web, "PIPER_OUTPUT_ROOT", outputs)
    monkeypatch.setattr(piper_web, "export_piper_checkpoint", fake_export)
    monkeypatch.setattr(piper_web.piper_voice_runtime, "synthesize", fake_synthesize)
    artifact = {"id": "checkpoint-id", "name": "epoch"}
    settings = {"length_scale": 1.0, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0, "speaker_id": None}

    first = piper_web.preview_piper_checkpoint(artifact, checkpoint, "测试。", settings)
    second = piper_web.preview_piper_checkpoint(artifact, checkpoint, "测试。", settings)

    assert first["cached"] is False
    assert second["cached"] is True
    assert len(exports) == 1
    assert len(syntheses) == 1


def test_student_export_endpoint_creates_discoverable_precision_artifact(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    checkpoint = runs / "speaker" / "checkpoints" / "epoch.ckpt"
    checkpoint.parent.mkdir(parents=True)
    models.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    (runs / "speaker" / "voice.json").write_text(
        json.dumps({"audio": {"sample_rate": 16000, "quality": "x_low"}, "language": {"code": "zh_CN"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)

    def fake_export(source, output, config, precision):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        Path(f"{output}.json").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        return output

    monkeypatch.setattr(piper_web, "export_piper_checkpoint_precision", fake_export)
    checkpoint_artifact = next(item for item in piper_web.list_piper_artifacts() if item["kind"] == "checkpoint")
    app = FastAPI()
    app.include_router(piper_web.router)

    response = TestClient(app).post(
        f"/api/export/artifact/{checkpoint_artifact['id']}",
        data={"precision": "int8"},
    )

    assert response.status_code == 200
    exported = response.json()["artifact"]
    assert exported["kind"] == "onnx"
    assert exported["precision"] == "int8"
    assert response.json()["download_url"].endswith(exported["id"])


def test_unified_student_preview_routes_onnx_to_runtime(monkeypatch, tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"onnx")
    artifact = {
        "id": "onnx-id",
        "kind": "onnx",
        "name": "voice",
    }
    captured = {}

    async def fake_preview(**kwargs):
        captured.update(kwargs)
        return {"filename": "preview.wav", "audio_url": "/audio", "duration": 1.5, "cached": False, "model": "voice"}

    monkeypatch.setattr(piper_web, "resolve_piper_artifact", lambda artifact_id: (artifact, model_path))
    monkeypatch.setattr(piper_web, "piper_preview", fake_preview)
    app = FastAPI()
    app.include_router(piper_web.router)

    response = TestClient(app).post(
        "/api/export/preview",
        data={
            "artifact_id": artifact["id"],
            "text": "统一试听。",
            "length_scale": "1.2",
            "noise_scale": "0.5",
            "noise_w_scale": "0.6",
            "volume": "0.8",
        },
    )

    assert response.status_code == 200
    assert response.json()["duration"] == 1.5
    assert captured == {
        "model_id": "onnx-id",
        "text": "统一试听。",
        "length_scale": 1.2,
        "noise_scale": 0.5,
        "noise_w_scale": 0.6,
        "volume": 0.8,
    }


def test_export_page_contains_model_filters_and_global_navigation():
    source = piper_web.EXPORT_PAGE.read_text(encoding="utf-8")
    assert 'id="filterOnnx"' in source
    assert 'id="filterCheckpoint"' in source
    assert 'id="architectureFilter"' in source
    assert 'id="exportPrecision"' in source
    assert 'fetch("/api/export/preview"' in source
    assert 'fetch(`/api/export/artifact/${item.id}`' in source
    for page in ("web.html", "lora.html", "datasets.html", "distill.html", "export.html"):
        assert 'href="/export"' in (piper_web.ROOT / page).read_text(encoding="utf-8")


def test_optimizer_head_model_is_identity_initialized_and_under_ten_megabytes():
    from scripts.train_optimizer_head import TinyCrnHybrid

    model = TinyCrnHybrid().eval()
    source = torch.rand(1, 1, 257, 41)

    with torch.inference_mode():
        output = model(source)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == source.shape
    assert torch.allclose(output, source)
    assert parameter_count == 1_784_018
    assert parameter_count * 4 < 10 * 1024**2


def test_optimizer_spectral_alignment_matches_student_time_axis(tmp_path):
    from scripts.train_optimizer_head import align_spectra

    sample_rate = 16000
    clean_time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    clean = 0.2 * np.sin(2 * np.pi * 220 * clean_time)
    student = np.concatenate((np.zeros(1600, dtype=np.float32), clean, np.zeros(800, dtype=np.float32)))
    clean_path = tmp_path / "clean.wav"
    student_path = tmp_path / "student.wav"
    sf.write(clean_path, clean, sample_rate)
    sf.write(student_path, student, sample_rate)

    input_log, target_log, metadata = align_spectra(student_path, clean_path, sample_rate, 512, 128)

    assert input_log.shape == target_log.shape
    assert input_log.shape[0] == 257
    assert metadata["student_frames"] == input_log.shape[1]
    assert metadata["student_seconds"] > metadata["clean_seconds"]
    assert np.isfinite(target_log).all()


def test_optimizer_training_endpoint_writes_job_and_starts_runtime(monkeypatch, tmp_path):
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    heads = tmp_path / "optimization-heads"
    downloads = tmp_path / "optimizer-downloads"
    dataset = tmp_path / "dataset"
    for directory in (models, runs, heads, downloads, dataset):
        directory.mkdir()
    model_dir = models / "student"
    model_dir.mkdir()
    model_path = model_dir / "student.onnx"
    model_path.write_bytes(b"onnx")
    Path(f"{model_path}.json").write_text(
        json.dumps({"audio": {"sample_rate": 16000, "quality": "x_low"}, "language": {"code": "zh_CN"}}),
        encoding="utf-8",
    )
    for index in range(2):
        sf.write(dataset / f"sample-{index}.wav", np.zeros(1600, dtype=np.float32), 16000)
        (dataset / f"sample-{index}.lab").write_text(f"测试文本 {index}", encoding="utf-8")

    class FakeRuntime:
        running = False

        def __init__(self):
            self.started = None

        def start(self, config_path, job_name, job_dir):
            self.started = (config_path, job_name, job_dir)

        def snapshot(self):
            return {"status": "idle", "running": False, "logs": "", "started_at": 0}

    runtime = FakeRuntime()
    monkeypatch.setattr(piper_web, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(piper_web, "PIPER_MODELS_ROOT", models)
    monkeypatch.setattr(piper_web, "PIPER_RUNS_ROOT", runs)
    monkeypatch.setattr(piper_web, "OPTIMIZER_HEAD_ROOT", heads)
    monkeypatch.setattr(piper_web, "OPTIMIZER_DOWNLOAD_ROOT", downloads)
    monkeypatch.setattr(piper_web, "optimizer_head_training", runtime)
    monkeypatch.setattr(piper_web.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(piper_web, "_other_training_running", lambda: False)
    artifact = piper_web.list_piper_artifacts()[0]
    app = FastAPI()
    app.include_router(piper_web.router)

    response = TestClient(app).post(
        "/api/optimizer/train",
        data={
            "model_id": artifact["id"],
            "dataset_dir": str(dataset),
            "architecture": "tiny_crn_hybrid",
            "output_name": "test-head",
            "epochs": "3",
            "batch_size": "2",
            "learning_rate": "0.0003",
            "training_precision": "fp32",
            "export_precision": "int8",
            "validation_split": "0.1",
            "chunk_frames": "128",
        },
    )

    assert response.status_code == 200
    config_path, job_name, job_dir = runtime.started
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert job_name == "test-head"
    assert job_dir == heads / "test-head"
    assert config["model_path"] == str(model_path.resolve())
    assert config["sample_rate"] == 16000
    assert config["n_fft"] == 512
    assert config["export_precision"] == "int8"


def test_optimizer_head_scan_download_and_page_navigation(monkeypatch, tmp_path):
    heads = tmp_path / "heads"
    downloads = tmp_path / "downloads"
    head_dir = heads / "voice-cleaner"
    head_dir.mkdir(parents=True)
    downloads.mkdir()
    model_path = head_dir / "optimizer-head.int8.onnx"
    model_path.write_bytes(b"head-onnx")
    (head_dir / "optimizer-head.json").write_text(
        json.dumps(
            {
                "format": "voxcpm-optimizer-head-v1",
                "display_name": "voice-cleaner",
                "architecture": "tiny_crn_hybrid",
                "model": model_path.name,
                "precision": "int8",
                "parameter_count": 1_784_018,
                "sample_rate": 22050,
                "validation_loss": 0.12,
                "best_epoch": 4,
                "source_model": {"id": "student", "name": "student"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_web, "OPTIMIZER_HEAD_ROOT", heads)
    monkeypatch.setattr(piper_web, "OPTIMIZER_DOWNLOAD_ROOT", downloads)
    app = FastAPI()
    app.include_router(piper_web.router)
    client = TestClient(app)

    listed = piper_web.list_optimizer_heads()
    response = client.get(f"/api/optimizer/download/{listed[0]['id']}")

    assert listed[0]["precision"] == "int8"
    assert listed[0]["parameter_count"] == 1_784_018
    assert response.status_code == 200
    with zipfile.ZipFile(next(downloads.glob("*.zip"))) as archive:
        assert set(archive.namelist()) == {"optimizer-head.int8.onnx", "optimizer-head.json"}
    for page in ("web.html", "lora.html", "datasets.html", "distill.html", "optimizer.html", "export.html"):
        source = (piper_web.ROOT / page).read_text(encoding="utf-8")
        assert 'href="/optimizer"' in source
    optimizer_source = (piper_web.ROOT / "optimizer.html").read_text(encoding="utf-8")
    assert 'fetch("/api/optimizer/train"' in optimizer_source
    assert 'id="modelId"' in optimizer_source
    assert 'id="datasetDir"' in optimizer_source


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
