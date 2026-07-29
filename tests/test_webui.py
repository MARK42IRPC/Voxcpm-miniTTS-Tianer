from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import threading
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from safetensors.torch import save_file

import webui
from webui import (
    LoRATrainingRuntime,
    InferenceJobRuntime,
    ModelConfig,
    ModelRuntime,
    build_text_tasks,
    build_generation_seed_tasks,
    build_prompt_kwargs,
    calculate_lora_training_schedule,
    inspect_lora_dataset,
    list_lora_checkpoints,
    merge_text_task_results,
    resolve_lora_checkpoint,
    safe_lora_job_name,
    split_sentence_text,
    validate_text_task_limits,
    infer_lora_model_key,
)


class FakeRunningProcess:
    def poll(self):
        return None


class FakeModel:
    def __init__(self):
        self.build_count = 0
        self.generate_count = 0
        self.generated_seeds = []
        self.loaded_loras = []
        self.lora_enabled = None
        self.lora_scale = None
        self.tts_model = SimpleNamespace(sample_rate=48000, last_successful_seed=None)

    def build_prompt_cache(self, **kwargs):
        self.build_count += 1
        return {"encoded": kwargs}

    def ensure_denoiser(self, model_path, device=None):
        return None

    def generate(self, text, prompt_cache, **kwargs):
        self.generate_count += 1
        self.generated_seeds.append(kwargs["seed"])
        self.tts_model.last_successful_seed = kwargs["seed"]
        assert prompt_cache is not None
        return np.zeros(8, dtype=np.float32)

    def load_lora(self, path):
        self.loaded_loras.append(path)
        return ["lora_A", "lora_B"], []

    def set_lora_enabled(self, enabled):
        self.lora_enabled = enabled

    def set_lora_scale(self, multiplier):
        self.lora_scale = multiplier


def make_runtime_with_model(config):
    runtime = ModelRuntime()
    model = FakeModel()
    runtime._model = model
    runtime._config = config
    runtime._identity = (config.model_key, config.device, config.optimize)
    return runtime, model


def test_cpu_thread_override(monkeypatch):
    calls = {}
    monkeypatch.setenv("VOXCPM_CPU_THREADS", "6")
    monkeypatch.setattr(webui.torch, "set_num_threads", lambda value: calls.setdefault("intra", value))
    monkeypatch.setattr(webui.torch, "set_num_interop_threads", lambda value: calls.setdefault("inter", value))

    assert webui.configure_torch_cpu_threads() == 6
    assert calls == {"intra": 6, "inter": 1}


def test_split_sentence_text_on_chinese_and_english_boundaries():
    assert split_sentence_text("第一句。“真的吗？！”继续；Yes! Version 1.5 works. 最后一段") == [
        "第一句。",
        "“真的吗？！”",
        "继续；",
        "Yes!",
        "Version 1.5 works.",
        "最后一段",
    ]


def test_ordinary_text_builds_one_task_from_all_sentences_and_lines():
    assert build_text_tasks("第一句。第二句！\n第三句？", "ordinary") == [
        {
            "text": "第一句。第二句！ 第三句？",
            "segments": ["第一句。", "第二句！", "第三句？"],
            "sentence_count": 3,
        }
    ]


def test_batch_text_builds_one_ordinary_task_per_non_empty_line():
    assert build_text_tasks("第一句。第二句！\n\n第三句？\r\n第四行", "batch") == [
        {"text": "第一句。第二句！", "segments": ["第一句。", "第二句！"], "sentence_count": 2},
        {"text": "第三句？", "segments": ["第三句？"], "sentence_count": 1},
        {"text": "第四行", "segments": ["第四行"], "sentence_count": 1},
    ]


def test_text_split_mode_none_keeps_each_output_task_whole():
    assert build_text_tasks("第一句。第二句！\n第三句？", "ordinary", "none", 1) == [
        {
            "text": "第一句。第二句！ 第三句？",
            "segments": ["第一句。第二句！ 第三句？"],
            "sentence_count": 3,
        }
    ]
    assert build_text_tasks("第一句。第二句！\n第三句？", "batch", "none", 1) == [
        {"text": "第一句。第二句！", "segments": ["第一句。第二句！"], "sentence_count": 2},
        {"text": "第三句？", "segments": ["第三句？"], "sentence_count": 1},
    ]


def test_text_split_groups_configured_sentence_count_and_preserves_english_spacing():
    tasks = build_text_tasks("第一句。第二句！Third! Fourth.", "ordinary", "sentences", 2)

    assert tasks[0]["segments"] == ["第一句。第二句！", "Third! Fourth."]
    assert tasks[0]["sentence_count"] == 4


def test_text_split_by_characters_rounds_up_to_sentence_boundary():
    tasks = build_text_tasks("甲乙。丙丁戊！己。最后一句。", "ordinary", "characters", 5)

    assert tasks[0]["segments"] == ["甲乙。丙丁戊！", "己。最后一句。"]
    assert all(segment.endswith(("。", "！")) for segment in tasks[0]["segments"])


def test_batch_seed_rotation_is_sequential_and_reproducible():
    tasks = build_text_tasks("第一句。第二句！\n第三句？", "batch")

    assert build_generation_seed_tasks(tasks, 42, True) == [[42, 43], [44]]
    assert build_generation_seed_tasks(tasks, 42, False) == [[42, 42], [42]]
    assert build_generation_seed_tasks(tasks, 2**32 - 1, True) == [[2**32 - 1, 0], [1]]


def test_batch_text_has_no_task_or_segment_limit():
    tasks = build_text_tasks("\n".join(["第一句。第二句！"] * 1200), "batch")

    segments = validate_text_task_limits(tasks, "batch")

    assert len(tasks) == 1200
    assert len(segments) == 2400


def test_ordinary_text_limit_remains_100_segments():
    tasks = build_text_tasks("一句。" * 101, "ordinary")

    with pytest.raises(HTTPException, match="100 个推理分段"):
        validate_text_task_limits(tasks, "ordinary")


def test_merge_text_task_results_preserves_waveform_dtype_and_task_boundaries():
    tasks = build_text_tasks("第一句。第二句！\n第三句？", "batch")
    results = [
        (np.array([0.1, 0.2], dtype=np.float32), 42, 1.25),
        (np.array([0.3], dtype=np.float32), 43, 0.5),
        (np.array([0.4, 0.5], dtype=np.float32), 42, 0.75),
    ]

    merged = merge_text_task_results(tasks, results)

    assert len(merged) == 2
    assert merged[0]["wav"].dtype == np.float32
    assert merged[0]["wav"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert merged[0]["successful_seeds"] == [42, 43]
    assert merged[0]["processing_seconds"] == 1.75
    assert merged[1]["wav"].tolist() == pytest.approx([0.4, 0.5])


def test_generate_batch_writes_one_merged_wav_per_non_empty_line(monkeypatch, tmp_path):
    generated_texts = []
    output_root = tmp_path / "web-cache"
    export_root = tmp_path / "batch-export"
    output_root.mkdir()
    export_root.mkdir()

    def fake_generate_tasks(
        config,
        text_tasks,
        prompt_cache_key,
        prompt_build_kwargs,
        lora_checkpoint,
        on_segment_complete=None,
        on_task_complete=None,
        task_seeds=None,
        **kwargs,
    ):
        assert task_seeds == [[42, 43], [44]]
        task_results = [
            [
                (np.full(2, 0.1, dtype=np.float32), 42, 0.1),
                (np.full(3, 0.2, dtype=np.float32), 42, 0.2),
            ],
            [(np.full(4, 0.3, dtype=np.float32), 42, 0.3)],
        ]
        for task_index, (texts, results) in enumerate(zip(text_tasks, task_results)):
            if task_index == 1:
                assert len(list(output_root.glob("*.wav"))) == 1
                assert len(list(export_root.glob("*.wav"))) == 1
                assert len(list(export_root.glob("*.lab"))) == 1
            generated_texts.extend(texts)
            for _ in results:
                on_segment_complete()
            on_task_complete(task_index, results, 16000, False)
        return [], 16000, False, False, len(text_tasks)

    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui, "BATCH_OUTPUT_DIRECTORY_REGISTRY", tmp_path / "batch-directories.json")
    monkeypatch.setattr(webui.runtime, "generate_tasks", fake_generate_tasks)
    webui.register_batch_output_directory(export_root)
    response = TestClient(webui.app).post(
        "/api/generate",
        data={
            "text": "第一句。第二句！\n第三句？",
            "model_key": "voxcpm-0.5b",
            "mode": "design",
            "batch_mode": "batch",
            "device": "cpu",
            "denoise": "false",
            "optimize": "false",
            "batch_output_dir": str(export_root),
            "create_training_pairs": "true",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert generated_texts == ["第一句。", "第二句！", "第三句？"]
    assert result["task_count"] == 2
    assert result["segment_count"] == 3
    assert result["batch_output_dir"] == str(export_root.resolve())
    assert result["training_pairs_created"] is True
    assert len(result["outputs"]) == 2
    first_path = output_root / result["outputs"][0]["filename"]
    second_path = output_root / result["outputs"][1]["filename"]
    assert sf.info(first_path).frames == 5
    assert sf.info(second_path).frames == 4
    assert webui.read_wav_metadata(first_path)["segments"] == ["第一句。", "第二句！"]
    assert webui.read_wav_metadata(first_path)["split_mode"] == "sentences"
    assert webui.read_wav_metadata(first_path)["split_value"] == 1
    assert webui.read_wav_metadata(first_path)["seed_strategy"] == "sequential"
    assert webui.read_wav_metadata(first_path)["effective_seeds"] == [42, 43]
    assert webui.read_wav_metadata(second_path)["text"] == "第三句？"
    exported_paths = [Path(output["exported_path"]) for output in result["outputs"]]
    assert all(path.parent == export_root.resolve() and path.is_file() for path in exported_paths)
    assert [Path(output["lab_path"]).read_text(encoding="utf-8") for output in result["outputs"]] == [
        "第一句。第二句！",
        "第三句？",
    ]
    request_params = webui.inference_job.snapshot()["request_params"]
    assert request_params["text"] == "第一句。第二句！\n第三句？"
    assert request_params["batch_mode"] == "batch"
    assert request_params["batch_output_dir"] == str(export_root.resolve())
    assert request_params["create_training_pairs"] is True
    assert request_params["rotate_seed"] is True
    assert request_params["reference_filename"] is None


def test_generate_applies_character_split_and_records_configuration(monkeypatch, tmp_path):
    captured = {}
    output_root = tmp_path / "web-cache"
    output_root.mkdir()

    def fake_generate_tasks(
        config,
        text_tasks,
        *args,
        on_segment_complete=None,
        on_task_complete=None,
        task_seeds=None,
        **kwargs,
    ):
        captured["text_tasks"] = text_tasks
        captured["task_seeds"] = task_seeds
        results = [
            (np.full(2, 0.1, dtype=np.float32), task_seeds[0][0], 0.1),
            (np.full(3, 0.2, dtype=np.float32), task_seeds[0][1], 0.2),
        ]
        on_segment_complete()
        on_segment_complete()
        on_task_complete(0, results, 16000, False)
        return [], 16000, False, False, 1

    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui.runtime, "generate_tasks", fake_generate_tasks)
    response = TestClient(webui.app).post(
        "/api/generate",
        data={
            "text": "甲乙。丙丁戊！己。最后一句。",
            "model_key": "voxcpm-0.5b",
            "mode": "design",
            "batch_mode": "ordinary",
            "split_mode": "characters",
            "split_value": 5,
            "device": "cpu",
            "denoise": "false",
            "optimize": "false",
        },
    )

    assert response.status_code == 200
    assert captured["text_tasks"] == [["甲乙。丙丁戊！", "己。最后一句。"]]
    assert captured["task_seeds"] == [[42, 42]]
    assert response.json()["segment_count"] == 2
    assert response.json()["split_mode"] == "characters"
    assert response.json()["split_value"] == 5
    metadata = webui.read_wav_metadata(output_root / response.json()["filename"])
    assert metadata["segments"] == ["甲乙。丙丁戊！", "己。最后一句。"]
    assert metadata["sentence_count"] == 4
    assert metadata["segment_count"] == 2


def test_generate_keeps_optimization_enabled_with_lora(monkeypatch, tmp_path):
    captured = {}
    output_root = tmp_path / "web-cache"
    output_root.mkdir()
    checkpoint = {
        "id": "speaker/checkpoints/step_1",
        "display_name": "speaker · step_1",
        "path": str(tmp_path / "checkpoint"),
        "signature": "same-config",
        "lora_config": {"enable_lm": False, "enable_dit": True, "enable_proj": False, "r": 4, "alpha": 8},
    }

    def fake_generate_tasks(config, text_tasks, *args, on_segment_complete=None, on_task_complete=None, **kwargs):
        captured["config"] = config
        result = [(np.full(4, 0.1, dtype=np.float32), 42, 0.1)]
        on_segment_complete()
        on_task_complete(0, result, 16000, False)
        return [], 16000, False, False, 1

    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui, "resolve_lora_checkpoint", lambda lora_id, model_key: checkpoint)
    monkeypatch.setattr(webui.runtime, "generate_tasks", fake_generate_tasks)
    monkeypatch.setattr(webui.torch.cuda, "is_available", lambda: True)

    response = TestClient(webui.app).post(
        "/api/generate",
        data={
            "text": "LoRA 优化测试。",
            "model_key": "voxcpm-0.5b",
            "lora_id": checkpoint["id"],
            "mode": "design",
            "batch_mode": "ordinary",
            "device": "cuda",
            "optimize": "true",
            "denoise": "false",
        },
    )

    assert response.status_code == 200
    assert captured["config"].optimize is True
    assert response.json()["lora_id"] == checkpoint["id"]
    assert webui.read_wav_metadata(output_root / response.json()["filename"])["optimize_effective"] is True


def test_inference_job_tracks_progress_outputs_and_soft_cancel():
    job = InferenceJobRuntime()
    request_params = {"model_key": "voxcpm1.5", "seed": 42}
    cancel_event = job.start(
        batch_mode="batch",
        model="voxcpm1.5",
        lora_name=None,
        device="cuda",
        total_tasks=2,
        total_segments=3,
        request_params=request_params,
    )
    request_params["seed"] = 99

    job.record_segment()
    job.record_output({"filename": "first.wav"})
    cancel_result = job.request_cancel()
    job.finish(cancelled=True)
    status = job.snapshot()

    assert cancel_result["accepted"] is True
    assert cancel_event.is_set() is True
    assert status["status"] == "cancelled"
    assert status["completed_tasks"] == 1
    assert status["completed_segments"] == 1
    assert status["progress"] == pytest.approx(1 / 3)
    assert status["outputs"] == [{"filename": "first.wav"}]
    assert status["request_params"] == {"model_key": "voxcpm1.5", "seed": 42}
    status["request_params"]["seed"] = 7
    assert job.snapshot()["request_params"]["seed"] == 42


def test_generate_tasks_stops_before_next_segment_after_cancel():
    config = ModelConfig("voxcpm2", "hybrid", False, True)
    runtime, model = make_runtime_with_model(config)
    cancel_event = threading.Event()
    completed_tasks = []

    def on_segment_complete():
        cancel_event.set()

    _, _, _, cancelled, task_count = runtime.generate_tasks(
        config,
        [["第一句。", "第二句。"], ["第三句。"]],
        ("cache",),
        {"reference_wav_path": "reference.wav", "denoise": True},
        cancel_event=cancel_event,
        on_segment_complete=on_segment_complete,
        on_task_complete=lambda *args: completed_tasks.append(args),
        seed=42,
    )

    assert cancelled is True
    assert task_count == 0
    assert completed_tasks == []
    assert model.generate_count == 1


def test_generate_tasks_exports_completed_task_before_cancel_stops_next_task():
    config = ModelConfig("voxcpm2", "hybrid", False, True)
    runtime, model = make_runtime_with_model(config)
    cancel_event = threading.Event()
    completed_tasks = []

    _, _, _, cancelled, task_count = runtime.generate_tasks(
        config,
        [["第一条。"], ["第二条。"]],
        ("cache",),
        {"reference_wav_path": "reference.wav", "denoise": True},
        cancel_event=cancel_event,
        on_segment_complete=cancel_event.set,
        on_task_complete=lambda task_index, *args: completed_tasks.append(task_index),
        seed=42,
    )

    assert cancelled is True
    assert task_count == 1
    assert completed_tasks == [0]
    assert model.generate_count == 1


def test_generate_tasks_applies_seed_plan_per_segment():
    config = ModelConfig("voxcpm2", "hybrid", False, True)
    runtime, model = make_runtime_with_model(config)

    runtime.generate_tasks(
        config,
        [["第一句。", "第二句。"], ["第三句。"]],
        ("cache",),
        {"reference_wav_path": "reference.wav", "denoise": True},
        task_seeds=[[42, 43], [44]],
        seed=999,
    )

    assert model.generated_seeds == [42, 43, 44]


def test_inference_status_and_cancel_endpoints(monkeypatch):
    job = InferenceJobRuntime()
    monkeypatch.setattr(webui, "inference_job", job)
    job.start(
        batch_mode="batch",
        model="voxcpm1.5",
        lora_name="speaker",
        device="cuda",
        total_tasks=4,
        total_segments=6,
    )
    client = TestClient(webui.app)

    status = client.get("/api/inference/status")
    cancelled = client.post("/api/inference/cancel")

    assert status.status_code == 200
    assert status.json()["running"] is True
    assert status.json()["total_tasks"] == 4
    assert cancelled.status_code == 200
    assert cancelled.json()["accepted"] is True
    assert cancelled.json()["cancel_requested"] is True


def test_generate_ordinary_rejects_batch_export_options():
    response = TestClient(webui.app).post(
        "/api/generate",
        data={
            "text": "普通推理。",
            "model_key": "voxcpm-0.5b",
            "mode": "design",
            "batch_mode": "ordinary",
            "device": "cpu",
            "batch_output_dir": "C:\\stale-batch-path",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "批量保存选项仅可用于批量推理模式"


@pytest.mark.parametrize(
    ("split_mode", "split_value", "expected_detail"),
    [
        ("unknown", 1, "Unknown text split mode"),
        ("sentences", 0, "Sentences per segment must be between 1 and 100"),
        ("sentences", 101, "Sentences per segment must be between 1 and 100"),
        ("characters", 0, "Target characters must be between 1 and 10000"),
        ("characters", 10001, "Target characters must be between 1 and 10000"),
    ],
)
def test_generate_rejects_invalid_text_split_config(split_mode, split_value, expected_detail):
    response = TestClient(webui.app).post(
        "/api/generate",
        data={
            "text": "切分配置测试。",
            "model_key": "voxcpm-0.5b",
            "mode": "design",
            "batch_mode": "ordinary",
            "split_mode": split_mode,
            "split_value": split_value,
            "device": "cpu",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_batch_export_destination_does_not_overwrite_existing_pair(tmp_path):
    source = tmp_path / "cache" / "voxcpm-26 07 28 12 00 00.wav"
    destination = tmp_path / "export"
    source.parent.mkdir()
    destination.mkdir()
    (destination / source.name).write_bytes(b"existing audio")
    (destination / source.with_suffix(".lab").name).write_text("existing text", encoding="utf-8")

    selected = webui.batch_export_destination(source, destination, create_training_pair=True)

    assert selected.name == "voxcpm-26 07 28 12 00 00-02.wav"
    assert (destination / source.name).read_bytes() == b"existing audio"
    assert (destination / source.with_suffix(".lab").name).read_text(encoding="utf-8") == "existing text"


def test_ultimate_prompt_uses_audio_once_as_continuation():
    kwargs = build_prompt_kwargs("ultimate", Path("reference.wav"), " 参考文案 ", True)

    assert kwargs == {
        "prompt_wav_path": "reference.wav",
        "prompt_text": "参考文案。",
        "denoise": True,
    }
    assert "reference_wav_path" not in kwargs


def test_ultimate_prompt_keeps_existing_sentence_boundary():
    kwargs = build_prompt_kwargs("ultimate", Path("reference.wav"), "参考文案！", False)

    assert kwargs["prompt_text"] == "参考文案！"


def test_reference_prompt_stays_structurally_isolated():
    assert build_prompt_kwargs("reference", Path("reference.wav"), "ignored", False) == {
        "reference_wav_path": "reference.wav",
        "denoise": False,
    }


def test_stable_hybrid_reuses_prompt_cache_across_requests():
    config = ModelConfig("voxcpm2", "hybrid", False, True)
    runtime, model = make_runtime_with_model(config)
    cache_key = ("voxcpm2", "reference", "hash", "", True)
    build_kwargs = {"reference_wav_path": "reference.wav", "denoise": True}

    first, sample_rate, first_hit = runtime.generate_many(
        config,
        ["第一句。", "第二句。"],
        cache_key,
        build_kwargs,
        seed=42,
    )
    second, _, second_hit = runtime.generate_many(
        config,
        ["第三句。"],
        cache_key,
        build_kwargs,
        seed=42,
    )

    assert sample_rate == 48000
    assert len(first) == 2
    assert len(second) == 1
    assert first_hit is False
    assert second_hit is True
    assert model.build_count == 1
    assert model.generate_count == 3


def test_non_stable_mode_reuses_only_within_current_batch():
    config = ModelConfig("voxcpm2", "cpu", False, False)
    runtime, model = make_runtime_with_model(config)
    cache_key = ("voxcpm2", "reference", "hash", "", False)
    build_kwargs = {"reference_wav_path": "reference.wav", "denoise": False}

    runtime.generate_many(config, ["第一句。", "第二句。"], cache_key, build_kwargs, seed=42)
    runtime.generate_many(config, ["第三句。"], cache_key, build_kwargs, seed=42)

    assert model.build_count == 2
    assert model.generate_count == 3
    assert runtime.prompt_cache_ready is False


def test_lora_dataset_inspection_matches_wav_and_lab():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sf.write(root / "sample.wav", np.zeros(1600, dtype=np.float32), 16000)
        (root / "sample.lab").write_text("测试文本。", encoding="utf-8")

        result = inspect_lora_dataset(directory)

    assert result["file_count"] == 1
    assert result["sample_rates"] == {16000: 1}
    assert result["records"][0]["text"] == "测试文本。"
    assert result["records"][0]["duration"] == 0.1


def test_lora_job_name_removes_windows_path_characters():
    assert safe_lora_job_name("爱弥斯:test/01") == "爱弥斯_test_01"


def test_lora_training_schedule_converts_epochs_to_optimizer_steps():
    schedule = calculate_lora_training_schedule(
        sample_count=106,
        batch_size=1,
        grad_accum_steps=16,
        num_epochs=20,
        save_every_epochs=5,
    )

    assert schedule == {
        "num_epochs": 20,
        "effective_batch_size": 16,
        "optimizer_steps": 133,
        "save_every_epochs": 5,
        "save_interval_steps": 34,
    }


def test_lora_pause_creates_cooperative_request(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "LORA_ROOT", tmp_path)
    training = LoRATrainingRuntime()
    training._process = FakeRunningProcess()
    training._status = "running"
    training._job_name = "test-job"

    assert training.pause() is True

    request_path = tmp_path / "test-job" / "checkpoints" / ".pause-request"
    assert request_path.is_file()
    snapshot = training.snapshot()
    assert snapshot["status"] == "pausing"
    assert snapshot["pause_supported"] is True
    assert "Pause requested" in snapshot["logs"]


def test_lora_checkpoint_discovery_reads_model_and_config(monkeypatch, tmp_path):
    checkpoint = tmp_path / "爱弥斯" / "checkpoints" / "step_0000042"
    checkpoint.mkdir(parents=True)
    (checkpoint / "lora_weights.safetensors").write_bytes(b"weights")
    lora_config = {
        "enable_lm": False,
        "enable_dit": True,
        "enable_proj": False,
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules_lm": ["q_proj"],
        "target_modules_dit": ["q_proj"],
        "target_proj_modules": ["enc_to_lm_proj"],
    }
    (checkpoint / "lora_config.json").write_text(
        json.dumps({"base_model": r"C:\models\VoxCPM-0.5B", "lora_config": lora_config}),
        encoding="utf-8",
    )
    (checkpoint / "training_state.json").write_text('{"step": 42}', encoding="utf-8")
    monkeypatch.setattr(webui, "LORA_ROOT", tmp_path)

    checkpoints = list_lora_checkpoints()

    assert len(checkpoints) == 1
    assert checkpoints[0]["id"] == "爱弥斯/checkpoints/step_0000042"
    assert checkpoints[0]["model_key"] == "voxcpm-0.5b"
    assert checkpoints[0]["rank"] == 4
    assert checkpoints[0]["step"] == 42


def test_lora_checkpoint_rejects_base_model_mismatch(monkeypatch):
    monkeypatch.setattr(
        webui,
        "list_lora_checkpoints",
        lambda: [{"id": "job/checkpoints/step_1", "model_key": "voxcpm-0.5b"}],
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_lora_checkpoint("job/checkpoints/step_1", "voxcpm2")

    assert exc_info.value.status_code == 400


def test_voxcpm15_model_and_lora_key_are_registered():
    assert webui.MODEL_PATHS["voxcpm1.5"].name == "VoxCPM1.5"
    assert infer_lora_model_key(r"C:\models\VoxCPM1.5") == "voxcpm1.5"


def make_importable_lora_weights(path: Path, rank: int = 4) -> None:
    save_file(
        {
            "base_lm.layers.0.self_attn.q_proj.lora_A": torch.ones(rank, 8),
            "base_lm.layers.0.self_attn.q_proj.lora_B": torch.ones(8, rank),
            "feat_decoder.estimator.layers.0.self_attn.v_proj.lora_A": torch.ones(rank, 8),
            "feat_decoder.estimator.layers.0.self_attn.v_proj.lora_B": torch.ones(8, rank),
            "enc_to_lm_proj.lora_A": torch.ones(rank, 8),
            "enc_to_lm_proj.lora_B": torch.ones(8, rank),
        },
        str(path),
    )


def test_import_standalone_lora_infers_config_and_reuses_duplicate(monkeypatch, tmp_path):
    weights = tmp_path / "speaker.safetensors"
    make_importable_lora_weights(weights)
    monkeypatch.setattr(webui, "LORA_ROOT", tmp_path / "lora")
    client = TestClient(webui.app)

    def upload():
        return client.post(
            "/api/lora/checkpoints/import",
            data={"model_key": "voxcpm1.5"},
            files={"checkpoint_file": (weights.name, weights.read_bytes(), "application/octet-stream")},
        )

    first = upload()
    second = upload()

    assert first.status_code == 200
    assert first.json()["reused"] is False
    assert second.status_code == 200
    assert second.json()["reused"] is True
    checkpoint = first.json()["checkpoint"]
    assert checkpoint["model_key"] == "voxcpm1.5"
    assert checkpoint["rank"] == 4
    config = checkpoint["lora_config"]
    assert config["alpha"] == 8
    assert config["enable_lm"] is True
    assert config["enable_dit"] is True
    assert config["enable_proj"] is True
    assert len(webui.list_lora_checkpoints()) == 1


def test_import_lora_zip_preserves_config_and_step(monkeypatch, tmp_path):
    weights = tmp_path / "lora_weights.safetensors"
    make_importable_lora_weights(weights)
    config = {
        "base_model": r"C:\models\VoxCPM1.5",
        "lora_config": {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": True,
            "r": 4,
            "alpha": 12,
            "dropout": 0.0,
            "target_modules_lm": ["q_proj"],
            "target_modules_dit": ["v_proj"],
            "target_proj_modules": ["enc_to_lm_proj"],
        },
    }
    archive_path = tmp_path / "speaker.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        parent = "export/checkpoints/step_0000042"
        archive.write(weights, f"{parent}/lora_weights.safetensors")
        archive.writestr(f"{parent}/lora_config.json", json.dumps(config))
        archive.writestr(f"{parent}/training_state.json", json.dumps({"step": 42}))
    monkeypatch.setattr(webui, "LORA_ROOT", tmp_path / "lora")

    result = webui.import_lora_checkpoint(archive_path, archive_path.name, "voxcpm1.5")

    assert result["checkpoint"]["name"] == "step_0000042"
    assert result["checkpoint"]["alpha"] == 12
    assert Path(result["checkpoint"]["path"], "lora_weights.safetensors").is_file()


def test_import_lora_zip_rejects_model_mismatch_and_unsafe_paths(monkeypatch, tmp_path):
    weights = tmp_path / "lora_weights.safetensors"
    make_importable_lora_weights(weights)
    monkeypatch.setattr(webui, "LORA_ROOT", tmp_path / "lora")

    mismatch = tmp_path / "mismatch.zip"
    config = {
        "base_model": r"C:\models\VoxCPM2",
        "lora_config": {"enable_lm": True, "enable_dit": True, "enable_proj": True, "r": 4, "alpha": 8},
    }
    with zipfile.ZipFile(mismatch, "w") as archive:
        archive.write(weights, "step/lora_weights.safetensors")
        archive.writestr("step/lora_config.json", json.dumps(config))
    with pytest.raises(ValueError, match="适用于 voxcpm2"):
        webui.import_lora_checkpoint(mismatch, mismatch.name, "voxcpm1.5")

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../lora_weights.safetensors", weights.read_bytes())
    with pytest.raises(ValueError, match="不安全路径"):
        webui.import_lora_checkpoint(unsafe, unsafe.name, "voxcpm1.5")


def test_compatible_lora_switch_is_hot_and_invalidates_prompt_cache():
    config = ModelConfig("voxcpm-0.5b", "cpu", False, False)
    runtime, model = make_runtime_with_model(config)
    runtime._lora_id = "old"
    runtime._lora_signature = "same-config"
    runtime._prompt_cache = {"encoded": True}
    runtime._prompt_cache_key = ("old",)
    checkpoint = {
        "id": "new",
        "path": "new-checkpoint",
        "signature": "same-config",
    }

    selected_model = runtime.get(config, checkpoint)

    assert selected_model is model
    assert model.loaded_loras == ["new-checkpoint"]
    assert model.lora_scale == 1.0
    assert runtime.lora_id == "new"
    assert runtime.prompt_cache_ready is False


def test_lora_strength_updates_hot_and_invalidates_prompt_cache():
    config = ModelConfig("voxcpm-0.5b", "cpu", False, False)
    runtime, model = make_runtime_with_model(config)
    runtime._lora_id = "same"
    runtime._lora_signature = "same-config"
    runtime._prompt_cache = {"encoded": True}
    runtime._prompt_cache_key = ("same", 1.0)
    checkpoint = {
        "id": "same",
        "path": "same-checkpoint",
        "signature": "same-config",
    }

    selected_model = runtime.get(config, checkpoint, 1.5)

    assert selected_model is model
    assert model.loaded_loras == []
    assert model.lora_scale == 1.5
    assert runtime.lora_strength == 1.5
    assert runtime.prompt_cache_ready is False

    runtime._prompt_cache = {"encoded": True}
    runtime.get(config, None)
    assert model.lora_enabled is False
    assert runtime.lora_id is None
    assert runtime.prompt_cache_ready is False


def test_optimized_lora_switch_rebuilds_before_warmup(monkeypatch):
    config = ModelConfig("voxcpm-0.5b", "cuda", True, False)
    runtime, model = make_runtime_with_model(config)
    runtime._lora_id = "old"
    runtime._lora_signature = "same-config"
    checkpoint = {
        "id": "new",
        "path": "new-checkpoint",
        "signature": "same-config",
    }

    class RebuildRequested(Exception):
        pass

    monkeypatch.setattr(runtime, "_release", lambda: (_ for _ in ()).throw(RebuildRequested()))

    with pytest.raises(RebuildRequested):
        runtime.get(config, checkpoint)

    assert model.loaded_loras == []


def test_optimized_lora_disable_rebuilds_instead_of_hot_disabling(monkeypatch):
    config = ModelConfig("voxcpm-0.5b", "cuda", True, False)
    runtime, model = make_runtime_with_model(config)
    runtime._lora_id = "current"
    runtime._lora_signature = "same-config"

    class RebuildRequested(Exception):
        pass

    monkeypatch.setattr(runtime, "_release", lambda: (_ for _ in ()).throw(RebuildRequested()))

    with pytest.raises(RebuildRequested):
        runtime.get(config, None)

    assert model.lora_enabled is None


def test_release_optimized_model_endpoint_only_releases_compiled_runtime(monkeypatch):
    optimized_runtime, _ = make_runtime_with_model(ModelConfig("voxcpm1.5", "cuda", True, False))
    monkeypatch.setattr(webui, "runtime", optimized_runtime)
    monkeypatch.setattr(webui, "inference_job", InferenceJobRuntime())
    client = TestClient(webui.app)

    first = client.post("/api/model/release-optimized")
    second = client.post("/api/model/release-optimized")

    assert first.status_code == 200
    assert first.json() == {"released": True}
    assert second.json() == {"released": False}


def test_postprocess_audio_keeps_original_and_reuses_hashed_cache(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    source = output_root / "voxcpm-test.wav"
    sample_rate = 16000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(source, 0.15 * np.sin(2 * np.pi * 220 * timeline), sample_rate)
    webui.write_wav_metadata(
        source,
        {
            "created_at": "2026-07-27T18:00:00+08:00",
            "filename": source.name,
            "text": "后处理测试。",
            "model_key": "voxcpm1.5",
            "lora_name": None,
            "device": "cuda",
        },
    )
    settings = {
        "preset": "clean",
        "highpass_enabled": True,
        "highpass_hz": 70.0,
        "mud_enabled": True,
        "mud_gain_db": -1.5,
        "presence_enabled": True,
        "presence_gain_db": 1.0,
        "air_enabled": True,
        "air_gain_db": 0.5,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 2.0,
        "loudness_enabled": False,
        "target_lufs": -16.0,
        "limiter_enabled": False,
        "limiter_db": -1.0,
        "reverb_enabled": False,
        "reverb_wet": 0.0,
        "pitch_enabled": False,
        "pitch_semitones": 0.0,
    }

    first = webui.postprocess_audio(source, settings)
    second = webui.postprocess_audio(source, settings)
    processed = output_root / first["filename"]
    metadata = webui.read_wav_metadata(processed)

    assert source.is_file()
    assert processed.is_file()
    assert first["filename"].startswith("voxcpm-test-af-")
    assert len(first["postprocess_hash"]) == 5
    assert first["cached"] is False
    assert second["filename"] == first["filename"]
    assert second["cached"] is True
    assert metadata["text"] == "后处理测试。"
    assert metadata["source_filename"] == source.name
    assert metadata["postprocess"] == settings


def test_move_audio_creates_lab_and_rejects_duplicate_pcm(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    dataset_root = tmp_path / "dataset"
    output_root.mkdir()
    dataset_root.mkdir()
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        webui,
        "list_training_datasets",
        lambda: [{"name": "test", "path": str(dataset_root), "file_count": 0, "default": True}],
    )
    audio_data = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)

    def create_output(name, text):
        path = output_root / name
        sf.write(path, audio_data, 16000)
        webui.write_wav_metadata(
            path,
            {"created_at": "2026-07-27T18:00:00+08:00", "filename": name, "text": text},
        )
        return path

    source = create_output("first.wav", "训练文本。")
    result = webui.move_audio_to_training_dataset(source.name, str(dataset_root))

    assert not source.exists()
    assert (dataset_root / result["filename"]).is_file()
    assert (dataset_root / result["lab_filename"]).read_text(encoding="utf-8") == "训练文本。"

    duplicate = create_output("duplicate.wav", "不同元数据不影响音频去重。")
    with pytest.raises(FileExistsError, match="本条已经存在于训练集"):
        webui.move_audio_to_training_dataset(duplicate.name, str(dataset_root))
    assert duplicate.is_file()


def test_copy_audio_creates_training_pair_and_preserves_source(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    dataset_root = tmp_path / "dataset"
    output_root.mkdir()
    dataset_root.mkdir()
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        webui,
        "list_training_datasets",
        lambda: [{"name": "test", "path": str(dataset_root), "file_count": 0, "default": True}],
    )
    source = output_root / "source.wav"
    sf.write(source, np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000)
    webui.write_wav_metadata(
        source,
        {"created_at": "2026-07-28T18:00:00+08:00", "text": "复制训练文本。"},
    )

    result = webui.copy_audio_to_training_dataset(source.name, str(dataset_root))

    assert source.is_file()
    assert (dataset_root / result["filename"]).is_file()
    assert Path(result["exported_path"]) == dataset_root / result["filename"]
    assert (dataset_root / result["lab_filename"]).read_text(encoding="utf-8") == "复制训练文本。"


def test_create_training_dataset_builds_standard_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "TRAINING_DATASET_ROOT", tmp_path)

    result = webui.create_training_dataset(" 新角色语音 ")

    expected = tmp_path / "新角色语音" / "train" / "wavs"
    assert expected.is_dir()
    assert result["name"] == "新角色语音"
    assert Path(result["path"]) == expected.resolve()
    assert result["file_count"] == 0

    with pytest.raises(FileExistsError):
        webui.create_training_dataset("新角色语音")


def test_create_training_dataset_endpoint_creates_lists_and_rejects_duplicate(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "TRAINING_DATASET_ROOT", tmp_path)
    monkeypatch.setattr(webui, "DEFAULT_TRAINING_DATASET", tmp_path / "missing" / "train" / "wavs")
    client = TestClient(webui.app)

    created = client.post("/api/training-datasets", data={"name": "接口测试"})
    listed = client.get("/api/training-datasets")
    duplicate = client.post("/api/training-datasets", data={"name": "接口测试"})

    assert created.status_code == 200
    assert Path(created.json()["path"]).is_dir()
    assert any(item["path"] == created.json()["path"] for item in listed.json()["datasets"])
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "同名训练集已经存在"


def test_register_training_dataset_resolves_standard_layout(monkeypatch, tmp_path):
    dataset_root = tmp_path / "character"
    wav_dir = dataset_root / "train" / "wavs"
    wav_dir.mkdir(parents=True)
    sf.write(wav_dir / "sample.wav", np.zeros(1600, dtype=np.float32), 16000)
    registry = tmp_path / "cache" / "training-datasets.json"
    monkeypatch.setattr(webui, "TRAINING_DATASET_REGISTRY", registry)
    monkeypatch.setattr(webui, "DEFAULT_TRAINING_DATASET", tmp_path / "missing")

    result = webui.register_training_dataset(dataset_root)
    listed = webui.list_training_datasets()

    assert Path(result["path"]) == wav_dir.resolve()
    assert result["file_count"] == 1
    assert json.loads(registry.read_text(encoding="utf-8")) == [str(wav_dir.resolve())]
    assert any(item["path"] == str(wav_dir.resolve()) for item in listed)


def test_browse_training_dataset_endpoint_registers_selection(monkeypatch, tmp_path):
    wav_dir = tmp_path / "selected" / "train" / "wavs"
    wav_dir.mkdir(parents=True)
    monkeypatch.setattr(webui, "TRAINING_DATASET_REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(webui, "DEFAULT_TRAINING_DATASET", tmp_path / "missing")
    monkeypatch.setattr(webui, "choose_training_dataset_directory", lambda initial: str(tmp_path / "selected"))
    client = TestClient(webui.app)

    response = client.post("/api/training-datasets/browse", data={"initial_path": ""})

    assert response.status_code == 200
    assert response.json()["cancelled"] is False
    assert Path(response.json()["path"]) == wav_dir.resolve()


def test_browse_training_dataset_endpoint_handles_cancel(monkeypatch):
    monkeypatch.setattr(webui, "choose_training_dataset_directory", lambda initial: None)
    response = TestClient(webui.app).post("/api/training-datasets/browse", data={"initial_path": ""})
    assert response.status_code == 200
    assert response.json() == {"cancelled": True}


def configure_dataset_review_test(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "character" / "train" / "wavs"
    dataset_dir.mkdir(parents=True)
    monkeypatch.setattr(webui, "TRAINING_DATASET_REGISTRY", tmp_path / "training-datasets.json")
    monkeypatch.setattr(webui, "DATASET_REVIEW_STATE", tmp_path / "dataset-review-state.json")
    monkeypatch.setattr(webui, "DATASET_REVIEW_DATABASE", tmp_path / "dataset-review.sqlite3")
    monkeypatch.setattr(webui, "DEFAULT_TRAINING_DATASET", tmp_path / "missing")
    monkeypatch.setattr(webui, "TRAINING_DATASET_ROOT", tmp_path / "audio-root")
    webui.DATASET_REVIEW_INDEXES.clear()
    webui.register_training_dataset(dataset_dir)
    return dataset_dir


def create_redo_test_pair(dataset_dir, name="sample.wav", text="重做测试。", **overrides):
    audio_path = dataset_dir / name
    sf.write(audio_path, np.full(800, 0.1, dtype=np.float32), 16000)
    audio_path.with_suffix(".lab").write_text(text, encoding="utf-8")
    metadata = {
        "schema": "voxcpm-generation-v1",
        "created_at": "2026-07-28T18:00:00+08:00",
        "filename": name,
        "source_text": text,
        "text": text,
        "batch_mode": "ordinary",
        "split_mode": "sentences",
        "split_value": 1,
        "model_key": "voxcpm-0.5b",
        "lora_id": None,
        "lora_strength": None,
        "mode": "design",
        "control": "",
        "prompt_text": "",
        "device": "cpu",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "min_len": 2,
        "max_len": 4096,
        "requested_seed": 42,
        "seed_strategy": "fixed",
        "normalize": False,
        "denoise": False,
        "optimize_requested": False,
        "optimize_effective": False,
    }
    metadata.update(overrides)
    webui.write_wav_metadata(audio_path, metadata)
    return audio_path, metadata


def test_dataset_review_keep_persists_and_updates_statistics(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    for name, text in (("first.wav", "第一条。"), ("second.wav", "第二条。")):
        sf.write(dataset_dir / name, np.zeros(800, dtype=np.float32), 16000)
        (dataset_dir / Path(name).with_suffix(".lab")).write_text(text, encoding="utf-8")
    client = TestClient(webui.app)

    initial = client.get("/api/dataset-review", params={"dataset_dir": str(dataset_dir)})
    kept = client.post(
        "/api/dataset-review/keep",
        data={"dataset_dir": str(dataset_dir), "filename": "first.wav"},
    )
    assert initial.status_code == 200
    assert initial.json()["total_count"] == 2
    assert [item["filename"] for item in initial.json()["items"]] == ["first.wav", "second.wav"]
    assert kept.json()["confirmed_count"] == 1
    assert kept.json()["pending_count"] == 1
    assert kept.json()["removed_filename"] == "first.wav"
    assert kept.json()["replacement_item"] is None
    webui.invalidate_dataset_review_index(dataset_dir)
    refreshed = client.get("/api/dataset-review", params={"dataset_dir": str(dataset_dir)})
    assert [item["filename"] for item in refreshed.json()["items"]] == ["second.wav"]
    with webui.sqlite3.connect(webui.DATASET_REVIEW_DATABASE) as connection:
        assert connection.execute("SELECT filename FROM decisions").fetchall() == [("first.wav",)]


def test_dataset_review_migrates_legacy_json_state(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    sf.write(dataset_dir / "sample.wav", np.zeros(800, dtype=np.float32), 16000)
    webui.DATASET_REVIEW_STATE.write_text(
        json.dumps({str(dataset_dir.resolve()): ["sample.wav"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    response = TestClient(webui.app).get("/api/dataset-review", params={"dataset_dir": str(dataset_dir)})

    assert response.status_code == 200
    assert response.json()["confirmed_count"] == 1
    assert response.json()["pending_count"] == 0


def test_dataset_review_delete_removes_audio_and_matching_lab(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    audio_path = dataset_dir / "remove.wav"
    lab_path = dataset_dir / "remove.lab"
    sf.write(audio_path, np.zeros(800, dtype=np.float32), 16000)
    lab_path.write_text("需要删除。", encoding="utf-8")

    response = TestClient(webui.app).post(
        "/api/dataset-review/delete",
        data={"dataset_dir": str(dataset_dir), "filename": audio_path.name},
    )

    assert response.status_code == 200
    assert response.json()["total_count"] == 0
    assert not audio_path.exists()
    assert not lab_path.exists()


def test_dataset_review_rejects_unregistered_directory_and_path_traversal(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    sf.write(dataset_dir / "sample.wav", np.zeros(800, dtype=np.float32), 16000)
    outside = tmp_path / "outside"
    outside.mkdir()
    sf.write(outside / "secret.wav", np.zeros(800, dtype=np.float32), 16000)
    client = TestClient(webui.app)

    unregistered = client.get("/api/dataset-review", params={"dataset_dir": str(outside)})
    traversal = client.get(
        "/api/dataset-review/audio",
        params={"dataset_dir": str(dataset_dir), "filename": "../secret.wav"},
    )

    assert unregistered.status_code == 400
    assert traversal.status_code == 404


def test_dataset_review_reset_returns_kept_audio_to_queue(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    sf.write(dataset_dir / "sample.wav", np.zeros(800, dtype=np.float32), 16000)
    client = TestClient(webui.app)
    client.post(
        "/api/dataset-review/keep",
        data={"dataset_dir": str(dataset_dir), "filename": "sample.wav"},
    )

    response = client.post("/api/dataset-review/reset", data={"dataset_dir": str(dataset_dir)})

    assert response.status_code == 200
    assert response.json()["confirmed_count"] == 0
    assert response.json()["pending_count"] == 1
    assert response.json()["items"][0]["filename"] == "sample.wav"


def test_dataset_review_action_uses_cache_and_returns_one_replacement(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    monkeypatch.setattr(webui, "DATASET_REVIEW_PAGE_SIZE", 2)
    for name in ("a.wav", "b.wav", "c.wav"):
        sf.write(dataset_dir / name, np.zeros(800, dtype=np.float32), 16000)
        (dataset_dir / Path(name).with_suffix(".lab")).write_text(name, encoding="utf-8")
    client = TestClient(webui.app)
    initial = client.get("/api/dataset-review", params={"dataset_dir": str(dataset_dir)})
    monkeypatch.setattr(
        webui,
        "_scan_dataset_review_index",
        lambda directory: pytest.fail("incremental action must not rescan the dataset"),
    )

    response = client.post(
        "/api/dataset-review/keep",
        data={"dataset_dir": str(dataset_dir), "filename": "a.wav"},
    )

    assert initial.status_code == 200
    assert [item["filename"] for item in initial.json()["items"]] == ["a.wav", "b.wav"]
    assert response.status_code == 200
    assert response.json()["replacement_item"]["filename"] == "c.wav"
    assert response.json()["visible_count"] == 2


def test_dataset_review_deduplicate_keeps_first_equal_lab_pair(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    samples = {
        "a.wav": "重复文本。",
        "b.wav": "\ufeff重复文本。  \n",
        "c.wav": "唯一文本。",
        "missing.wav": None,
    }
    for name, text in samples.items():
        sf.write(dataset_dir / name, np.zeros(800, dtype=np.float32), 16000)
        if text is not None:
            (dataset_dir / Path(name).with_suffix(".lab")).write_text(text, encoding="utf-8")

    response = TestClient(webui.app).post(
        "/api/dataset-review/deduplicate",
        data={"dataset_dir": str(dataset_dir)},
    )

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 1
    assert response.json()["duplicate_groups"] == 1
    assert response.json()["total_count"] == 3
    assert (dataset_dir / "a.wav").is_file()
    assert (dataset_dir / "a.lab").is_file()
    assert not (dataset_dir / "b.wav").exists()
    assert not (dataset_dir / "b.lab").exists()
    assert (dataset_dir / "c.wav").is_file()
    assert (dataset_dir / "missing.wav").is_file()


def test_dataset_review_only_marks_fully_reproducible_metadata_for_redo(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setitem(webui.MODEL_PATHS, "voxcpm-0.5b", model_dir)
    create_redo_test_pair(dataset_dir, "supported.wav")
    create_redo_test_pair(dataset_dir, "reference.wav", mode="reference")
    create_redo_test_pair(dataset_dir, "mismatch.wav", text="元数据文本。")
    (dataset_dir / "mismatch.lab").write_text("不同的 LAB。", encoding="utf-8")

    response = TestClient(webui.app).get("/api/dataset-review", params={"dataset_dir": str(dataset_dir)})

    assert response.status_code == 200
    items = {item["filename"]: item for item in response.json()["items"]}
    assert items["supported.wav"]["redo_supported"] is True
    assert items["reference.wav"]["redo_supported"] is False
    assert items["mismatch.wav"]["redo_supported"] is False


def test_dataset_review_filters_all_pending_audio_by_metadata_mark(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setitem(webui.MODEL_PATHS, "voxcpm-0.5b", model_dir)
    create_redo_test_pair(dataset_dir, "a.wav", marked=True)
    create_redo_test_pair(dataset_dir, "b.wav")
    create_redo_test_pair(dataset_dir, "c.wav", marked=True)
    client = TestClient(webui.app)

    marked = client.get(
        "/api/dataset-review",
        params={"dataset_dir": str(dataset_dir), "mark_filter": "marked"},
    )
    unmarked = client.get(
        "/api/dataset-review",
        params={"dataset_dir": str(dataset_dir), "mark_filter": "unmarked"},
    )
    kept = client.post(
        "/api/dataset-review/keep",
        data={"dataset_dir": str(dataset_dir), "filename": "a.wav", "mark_filter": "marked"},
    )
    invalid = client.get(
        "/api/dataset-review",
        params={"dataset_dir": str(dataset_dir), "mark_filter": "unknown"},
    )

    assert marked.status_code == 200
    assert [item["filename"] for item in marked.json()["items"]] == ["a.wav", "c.wav"]
    assert all(item["marked"] for item in marked.json()["items"])
    assert marked.json()["filtered_count"] == 2
    assert marked.json()["pending_count"] == 3
    assert [item["filename"] for item in unmarked.json()["items"]] == ["b.wav"]
    assert kept.json()["filtered_count"] == 1
    assert kept.json()["pending_count"] == 2
    assert invalid.status_code == 400


def test_dataset_review_label_filters_and_sorts_perfect_audio_last(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setitem(webui.MODEL_PATHS, "voxcpm-0.5b", model_dir)
    create_redo_test_pair(dataset_dir, "a.wav", marked=True)
    create_redo_test_pair(dataset_dir, "b.wav", perfect=True)
    create_redo_test_pair(dataset_dir, "c.wav", marked=True, perfect=True)
    create_redo_test_pair(dataset_dir, "d.wav")
    client = TestClient(webui.app)

    def filenames(label_filter):
        response = client.get(
            "/api/dataset-review",
            params={"dataset_dir": str(dataset_dir), "label_filter": label_filter},
        )
        assert response.status_code == 200
        return response.json()

    all_items = filenames("marked,perfect,other")
    assert [item["filename"] for item in all_items["items"]] == ["a.wav", "d.wav", "b.wav", "c.wav"]
    assert all_items["label_filter"] == ["marked", "perfect", "other"]
    assert [item["filename"] for item in filenames("marked")["items"]] == ["a.wav", "c.wav"]
    assert [item["filename"] for item in filenames("perfect")["items"]] == ["b.wav", "c.wav"]
    assert [item["filename"] for item in filenames("other")["items"]] == ["d.wav"]
    assert [item["filename"] for item in filenames("marked,perfect")["items"]] == ["a.wav", "b.wav", "c.wav"]
    assert filenames("")["items"] == []


def test_dataset_review_label_cache_reuses_metadata_and_invalidates_on_file_change(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    path, metadata = create_redo_test_pair(dataset_dir, "cached.wav", marked=True)
    original_reader = webui.read_wav_metadata
    calls = []

    def tracking_reader(audio_path):
        calls.append(Path(audio_path))
        return original_reader(audio_path)

    monkeypatch.setattr(webui, "read_wav_metadata", tracking_reader)
    first = webui.get_dataset_review_index(dataset_dir, refresh=True)
    assert first.labels["cached.wav"] == (True, False)
    assert calls == [path]

    webui.DATASET_REVIEW_INDEXES.clear()
    second = webui.get_dataset_review_index(dataset_dir, refresh=True)
    assert second.labels["cached.wav"] == (True, False)
    assert calls == [path]

    webui.write_wav_metadata(path, {**metadata, "marked": True, "perfect": True})
    webui.DATASET_REVIEW_INDEXES.clear()
    changed = webui.get_dataset_review_index(dataset_dir, refresh=True)
    assert changed.labels["cached.wav"] == (True, True)
    assert calls == [path, path]


def test_generated_audio_mark_updates_cache_and_registered_export(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    export_root = tmp_path / "export"
    output_root.mkdir()
    export_root.mkdir()
    registry = tmp_path / "batch-output-directories.json"
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui, "BATCH_OUTPUT_DIRECTORY_REGISTRY", registry)
    webui.register_batch_output_directory(export_root)
    metadata = {
        "schema": "voxcpm-generation-v1",
        "created_at": "2026-07-28T18:00:00+08:00",
        "filename": "sample.wav",
        "text": "标记测试。",
    }
    for path in (output_root / "sample.wav", export_root / "sample.wav"):
        sf.write(path, np.full(800, 0.1, dtype=np.float32), 16000)
        webui.write_wav_metadata(path, metadata)

    client = TestClient(webui.app)
    marked_response = client.post(
        "/api/audio/mark",
        data={"filename": "sample.wav", "marked": "true", "exported_path": str(export_root / "sample.wav")},
    )
    marked = marked_response.json()
    assert marked_response.status_code == 200
    assert marked["marked"] is True
    assert marked["marked_at"]
    assert len(marked["updated_paths"]) == 2
    for path in (output_root / "sample.wav", export_root / "sample.wav"):
        latest = webui.read_wav_metadata(path)
        assert latest["marked"] is True
        assert latest["marked_at"] == marked["marked_at"]

    unmarked_response = client.post(
        "/api/audio/mark",
        data={"filename": "sample.wav", "marked": "false", "exported_path": str(export_root / "sample.wav")},
    )
    unmarked = unmarked_response.json()

    assert unmarked_response.status_code == 200
    assert unmarked["marked"] is False
    for path in (output_root / "sample.wav", export_root / "sample.wav"):
        latest = webui.read_wav_metadata(path)
        assert latest["marked"] is False
        assert "marked_at" not in latest


def test_generated_audio_perfect_label_updates_cache_and_registered_export(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    export_root = tmp_path / "export"
    output_root.mkdir()
    export_root.mkdir()
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui, "BATCH_OUTPUT_DIRECTORY_REGISTRY", tmp_path / "batch-output-directories.json")
    webui.register_batch_output_directory(export_root)
    metadata = {
        "schema": "voxcpm-generation-v1",
        "created_at": "2026-07-28T18:00:00+08:00",
        "text": "完美标记测试。",
    }
    paths = (output_root / "sample.wav", export_root / "sample.wav")
    for path in paths:
        sf.write(path, np.full(800, 0.1, dtype=np.float32), 16000)
        webui.write_wav_metadata(path, metadata)

    client = TestClient(webui.app)
    tagged_response = client.post(
        "/api/audio/label",
        data={"filename": "sample.wav", "label": "perfect", "enabled": "true", "exported_path": str(paths[1])},
    )
    tagged = tagged_response.json()
    assert tagged_response.status_code == 200
    assert tagged["perfect"] is True
    assert tagged["perfect_at"]
    for path in paths:
        latest = webui.read_wav_metadata(path)
        assert latest["perfect"] is True
        assert latest["perfect_at"] == tagged["perfect_at"]

    cleared_response = client.post(
        "/api/audio/label",
        data={"filename": "sample.wav", "label": "perfect", "enabled": "false", "exported_path": str(paths[1])},
    )
    assert cleared_response.status_code == 200
    for path in paths:
        latest = webui.read_wav_metadata(path)
        assert latest["perfect"] is False
        assert "perfect_at" not in latest


def test_web_session_output_updates_reuse_existing_audio_nodes():
    source = webui.WEB_PAGE.read_text(encoding="utf-8")
    render = source[source.index("function renderSessionOutputs()") : source.index("function addSessionOutputs")]
    merge = source[source.index("function addSessionOutputs") : source.index("function renderInferenceProgress")]

    assert 'existingRows.get(item.id) || createSessionOutputRow(item)' in render
    assert 'if (row !== cursor) sessionList.insertBefore(row, cursor)' in render
    assert render.index('if (!sessionOutputs.length)') < render.index('sessionList.replaceChildren()')
    assert 'if (existing) Object.assign(existing, values)' in merge
    assert "newFilenames" not in merge
    assert ".slice(0, 500)" not in merge
    assert "const SESSION_AUDIO_LIMIT = 50" in source
    assert "outputs.slice(-SESSION_AUDIO_LIMIT)" in source
    assert "playingItem && !limited.some" in source
    assert "function syncInferenceParams(status)" in source
    assert "syncedInferenceJobId === status.job_id" in source
    assert ".audio-item.playing" in source
    assert "setSessionRowPlaying(row, true)" in source
    assert 'row.setAttribute("aria-current", "true")' in source
    session_row = source[source.index("function createSessionOutputRow") : source.index("function renderSessionOutputs")]
    assert session_row.index('perfectButton.className = "perfect-audio"') < session_row.index('markButton.className = "mark-audio"')
    assert 'actions.append(perfectButton, markButton, moveButton, download)' in source
    assert 'showLabelFeedback(labelButton, item[label], label)' in source
    assert '复制的目标训练集文件夹' in source
    assert 'fetch("/api/training-datasets/copy"' in source


def test_wav_metadata_reader_uses_latest_appended_metadata(tmp_path):
    path = tmp_path / "metadata.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    webui.write_wav_metadata(path, {"created_at": "2026-07-28T18:00:00+08:00", "value": "old"})
    webui.write_wav_metadata(path, {"created_at": "2026-07-28T18:01:00+08:00", "value": "new"})

    assert webui.read_wav_metadata(path)["value"] == "new"


def test_dataset_review_redo_changes_only_seed_and_atomically_replaces_audio(monkeypatch, tmp_path):
    dataset_dir = configure_dataset_review_test(monkeypatch, tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setitem(webui.MODEL_PATHS, "voxcpm-0.5b", model_dir)
    monkeypatch.setattr(webui, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(webui, "inference_job", webui.InferenceJobRuntime())
    monkeypatch.setattr(webui, "next_dataset_review_redo_seed", lambda previous: 314159)
    original_path, original_metadata = create_redo_test_pair(
        dataset_dir,
        cfg_value=2.5,
        inference_timesteps=12,
        min_len=3,
        max_len=2048,
        normalize=True,
        optimize_requested=True,
        marked=True,
        marked_at="2026-07-28T18:02:00+08:00",
        perfect=True,
        perfect_at="2026-07-28T18:03:00+08:00",
    )
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        webui.inference_job.start(
            batch_mode=kwargs["batch_mode"],
            model=kwargs["model_key"],
            lora_name=None,
            device=kwargs["device"],
            total_tasks=1,
            total_segments=1,
        )
        generated_path = output_root / "generated.wav"
        sf.write(generated_path, np.full(1600, 0.75, dtype=np.float32), 16000)
        regenerated_metadata = {
            **original_metadata,
            "created_at": "2026-07-28T18:05:00+08:00",
            "filename": generated_path.name,
            "requested_seed": kwargs["seed"],
            "effective_seeds": [kwargs["seed"]],
        }
        webui.write_wav_metadata(generated_path, regenerated_metadata)
        webui.inference_job.record_segment()
        webui.inference_job.record_output({"filename": generated_path.name})
        webui.inference_job.finish()
        return {
            "outputs": [{"filename": generated_path.name}],
            "cancelled": False,
            "job_id": webui.inference_job.snapshot()["job_id"],
        }

    monkeypatch.setattr(webui, "generate", fake_generate)
    response = TestClient(webui.app).post(
        "/api/dataset-review/redo",
        data={"dataset_dir": str(dataset_dir), "filename": original_path.name},
    )

    assert response.status_code == 200
    assert response.json()["previous_seed"] == 42
    assert response.json()["new_seed"] == 314159
    assert response.json()["item"]["filename"] == original_path.name
    assert response.json()["item"]["redo_supported"] is True
    assert captured == {
        "text": "重做测试。",
        "model_key": "voxcpm-0.5b",
        "lora_id": "",
        "lora_strength": 1.0,
        "mode": "design",
        "control": "",
        "prompt_text": "",
        "batch_mode": "ordinary",
        "split_mode": "sentences",
        "split_value": 1,
        "batch_output_dir": "",
        "create_training_pairs": False,
        "rotate_seed": False,
        "device": "cpu",
        "cfg_value": 2.5,
        "inference_timesteps": 12,
        "min_len": 3,
        "max_len": 2048,
        "seed": 314159,
        "normalize": True,
        "denoise": False,
        "optimize": True,
        "reference_audio": None,
    }
    replaced_audio, sample_rate = sf.read(original_path, dtype="float32")
    assert sample_rate == 16000
    assert replaced_audio.shape[0] == 1600
    assert float(replaced_audio.mean()) == pytest.approx(0.75, abs=1e-3)
    assert original_path.with_suffix(".lab").read_text(encoding="utf-8") == "重做测试。"
    replaced_metadata = webui.read_wav_metadata(original_path)
    assert replaced_metadata["filename"] == original_path.name
    assert replaced_metadata["requested_seed"] == 314159
    assert replaced_metadata["redo_previous_seed"] == 42
    assert replaced_metadata["marked"] is True
    assert replaced_metadata["marked_at"] == "2026-07-28T18:02:00+08:00"
    assert replaced_metadata["perfect"] is True
    assert replaced_metadata["perfect_at"] == "2026-07-28T18:03:00+08:00"
    assert not (output_root / "generated.wav").exists()
    assert webui.inference_job.snapshot()["status"] == "idle"


def test_batch_output_directory_registry_persists_last_selection(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = tmp_path / "batch-output-directories.json"
    monkeypatch.setattr(webui, "BATCH_OUTPUT_DIRECTORY_REGISTRY", registry)

    webui.register_batch_output_directory(first)
    webui.register_batch_output_directory(second)
    webui.register_batch_output_directory(first)
    response = TestClient(webui.app).get("/api/batch-output-directories")

    assert response.status_code == 200
    assert response.json() == {
        "directories": [str(second.resolve()), str(first.resolve())],
        "default": str(first.resolve()),
    }
    assert webui.resolve_registered_batch_output_directory(str(first)) == first.resolve()


def test_browse_batch_output_directory_endpoint_registers_selection(monkeypatch, tmp_path):
    selected = tmp_path / "batch-output"
    selected.mkdir()
    monkeypatch.setattr(webui, "BATCH_OUTPUT_DIRECTORY_REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(webui, "choose_batch_output_directory", lambda initial: str(selected))

    response = TestClient(webui.app).post(
        "/api/batch-output-directories/browse",
        data={"initial_path": str(tmp_path)},
    )

    assert response.status_code == 200
    assert response.json() == {"cancelled": False, "path": str(selected.resolve())}
    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8")) == [str(selected.resolve())]


def test_browse_batch_output_directory_endpoint_handles_cancel(monkeypatch):
    monkeypatch.setattr(webui, "choose_batch_output_directory", lambda initial: None)
    response = TestClient(webui.app).post("/api/batch-output-directories/browse", data={"initial_path": ""})

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}


def test_labs_page_serves_lora_fusion_workspace():
    response = TestClient(webui.app).get("/labs")

    assert response.status_code == 200
    assert "LoRA 融合试听" in response.text
    assert "/api/generate" in response.text


def test_assembly_page_serves_component_workbench():
    response = TestClient(webui.app).get("/labs/assembly")

    assert response.status_code == 200
    assert "按当前装配推理" in response.text
    assert "AudioVAE Decoder" in response.text


@pytest.mark.parametrize("name", ["", "..", "角色/语音", "角色:语音", "CON", "LPT1.txt", "尾部."])
def test_create_training_dataset_rejects_invalid_windows_names(monkeypatch, tmp_path, name):
    monkeypatch.setattr(webui, "TRAINING_DATASET_ROOT", tmp_path)

    with pytest.raises(ValueError):
        webui.create_training_dataset(name)
