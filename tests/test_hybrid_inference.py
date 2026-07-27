import json
from types import SimpleNamespace

import torch
from torch import nn

import voxcpm.core as core
from voxcpm.model.voxcpm2 import VoxCPM2Model, _continuation_badcase_limit


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.input_dtype = None

    def forward(self, feat):
        self.input_dtype = feat.dtype
        return feat.mean(dim=2)


def test_hybrid_fp32_feature_encoder_restores_cpu_projection_dtype():
    model = VoxCPM2Model.__new__(VoxCPM2Model)
    nn.Module.__init__(model)
    model.feat_encoder = RecordingEncoder()
    model.enc_to_lm_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.hybrid_feat_encoder = True
    model.hybrid_device = torch.device("cpu")
    model.device = "cpu"

    result = model._hybrid_feat_encode(torch.ones(1, 3, 2, 2, dtype=torch.bfloat16))

    assert model.feat_encoder.input_dtype == torch.float32
    assert result.dtype == torch.bfloat16
    assert result.shape == (1, 3, 2)


def test_continuation_badcase_limit_uses_reference_speaking_rate():
    # 33 prompt patches / 23 prompt tokens predicts about 29 target patches.
    # The pace-aware cap catches a prompt+target continuation before the old
    # generic 6x target-token threshold would.
    limit = _continuation_badcase_limit(20, 23, 33, 6.0)

    assert abs(limit - 42.17391304347826) < 1e-9


def test_stable_hybrid_keeps_encoder_on_cpu_and_compiles_only_dit(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}), encoding="utf-8")
    calls = {}

    class FakeTTSModel:
        sample_rate = 48000

        def enable_hybrid_inference(self, device, **kwargs):
            calls["hybrid"] = (device, kwargs)
            return self

        def optimize_hybrid_decoder(self):
            calls["optimized"] = True
            return self

        def generate(self, **kwargs):
            calls["warmup"] = kwargs
            return torch.zeros(1)

    def fake_from_local(path, **kwargs):
        calls["load"] = (path, kwargs)
        return FakeTTSModel()

    monkeypatch.setattr(core.VoxCPM2Model, "from_local", fake_from_local)

    pipeline = core.VoxCPM(
        voxcpm_model_path=str(tmp_path),
        enable_denoiser=False,
        optimize=True,
        device="hybrid",
    )

    assert pipeline.tts_model.sample_rate == 48000
    assert calls["load"][1]["device"] == "cpu"
    assert calls["load"][1]["optimize"] is False
    assert calls["hybrid"][0] == "cuda"
    assert calls["hybrid"][1]["accelerate_feat_encoder"] is False
    assert calls["hybrid"][1]["accelerate_vae_decoder"] is True
    assert calls["hybrid"][1]["accelerate_base_lm"] is False
    assert calls["hybrid"][1]["accelerate_residual_lm"] is False
    assert calls["optimized"] is True
    assert calls["warmup"]["max_len"] == 10


def test_hybrid_dit_compile_avoids_reduce_overhead_cuda_graph(monkeypatch):
    model = VoxCPM2Model.__new__(VoxCPM2Model)
    nn.Module.__init__(model)
    model.hybrid_device = torch.device("cpu")
    model.hybrid_decoder_optimized = False
    model.feat_decoder = nn.Module()
    model.feat_decoder.estimator = nn.Identity()
    calls = {}

    def fake_compile(module, **kwargs):
        calls.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    model.optimize_hybrid_decoder()

    assert calls == {"mode": "default", "fullgraph": True}
    assert model.hybrid_decoder_optimized is True
