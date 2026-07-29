from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16


ROOT = Path(__file__).resolve().parents[1]
MELO_ROOT = ROOT / "third_party" / "MeloTTS"
sys.path.insert(0, str(MELO_ROOT))

from melo.models import SynthesizerTrn
from melo.text import language_id_map
from melo.utils import get_hparams_from_file


class SherpaMeloWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model
        self.lang_id = language_id_map["ZH_MIX_EN"]

    def forward(
        self,
        x,
        x_lengths,
        tones,
        sid,
        noise_scale,
        length_scale,
        noise_scale_w,
    ):
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        language = torch.zeros_like(x)
        language[:, 1::2] = self.lang_id
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tones,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0]


class NativeMeloWrapper(torch.nn.Module):
    """Keep MeloTTS frontend features and stochastic duration control as graph inputs."""

    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        tones,
        language,
        bert,
        ja_bert,
        sid,
        noise_scale,
        length_scale,
        noise_scale_w,
        sdp_ratio,
    ):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tones,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
        )[0]


def _metadata(model_path: Path, values: dict[str, Any]) -> None:
    model = onnx.load(model_path)
    while model.metadata_props:
        model.metadata_props.pop()
    for key, value in values.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(model, model_path)


def _sort_graph(graph) -> None:
    remaining = list(graph.node)
    produced = {name for node in remaining for name in node.output if name}
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(name for node in remaining for name in node.input if name and name not in produced)
    ordered = []
    while remaining:
        ready = [node for node in remaining if all(not name or name in available for name in node.input)]
        if not ready:
            raise RuntimeError("FP16 ONNX conversion produced an invalid graph topology")
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
            remaining.remove(node)
    del graph.node[:]
    graph.node.extend(ordered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--runtime", choices=("native", "sherpa"), default="native")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    precision = "int8" if args.int8 else args.precision

    hps = get_hparams_from_file(str(args.config.resolve()))
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    ).cpu()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.dec.remove_weight_norm()
    wrapper = NativeMeloWrapper(model) if args.runtime == "native" else SherpaMeloWrapper(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output.with_name(f"{args.output.stem}.fp32.onnx") if precision != "fp32" else args.output
    if precision != "fp32":
        fp32_path.unlink(missing_ok=True)
    x = torch.randint(0, 10, (1, 60), dtype=torch.int64)
    x_lengths = torch.tensor([60], dtype=torch.int64)
    tones = torch.zeros_like(x)
    language = torch.zeros_like(x)
    bert = torch.zeros((1, 1024, x.shape[1]), dtype=torch.float32)
    ja_bert = torch.zeros((1, 768, x.shape[1]), dtype=torch.float32)
    sid = torch.tensor([1], dtype=torch.int64)
    scale = torch.tensor([1.0], dtype=torch.float32)
    sdp_ratio = torch.tensor([0.2], dtype=torch.float32)
    native_inputs = (x, x_lengths, tones, language, bert, ja_bert, sid, scale, scale, scale, sdp_ratio)
    sherpa_inputs = (x, x_lengths, tones, sid, scale, scale, scale)
    export_inputs = native_inputs if args.runtime == "native" else sherpa_inputs
    if args.runtime == "native":
        input_names = [
            "x", "x_lengths", "tones", "language", "bert", "ja_bert", "sid",
            "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio",
        ]
        dynamic_axes = {
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "language": {0: "N", 1: "L"},
            "bert": {0: "N", 2: "L"},
            "ja_bert": {0: "N", 2: "L"},
            "sid": {0: "N"},
            "y": {0: "N", 1: "S", 2: "T"},
        }
    else:
        input_names = ["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"]
        dynamic_axes = {
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "y": {0: "N", 1: "S", 2: "T"},
        }
    try:
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(fp32_path),
            opset_version=18,
            dynamo=False,
            input_names=input_names,
            output_names=["y"],
            dynamic_axes=dynamic_axes,
        )
        metadata = {
            "model_type": "melo-vits-native" if args.runtime == "native" else "melo-vits",
            "engine": "melo_onnx_native" if args.runtime == "native" else "sherpa_onnx",
            "comment": "melo fine-tuned by VoxCPM distiller",
            "version": 2,
            "language": "Chinese + English",
            "add_blank": int(hps.data.add_blank),
            "n_speakers": 1,
            "jieba": 1,
            "sample_rate": hps.data.sampling_rate,
            "bert_dim": 1024,
            "ja_bert_dim": 768,
            "speaker_id": 1,
            "lang_id": language_id_map["ZH_MIX_EN"],
            "tone_start": 0,
            "url": "https://github.com/myshell-ai/MeloTTS",
            "license": "MIT",
            "description": "MeloTTS fine-tuned by VoxCPM distiller",
            "onnx.infer": "onnxruntime.quant" if precision == "int8" else "onnxruntime",
            "precision": precision,
            "sdp_ratio": 0.2,
        }
        _metadata(fp32_path, metadata)
        if precision == "int8":
            onnx_temp = Path(os.environ.get("VOXCPM_ONNX_TEMP_DIR", r"C:\tmp\voxcpm-onnx" if os.name == "nt" else "/tmp/voxcpm-onnx"))
            onnx_temp.mkdir(parents=True, exist_ok=True)
            previous_tempdir = tempfile.tempdir
            tempfile.tempdir = str(onnx_temp)
            try:
                quantize_dynamic(
                    str(fp32_path),
                    str(args.output),
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gather"],
                )
            finally:
                tempfile.tempdir = previous_tempdir
            _metadata(args.output, metadata)
        elif precision == "fp16":
            converted = convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
            _sort_graph(converted.graph)
            onnx.checker.check_model(converted)
            onnx.save(converted, args.output)
            _metadata(args.output, metadata)
    finally:
        if precision != "fp32":
            fp32_path.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "size": args.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
