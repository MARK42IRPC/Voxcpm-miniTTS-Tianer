from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "pretrained_models"
PIPER_MODEL_ROOT = ROOT / "piper" / "models"
MELO_BASE_ROOT = ROOT / "piper" / "melo-bases" / "MeloTTS-Chinese"

VOXCPM_MODELS = {
    "voxcpm-0.5b": {
        "repo": "openbmb/VoxCPM-0.5B",
        "destination": MODEL_ROOT / "VoxCPM-0.5B",
        "required": ("config.json", "pytorch_model.bin", "audiovae.pth", "tokenizer.json"),
    },
    "voxcpm1.5": {
        "repo": "openbmb/VoxCPM1.5",
        "destination": MODEL_ROOT / "VoxCPM1.5",
        "required": ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"),
    },
    "voxcpm2": {
        "repo": "openbmb/VoxCPM2",
        "destination": MODEL_ROOT / "VoxCPM2",
        "required": ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"),
    },
}

ZIPENHANCER = {
    "repo": "iic/speech_zipenhancer_ans_multiloss_16k_base",
    "destination": MODEL_ROOT / "ZipEnhancer",
    "required": ("configuration.json", "pytorch_model.bin"),
}

PIPER_VOICES = {
    "huayan-x-low": ("huayan", "x_low", "zh_CN-huayan-x_low"),
    "huayan-medium": ("huayan", "medium", "zh_CN-huayan-medium"),
    "xiao-ya-medium": ("xiao_ya", "medium", "zh_CN-xiao_ya-medium"),
    "chaowen-medium": ("chaowen", "medium", "zh_CN-chaowen-medium"),
}

PROFILE_COMPONENTS = {
    "none": (),
    "lite": ("voxcpm-0.5b", "zipenhancer", "huayan-x-low"),
    "recommended": ("voxcpm2", "voxcpm-0.5b", "zipenhancer", "huayan-x-low", "chaowen-medium"),
    "full": (
        "voxcpm2",
        "voxcpm1.5",
        "voxcpm-0.5b",
        "zipenhancer",
        "huayan-x-low",
        "huayan-medium",
        "xiao-ya-medium",
        "chaowen-medium",
        "melo-base",
    ),
}


def _complete(destination: Path, required: tuple[str, ...]) -> bool:
    return all((destination / name).is_file() and (destination / name).stat().st_size > 0 for name in required)


def _install_huggingface_snapshot(name: str, spec: dict, force: bool) -> None:
    destination = spec["destination"]
    if not force and _complete(destination, spec["required"]):
        print(f"[skip] {name}: already installed at {destination}", flush=True)
        return

    from huggingface_hub import snapshot_download

    print(f"[download] {name}: {spec['repo']}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec["repo"], local_dir=destination, max_workers=4)
    if not _complete(destination, spec["required"]):
        raise RuntimeError(f"{name} download completed but required files are missing: {destination}")


def _install_zipenhancer(force: bool) -> None:
    destination = ZIPENHANCER["destination"]
    if not force and _complete(destination, ZIPENHANCER["required"]):
        print(f"[skip] ZipEnhancer: already installed at {destination}", flush=True)
        return

    from modelscope import snapshot_download

    print(f"[download] ZipEnhancer: {ZIPENHANCER['repo']}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(ZIPENHANCER["repo"], local_dir=str(destination))
    if not _complete(destination, ZIPENHANCER["required"]):
        raise RuntimeError(f"ZipEnhancer download completed but required files are missing: {destination}")


def _install_piper_voice(component: str, force: bool) -> None:
    speaker, quality, model_name = PIPER_VOICES[component]
    destination = PIPER_MODEL_ROOT / model_name
    required = (f"{model_name}.onnx", f"{model_name}.onnx.json")
    if not force and _complete(destination, required):
        print(f"[skip] Piper {model_name}: already installed", flush=True)
        return

    from huggingface_hub import hf_hub_download

    print(f"[download] Piper {model_name}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    source_root = f"zh/zh_CN/{speaker}/{quality}"
    for filename in required:
        cached_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=f"{source_root}/{filename}")
        shutil.copy2(cached_path, destination / filename)
    if not _complete(destination, required):
        raise RuntimeError(f"Piper voice download completed but required files are missing: {destination}")


def _install_melo_base(force: bool) -> None:
    spec = {
        "repo": "myshell-ai/MeloTTS-Chinese",
        "destination": MELO_BASE_ROOT,
        "required": ("checkpoint.pth", "config.json"),
    }
    _install_huggingface_snapshot("MeloTTS Chinese/English training base", spec, force)


def install_component(component: str, force: bool) -> None:
    if component in VOXCPM_MODELS:
        _install_huggingface_snapshot(component, VOXCPM_MODELS[component], force)
    elif component == "zipenhancer":
        _install_zipenhancer(force)
    elif component in PIPER_VOICES:
        _install_piper_voice(component, force)
    elif component == "melo-base":
        _install_melo_base(force)
    else:
        raise ValueError(f"Unknown model component: {component}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install local models used by the VoxCPM miniTTS WebUI")
    parser.add_argument("--profile", choices=PROFILE_COMPONENTS, default="recommended")
    parser.add_argument("--force", action="store_true", help="Revalidate and redownload model files")
    parser.add_argument("--dry-run", action="store_true", help="Print the model plan without downloading")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint or mirror")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    components = PROFILE_COMPONENTS[args.profile]
    print(f"Model profile: {args.profile}", flush=True)
    if not components:
        print("No model downloads requested.", flush=True)
        return
    for index, component in enumerate(components, 1):
        print(f"[{index}/{len(components)}] {component}", flush=True)
        if not args.dry_run:
            install_component(component, args.force)
    if args.dry_run:
        print("Dry run complete; no files were changed.", flush=True)
    else:
        print("Model installation complete.", flush=True)


if __name__ == "__main__":
    main()
