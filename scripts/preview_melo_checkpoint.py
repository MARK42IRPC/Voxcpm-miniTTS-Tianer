from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MELO_ROOT = ROOT / "third_party" / "MeloTTS"
sys.path.insert(0, str(MELO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from melo.api import TTS

    model = TTS(
        language="ZH",
        device=args.device,
        config_path=str(args.config.resolve()),
        ckpt_path=str(args.checkpoint.resolve()),
    )
    speaker_id = next(iter(model.hps.data.spk2id.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.tts_to_file(
        args.text,
        speaker_id,
        str(args.output.resolve()),
        speed=args.speed,
        quiet=True,
    )


if __name__ == "__main__":
    main()
