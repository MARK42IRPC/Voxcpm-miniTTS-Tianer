from __future__ import annotations

import os
from pathlib import Path

from piper import espeakbridge
from piper.phonemize_espeak import EspeakPhonemizer


ESPEAK_DATA = Path(os.environ.get("VOXCPM_CACHE_DIR", r"C:\tmp\voxcpm")) / "piper-espeak-data"


def initialize_espeak(self, espeak_data_dir=ESPEAK_DATA) -> None:
    del self
    espeakbridge.initialize(str(ESPEAK_DATA))


EspeakPhonemizer.__init__ = initialize_espeak

from piper.train.__main__ import main


if __name__ == "__main__":
    main()
