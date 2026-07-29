from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]

pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

training_pkg = types.ModuleType("voxcpm.training")
training_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "training")]
sys.modules.setdefault("voxcpm.training", training_pkg)

from voxcpm.training.data import filter_dataset_by_duration

# Do not shadow the real package while the rest of the test suite imports WebUI.
sys.modules.pop("voxcpm.training", None)
sys.modules.pop("voxcpm", None)


def test_duration_filter_keeps_inclusive_configured_range():
    dataset = Dataset.from_dict(
        {
            "duration": [0.4, 1.0, 5.0, 5.1],
            "text": ["short", "minimum", "maximum", "long"],
        }
    )

    filtered = filter_dataset_by_duration(dataset, min_duration=1.0, max_duration=5.0)

    assert filtered["text"] == ["minimum", "maximum"]


def test_duration_filter_rejects_invalid_range():
    dataset = Dataset.from_dict({"duration": [1.0]})

    with pytest.raises(ValueError, match="cannot exceed"):
        filter_dataset_by_duration(dataset, min_duration=3.0, max_duration=2.0)
