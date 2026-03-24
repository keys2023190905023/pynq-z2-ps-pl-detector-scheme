from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import QuantizedConvConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_preset_path() -> Path:
    return _repo_root() / "configs" / "presets.json"


def default_overlay_path() -> Path:
    candidates = [
        _repo_root() / "hardware" / "overlay" / "yolo_pynq_z2.bit",
        _repo_root() / "hardware" / "yolo_pynq_z2.bit",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_presets(path: Path | str | None = None) -> Dict[str, QuantizedConvConfig]:
    preset_path = Path(path) if path is not None else default_preset_path()
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    return {name: QuantizedConvConfig.from_dict(item) for name, item in raw.items()}


def load_preset(name: str, path: Path | str | None = None) -> QuantizedConvConfig:
    presets = load_presets(path)
    try:
        return presets[name]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise KeyError(f"preset '{name}' not found. available: {available}") from exc
