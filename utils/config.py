"""Configuration loading for object detection runtime."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "source": "webcam",
    "backend": "yolo",
    "confidence": 0.5,
    "weights": "yolov8n.pt",
    "save": False,
    "log": False,
    "headless": False,
    "tracking": {"enabled": False, "iou_threshold": 0.3, "max_missed": 10},
    "alert": {"enabled": False, "class": "person", "cooldown_seconds": 10},
    "detection": {"classes": [], "frame_skip": 1, "roi": None},
    "output": {"directory": "output"},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load project configuration from YAML and environment variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = deepcopy(DEFAULT_CONFIG)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")
        config = deep_update(config, loaded)

    env_backend = os.getenv("DETECTION_BACKEND")
    env_confidence = os.getenv("DETECTION_CONF")
    env_weights = os.getenv("YOLO_WEIGHTS")
    if env_backend:
        config["backend"] = env_backend
    if env_confidence:
        config["confidence"] = float(env_confidence)
    if env_weights:
        config["weights"] = env_weights

    return config
