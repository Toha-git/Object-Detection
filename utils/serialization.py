"""Serialization helpers for API and dashboard responses."""

from __future__ import annotations

from typing import Any


def serialize_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert detections to JSON-safe values."""
    serialized: list[dict[str, Any]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        item: dict[str, Any] = {
            "label": str(detection["label"]),
            "confidence": float(detection["confidence"]),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        }
        if "track_id" in detection:
            item["track_id"] = int(detection["track_id"])
        serialized.append(item)
    return serialized
