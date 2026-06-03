"""Detection post-processing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_classes(raw_classes: str | list[str] | None) -> list[str]:
    """Parse class filters from CLI/config values."""
    if raw_classes is None:
        return []
    if isinstance(raw_classes, list):
        return [str(value).strip() for value in raw_classes if str(value).strip()]
    return [value.strip() for value in raw_classes.split(",") if value.strip()]


def filter_by_classes(detections: list[dict[str, Any]], classes: list[str]) -> list[dict[str, Any]]:
    """Return detections whose label is in the requested class list."""
    if not classes:
        return detections
    allowed = {label.lower() for label in classes}
    return [detection for detection in detections if str(detection["label"]).lower() in allowed]


def crop_roi(
    frame: np.ndarray, roi: tuple[int, int, int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a frame to an ROI and return the crop plus coordinate offset."""
    if roi is None:
        return frame, (0, 0)

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return frame[y1:y2, x1:x2], (x1, y1)


def offset_boxes(detections: list[dict[str, Any]], offset: tuple[int, int]) -> list[dict[str, Any]]:
    """Offset detection boxes by an ROI origin."""
    offset_x, offset_y = offset
    if offset_x == 0 and offset_y == 0:
        return detections

    adjusted: list[dict[str, Any]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        copy = dict(detection)
        copy["box"] = (
            int(x1) + offset_x,
            int(y1) + offset_y,
            int(x2) + offset_x,
            int(y2) + offset_y,
        )
        adjusted.append(copy)
    return adjusted


def parse_roi(
    raw_roi: str | list[int] | tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    """Parse an ROI as x1,y1,x2,y2."""
    if raw_roi is None or raw_roi == "":
        return None
    if isinstance(raw_roi, tuple) and len(raw_roi) == 4:
        return raw_roi
    if isinstance(raw_roi, list) and len(raw_roi) == 4:
        return (int(raw_roi[0]), int(raw_roi[1]), int(raw_roi[2]), int(raw_roi[3]))
    if isinstance(raw_roi, str):
        parts = [part.strip() for part in raw_roi.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi must use x1,y1,x2,y2 format.")
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    raise ValueError("ROI must be empty or four integer coordinates.")
