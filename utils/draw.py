"""Drawing utilities for object detection annotations."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np


def color_for_label(label: str) -> tuple[int, int, int]:
    """Return a stable BGR color for a class label."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a filled label background and text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    y = max(y, text_size[1] + baseline + 4)
    cv2.rectangle(
        frame,
        (x, y - text_size[1] - baseline - 6),
        (x + text_size[0] + 8, y + baseline - 2),
        color,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame, text, (x + 4, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    backend: str,
    fps: float,
) -> np.ndarray:
    """Return an annotated copy of a frame with boxes, labels, FPS, and count."""
    annotated = frame.copy()

    for detection in detections:
        label = str(detection["label"])
        confidence = float(detection["confidence"])
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        color = color_for_label(label)
        display_label = f"{label} {confidence * 100:.1f}%"
        if "track_id" in detection:
            display_label = f"{display_label} ID:{int(detection['track_id'])}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_label(annotated, display_label, (x1, y1), color)

        trail = detection.get("trail", [])
        for point_index in range(1, len(trail)):
            previous_point = tuple(int(value) for value in trail[point_index - 1])
            current_point = tuple(int(value) for value in trail[point_index])
            cv2.line(annotated, previous_point, current_point, color, 2)

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    height = annotated.shape[0]
    cv2.putText(
        annotated,
        f"{backend.upper()} | Objects: {len(detections)}",
        (12, max(28, height - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated
