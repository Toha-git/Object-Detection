"""Preprocessing helpers for frames and images."""

from __future__ import annotations

import cv2
import numpy as np


def resize_frame(
    frame: np.ndarray, width: int | None = None, height: int | None = None
) -> np.ndarray:
    """Resize a frame while preserving aspect ratio when one dimension is omitted."""
    if width is None and height is None:
        return frame

    original_height, original_width = frame.shape[:2]
    if width is None:
        assert height is not None
        ratio = height / float(original_height) if height is not None else 1.0
        dimensions = (int(original_width * ratio), int(height))
    elif height is None:
        ratio = width / float(original_width)
        dimensions = (int(width), int(original_height * ratio))
    else:
        dimensions = (int(width), int(height))

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize an image frame to float32 values in the 0 to 1 range."""
    return frame.astype(np.float32) / 255.0


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
