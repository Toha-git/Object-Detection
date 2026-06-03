"""Tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np

from utils.preprocess import bgr_to_rgb, normalize_frame, resize_frame


def test_normalize_frame_scales_to_unit_range() -> None:
    """normalize_frame returns float32 values in 0..1."""
    frame = np.array([[[0, 127, 255]]], dtype=np.uint8)
    normalized = normalize_frame(frame)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_bgr_to_rgb_swaps_channels() -> None:
    """bgr_to_rgb converts OpenCV channel order."""
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    converted = bgr_to_rgb(frame)
    assert converted.tolist() == [[[30, 20, 10]]]


def test_resize_frame_preserves_aspect_ratio() -> None:
    """resize_frame preserves aspect ratio when only width is provided."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_frame(frame, width=100)
    assert resized.shape[:2] == (50, 100)
