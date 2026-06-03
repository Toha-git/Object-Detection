"""Tests for detection post-processing helpers."""

from __future__ import annotations

import numpy as np

from utils.detection import crop_roi, filter_by_classes, offset_boxes, parse_classes, parse_roi


def test_parse_classes_accepts_commas_and_lists() -> None:
    """parse_classes normalizes class filters."""
    assert parse_classes("person, car") == ["person", "car"]
    assert parse_classes(["dog", ""]) == ["dog"]


def test_filter_by_classes_is_case_insensitive() -> None:
    """filter_by_classes matches labels case-insensitively."""
    detections = [{"label": "Person", "confidence": 1.0, "box": (0, 0, 1, 1)}]
    assert filter_by_classes(detections, ["person"]) == detections
    assert filter_by_classes(detections, ["car"]) == []


def test_roi_crop_and_offset() -> None:
    """ROI cropping and box offsetting keep coordinates aligned."""
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    crop, offset = crop_roi(frame, parse_roi("5,6,15,16"))
    assert crop.shape[:2] == (10, 10)
    assert offset == (5, 6)
    detections = offset_boxes([{"label": "x", "confidence": 1, "box": (0, 1, 2, 3)}], offset)
    assert detections[0]["box"] == (5, 7, 7, 9)
