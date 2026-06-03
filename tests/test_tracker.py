"""Tests for SORT-style tracker helpers."""

from __future__ import annotations

from tracker.sort_tracker import SortTracker, iou


def test_iou_overlap() -> None:
    """IoU returns the expected ratio for overlapping boxes."""
    assert round(iou((0, 0, 10, 10), (5, 5, 15, 15)), 2) == 0.14


def test_tracker_keeps_id_for_matching_detection() -> None:
    """SortTracker keeps the same ID across matching frames."""
    tracker = SortTracker(iou_threshold=0.1)
    first = tracker.update([{"label": "person", "confidence": 0.9, "box": (0, 0, 10, 10)}])
    second = tracker.update([{"label": "person", "confidence": 0.8, "box": (1, 1, 11, 11)}])

    assert first[0]["track_id"] == second[0]["track_id"]
    assert len(second[0]["trail"]) == 2
