"""Tests for JSON serialization helpers."""

from __future__ import annotations

from utils.serialization import serialize_detections


def test_serialize_detections_converts_tuple_box_and_track_id() -> None:
    """serialize_detections returns JSON-safe values."""
    result = serialize_detections(
        [{"label": "dog", "confidence": 0.5, "box": (1, 2, 3, 4), "track_id": 7}]
    )
    assert result == [{"label": "dog", "confidence": 0.5, "box": [1, 2, 3, 4], "track_id": 7}]
