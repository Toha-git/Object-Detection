"""Tests for CSV detection logging."""

from __future__ import annotations

import csv

from utils.logger import DetectionLogger


def test_detection_logger_writes_expected_columns(tmp_path) -> None:
    """DetectionLogger writes header and detection rows."""
    logger = DetectionLogger(str(tmp_path))
    logger.log(
        3,
        [
            {
                "label": "person",
                "confidence": 0.75,
                "box": (1, 2, 3, 4),
            }
        ],
    )
    logger.close()

    with logger.path.open("r", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["timestamp", "frame_index", "label", "confidence", "x1", "y1", "x2", "y2"]
    assert rows[1][1:] == ["3", "person", "0.750000", "1", "2", "3", "4"]
