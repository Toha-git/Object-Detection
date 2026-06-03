"""CSV logging for object detections."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class DetectionLogger:
    """Write detections to a CSV file in the output directory."""

    def __init__(self, output_dir: str | None = None) -> None:
        """Create a detection logger and CSV file."""
        base_dir = Path(__file__).resolve().parents[1]
        self.output_dir = Path(output_dir) if output_dir is not None else base_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"detections_{timestamp}.csv"
        self.file: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["timestamp", "frame_index", "label", "confidence", "x1", "y1", "x2", "y2"]
        )
        print(f"[INFO] Logging detections to: {self.path}")

    def log(self, frame_index: int, detections: list[dict[str, Any]]) -> None:
        """Append detections for one frame to the CSV file."""
        timestamp = datetime.now(timezone.utc).isoformat()
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            self.writer.writerow(
                [
                    timestamp,
                    frame_index,
                    detection["label"],
                    f"{float(detection['confidence']):.6f}",
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ]
            )
        self.file.flush()

    def close(self) -> None:
        """Close the underlying CSV file."""
        self.file.close()

    def __enter__(self) -> DetectionLogger:
        """Return this logger for context manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the CSV file when leaving a context manager."""
        self.close()
