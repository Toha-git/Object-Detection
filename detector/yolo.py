"""YOLOv8 detector implementation using ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Object detector backed by a YOLOv8 model from the ultralytics package."""

    def __init__(self, conf_threshold: float, weights: str = "yolov8n.pt") -> None:
        """Initialize the detector with a confidence threshold."""
        self.conf_threshold = conf_threshold
        self.weights = weights
        if weights.endswith(".pt") and not Path(weights).exists() and weights != "yolov8n.pt":
            raise FileNotFoundError(f"YOLO weights file not found: {weights}")
        self.model = YOLO(weights)
        self.names = self.model.names
        print(f"[INFO] YOLOv8 model loaded: {weights}")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect objects in a BGR frame."""
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        "label": str(self.names.get(class_id, class_id)),
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    }
                )

        return detections
