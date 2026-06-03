"""Haar Cascade detector implementation using OpenCV."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class HaarCascadeDetector:
    """Lightweight face detector backed by OpenCV Haar Cascades."""

    def __init__(self, conf_threshold: float) -> None:
        """Initialize the Haar Cascade detector."""
        self.conf_threshold = conf_threshold
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError("Unable to load Haar Cascade classifier.")
        print("[INFO] Haar Cascade face detector loaded.")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        detections: list[dict[str, Any]] = []

        for x, y, width, height in faces:
            detections.append(
                {
                    "label": "face",
                    "confidence": 1.0,
                    "box": (int(x), int(y), int(x + width), int(y + height)),
                }
            )

        return detections
