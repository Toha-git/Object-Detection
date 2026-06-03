"""Flask REST API for object detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.haar import HaarCascadeDetector
from detector.ssd import SSDMobileNetDetector
from detector.yolo import YOLODetector
from utils.detection import filter_by_classes, parse_classes
from utils.serialization import serialize_detections

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "16")) * 1024 * 1024
DETECTOR_CACHE: dict[tuple[str, float, str], object] = {}


def get_detector(backend: str, conf_threshold: float, weights: str) -> object:
    """Return a cached detector for the requested backend."""
    key = (backend, conf_threshold, weights)
    if key in DETECTOR_CACHE:
        return DETECTOR_CACHE[key]

    detector: object
    if backend == "yolo":
        detector = YOLODetector(conf_threshold=conf_threshold, weights=weights)
    elif backend == "ssd":
        detector = SSDMobileNetDetector(conf_threshold=conf_threshold)
    elif backend == "haar":
        detector = HaarCascadeDetector(conf_threshold=conf_threshold)
    else:
        raise ValueError("backend must be one of: yolo, ssd, haar")

    DETECTOR_CACHE[key] = detector
    return detector


@app.get("/health")
def health() -> Any:
    """Return API health status."""
    return jsonify({"status": "ok", "service": "object-detection"})


@app.post("/detect")
def detect() -> tuple[Any, int] | Any:
    """Detect objects from an uploaded image file."""
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file with form field name 'image'."}), 400

    backend = request.form.get("backend", os.getenv("DETECTION_BACKEND", "yolo")).lower()
    conf_raw = request.form.get("conf", os.getenv("DETECTION_CONF", "0.5"))
    weights = request.form.get("weights", os.getenv("YOLO_WEIGHTS", "yolov8n.pt"))
    classes = parse_classes(request.form.get("classes", ""))

    try:
        conf_threshold = float(conf_raw)
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf must be between 0 and 1")

        uploaded_file = request.files["image"]
        if (
            uploaded_file.content_length is not None
            and uploaded_file.content_length > app.config["MAX_CONTENT_LENGTH"]
        ):
            return jsonify({"error": "Uploaded file is too large."}), 413

        image_bytes = uploaded_file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        detector = get_detector(backend=backend, conf_threshold=conf_threshold, weights=weights)
        detections = detector.detect(frame)  # type: ignore[attr-defined]
        detections = filter_by_classes(detections, classes)
        serialized = serialize_detections(detections)
        return jsonify({"detections": serialized, "count": len(serialized), "backend": backend})
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("[INFO] Starting detection API on port 5000.")
    app.run(host="0.0.0.0", port=5000)
