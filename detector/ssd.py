"""SSD MobileNet detector implementation using OpenCV DNN."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SSDMobileNetDetector:
    """Object detector backed by TensorFlow SSD MobileNet V2 through cv2.dnn."""

    COCO_LABELS = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "plate",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    DEFAULT_MODEL_URL = (
        "http://download.tensorflow.org/models/object_detection/"
        "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    )
    DEFAULT_CONFIG_URL = (
        "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/"
        "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    )

    def __init__(self, conf_threshold: float) -> None:
        """Initialize the SSD MobileNet detector."""
        self.conf_threshold = conf_threshold
        self.models_dir = Path(__file__).resolve().parents[1] / "models"
        self.weights_path = Path(
            os.getenv("SSD_MODEL_PATH", str(self.models_dir / "frozen_inference_graph.pb"))
        )
        self.config_path = Path(
            os.getenv("SSD_CONFIG_PATH", str(self.models_dir / "ssd_mobilenet_v2_coco.pbtxt"))
        )
        self._ensure_model_files()
        self.net = cv2.dnn.readNetFromTensorflow(str(self.weights_path), str(self.config_path))
        print("[INFO] SSD MobileNet V2 model loaded.")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect objects in a BGR frame."""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        detections: list[dict[str, Any]] = []

        for detection in output[0, 0]:
            confidence = float(detection[2])
            if confidence < self.conf_threshold:
                continue

            class_id = int(detection[1])
            label = (
                self.COCO_LABELS[class_id] if class_id < len(self.COCO_LABELS) else str(class_id)
            )
            x1 = max(0, min(width - 1, int(detection[3] * width)))
            y1 = max(0, min(height - 1, int(detection[4] * height)))
            x2 = max(0, min(width - 1, int(detection[5] * width)))
            y2 = max(0, min(height - 1, int(detection[6] * height)))

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                }
            )

        return detections

    def _ensure_model_files(self) -> None:
        """Ensure SSD model files exist, downloading them when explicitly enabled."""
        if self.weights_path.exists() and self.config_path.exists():
            return

        auto_download = os.getenv("SSD_AUTO_DOWNLOAD", "0").lower() in {"1", "true", "yes"}
        if not auto_download:
            raise FileNotFoundError(
                "SSD model files are missing. Place frozen_inference_graph.pb and "
                "ssd_mobilenet_v2_coco.pbtxt in object_detection/models/, set "
                "SSD_MODEL_PATH and SSD_CONFIG_PATH, or run with SSD_AUTO_DOWNLOAD=1."
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._download_ssd_files()

    def _download_ssd_files(self) -> None:
        """Download SSD weights and config to the models directory."""
        import tarfile
        import tempfile

        print("[INFO] Downloading SSD MobileNet V2 model files.")
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "ssd_mobilenet_v2_coco.tar.gz"
            urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                member = "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError("Unable to extract SSD weights from archive.")
                self.weights_path.write_bytes(extracted.read())

        urllib.request.urlretrieve(self.DEFAULT_CONFIG_URL, self.config_path)
