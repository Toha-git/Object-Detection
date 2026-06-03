"""Detector backends for the object detection project."""

from detector.haar import HaarCascadeDetector
from detector.ssd import SSDMobileNetDetector
from detector.yolo import YOLODetector

__all__ = ["YOLODetector", "SSDMobileNetDetector", "HaarCascadeDetector"]
