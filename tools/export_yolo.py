"""Export YOLOv8 weights to deployment formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse export arguments."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO .pt weights path.")
    parser.add_argument(
        "--format", default="onnx", choices=["onnx", "openvino", "engine"], help="Export format."
    )
    parser.add_argument("--imgsz", default=640, type=int, help="Export image size.")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported.")
    return parser.parse_args()


def main() -> int:
    """Run YOLO export."""
    args = parse_args()
    try:
        if (
            args.weights.endswith(".pt")
            and not Path(args.weights).exists()
            and args.weights != "yolov8n.pt"
        ):
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        model = YOLO(args.weights)
        output = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
        print(f"[INFO] Exported model: {output}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
