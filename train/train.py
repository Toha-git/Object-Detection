"""Fine-tune a YOLOv8 model on a custom dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse YOLOv8 training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset.")
    parser.add_argument("--data", default="train/dataset.yaml", help="Path to dataset.yaml.")
    parser.add_argument(
        "--model", default="yolov8n.pt", help="Base YOLOv8 model or .pt weights path."
    )
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--imgsz", default=640, type=int, help="Training image size.")
    parser.add_argument("--batch", default=16, type=int, help="Training batch size.")
    parser.add_argument(
        "--project", default="runs/train", help="Training output project directory."
    )
    parser.add_argument("--name", default="exp", help="Training run name.")
    parser.add_argument("--device", default=None, help="Device, such as 'cpu', '0', or '0,1'.")
    parser.add_argument("--patience", default=20, type=int, help="Early stopping patience.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate training arguments."""
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32.")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1.")


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    """Run YOLOv8 fine-tuning and return training metrics."""
    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    return dict(results.results_dict) if hasattr(results, "results_dict") else {}


def main() -> int:
    """Train a custom YOLOv8 detector."""
    args = parse_args()

    try:
        validate_args(args)
        print(f"[INFO] Starting YOLOv8 training with model: {args.model}")
        print(f"[INFO] Dataset config: {args.data}")
        metrics = train_model(args)
        best_weights = Path(args.project) / args.name / "weights" / "best.pt"
        print("[INFO] Training finished.")
        print(f"[INFO] Best weights expected at: {best_weights}")
        if metrics:
            print(f"[INFO] Final metrics: {metrics}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
