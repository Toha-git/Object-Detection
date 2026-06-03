"""Benchmark detector latency and FPS on an image or video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_detector


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark detector performance.")
    parser.add_argument("--source", required=True, help="Image or video path.")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "ssd", "haar"])
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", default=0.5, type=float)
    parser.add_argument("--frames", default=120, type=int, help="Max video frames to process.")
    return parser.parse_args()


def benchmark_image(path: str, detector: object) -> None:
    """Benchmark one image."""
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Unable to read image: {path}")
    start = time.perf_counter()
    detections = detector.detect(frame)  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - start
    print(f"[INFO] Latency: {elapsed * 1000:.2f} ms")
    print(f"[INFO] Detections: {len(detections)}")


def benchmark_video(path: str, detector: object, max_frames: int) -> None:
    """Benchmark a video stream."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    frame_count = 0
    start = time.perf_counter()
    try:
        while frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            detector.detect(frame)  # type: ignore[attr-defined]
            frame_count += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"[INFO] Frames: {frame_count}")
    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] Average latency: {(elapsed / max(frame_count, 1)) * 1000:.2f} ms")


def main() -> int:
    """Run benchmark."""
    args = parse_args()
    try:
        detector = build_detector(args.backend, args.conf, args.weights)
        suffix = Path(args.source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            benchmark_image(args.source, detector)
        else:
            benchmark_video(args.source, detector, args.frames)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
