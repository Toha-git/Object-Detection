"""Command-line entry point for real-time object detection."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2

from detector.base import Detector
from detector.haar import HaarCascadeDetector
from detector.ssd import SSDMobileNetDetector
from detector.yolo import YOLODetector
from tracker.sort_tracker import SortTracker
from utils.alert import DetectionAlert
from utils.config import load_config
from utils.detection import crop_roi, filter_by_classes, offset_boxes, parse_classes, parse_roi
from utils.draw import draw_detections
from utils.logger import DetectionLogger

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to config.yaml.")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Real-time object detection with OpenCV.", parents=[config_parser]
    )
    parser.add_argument(
        "--source", default=config["source"], help="'webcam', an image path, or a video path."
    )
    parser.add_argument(
        "--backend",
        default=config["backend"],
        choices=("yolo", "ssd", "haar"),
        help="Detector backend.",
    )
    parser.add_argument(
        "--conf",
        default=float(config["confidence"]),
        type=float,
        help="Confidence threshold from 0 to 1.",
    )
    parser.add_argument(
        "--weights", default=config["weights"], help="YOLO .pt weights path or model name."
    )
    parser.add_argument(
        "--track", action="store_true", help="Enable SORT object tracking across video frames."
    )
    parser.add_argument(
        "--alert", default=None, help="Trigger a desktop alert when this class is detected."
    )
    parser.add_argument(
        "--alert-cooldown", default=int(config["alert"]["cooldown_seconds"]), type=int
    )
    parser.add_argument(
        "--classes",
        default=",".join(config["detection"]["classes"]),
        help="Comma-separated labels to keep.",
    )
    parser.add_argument(
        "--roi", default=config["detection"]["roi"], help="Optional ROI as x1,y1,x2,y2."
    )
    parser.add_argument("--frame-skip", default=int(config["detection"]["frame_skip"]), type=int)
    parser.add_argument("--save", action="store_true", help="Save annotated output to output/.")
    parser.add_argument("--log", action="store_true", help="Write CSV detection logs to output/.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI windows.")
    parser.set_defaults(
        save=bool(config["save"]),
        log=bool(config["log"]),
        headless=bool(config["headless"]),
        track=bool(config["tracking"]["enabled"]),
    )
    args = parser.parse_args()
    args.config_data = config
    if args.alert is None and bool(config["alert"]["enabled"]):
        args.alert = config["alert"]["class"]
    return args


def build_detector(backend: str, conf_threshold: float, weights: str) -> Detector:
    """Create the requested detector instance."""
    if backend == "yolo":
        return YOLODetector(conf_threshold=conf_threshold, weights=weights)
    if backend == "ssd":
        return SSDMobileNetDetector(conf_threshold=conf_threshold)
    if backend == "haar":
        return HaarCascadeDetector(conf_threshold=conf_threshold)
    raise ValueError(f"Unsupported backend: {backend}")


def validate_confidence(conf_threshold: float) -> None:
    """Validate confidence threshold range."""
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("--conf must be a float between 0 and 1.")


def ensure_output_dir(output_directory: str = "output") -> Path:
    """Create and return the output directory."""
    output_dir = Path(output_directory)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_webcam_source(source: str) -> bool:
    """Return True when the source should use the default webcam."""
    return source.lower() == "webcam"


def source_kind(source: str) -> str:
    """Classify a source as webcam, image, or video."""
    if is_webcam_source(source):
        return "webcam"
    ext = Path(source).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported source type: {source}")


def open_video_capture(source: str) -> cv2.VideoCapture:
    """Open a webcam or video file source."""
    if is_webcam_source(source) and platform.system() == "Darwin":
        capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        capture_source: int | str = 0 if is_webcam_source(source) else source
        capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        if is_webcam_source(source) and platform.system() == "Darwin":
            raise RuntimeError(
                "Unable to open webcam. Grant Camera permission to the app running Python "
                "in System Settings > Privacy & Security > Camera, then restart."
            )
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def create_video_writer(
    output_dir: Path,
    source: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create a video writer for annotated stream output."""
    safe_name = "webcam" if is_webcam_source(source) else Path(source).stem
    output_path = output_dir / f"{safe_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), frame_size)


def process_image(
    source: str,
    detector: Detector,
    backend: str,
    save: bool,
    logger: Optional[DetectionLogger],
    headless: bool,
    alert: Optional[DetectionAlert],
    classes: list[str],
    roi: tuple[int, int, int, int] | None,
    output_directory: str,
) -> None:
    """Run detection on a single image."""
    frame = cv2.imread(source)
    if frame is None:
        raise RuntimeError(f"Unable to read image: {source}")

    input_frame, offset = crop_roi(frame, roi)
    detections = offset_boxes(detector.detect(input_frame), offset)
    detections = filter_by_classes(detections, classes)
    if alert is not None:
        alert.check(detections)
    annotated = draw_detections(frame, detections, backend=backend, fps=0.0)

    if logger is not None:
        logger.log(frame_index=0, detections=detections)

    if save:
        output_dir = ensure_output_dir(output_directory)
        output_path = output_dir / f"{Path(source).stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"[INFO] Saved annotated image: {output_path}")

    if not headless:
        cv2.imshow("Object Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[INFO] Detected {len(detections)} object(s).")


def process_stream(
    source: str,
    detector: Detector,
    backend: str,
    save: bool,
    logger: Optional[DetectionLogger],
    headless: bool,
    tracker: Optional[SortTracker],
    alert: Optional[DetectionAlert],
    classes: list[str],
    roi: tuple[int, int, int, int] | None,
    frame_skip: int,
    output_directory: str,
) -> None:
    """Run detection on a webcam or video stream."""
    capture = open_video_capture(source)
    fps_source = capture.get(cv2.CAP_PROP_FPS)
    fps_source = fps_source if fps_source and fps_source > 0 else 30.0
    writer: Optional[cv2.VideoWriter] = None
    output_dir = ensure_output_dir(output_directory) if save else None
    frame_index = 0
    previous_tick = cv2.getTickCount()
    last_detections: list[dict[str, object]] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_tick = cv2.getTickCount()
            elapsed = (current_tick - previous_tick) / cv2.getTickFrequency()
            previous_tick = current_tick
            runtime_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            should_detect = frame_index % frame_skip == 0 or not last_detections
            if should_detect:
                input_frame, offset = crop_roi(frame, roi)
                detections = offset_boxes(detector.detect(input_frame), offset)
                detections = filter_by_classes(detections, classes)
                if tracker is not None:
                    detections = tracker.update(detections)
                last_detections = detections
            else:
                detections = last_detections
            if alert is not None:
                alert.check(detections)
            annotated = draw_detections(frame, detections, backend=backend, fps=runtime_fps)

            if logger is not None:
                logger.log(frame_index=frame_index, detections=detections)

            if save and writer is None and output_dir is not None:
                height, width = annotated.shape[:2]
                writer = create_video_writer(output_dir, source, fps_source, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Unable to create video writer.")

            if writer is not None:
                writer.write(annotated)

            if not headless:
                cv2.imshow("Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped by user.")
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[INFO] Processed {frame_index} frame(s).")


def main() -> int:
    """Run the object detection CLI."""
    args = parse_args()

    try:
        validate_confidence(args.conf)
        if args.frame_skip < 1:
            raise ValueError("--frame-skip must be at least 1.")
        kind = source_kind(args.source)
        classes = parse_classes(args.classes)
        roi = parse_roi(args.roi)
        output_directory = str(args.config_data["output"]["directory"])

        if kind != "webcam" and not os.path.exists(args.source):
            raise FileNotFoundError(f"Source does not exist: {args.source}")

        print(f"[INFO] Loading backend: {args.backend}")
        if args.backend != "yolo" and args.weights != "yolov8n.pt":
            print("[INFO] --weights is only used with the YOLO backend.")
        detector = build_detector(args.backend, args.conf, args.weights)
        tracking_config = args.config_data["tracking"]
        tracker = (
            SortTracker(
                iou_threshold=float(tracking_config["iou_threshold"]),
                max_missed=int(tracking_config["max_missed"]),
            )
            if args.track
            else None
        )
        alert = (
            DetectionAlert(args.alert, cooldown_seconds=args.alert_cooldown) if args.alert else None
        )

        logger = DetectionLogger(output_directory) if args.log else None
        if kind == "image":
            process_image(
                args.source,
                detector,
                args.backend,
                args.save,
                logger,
                args.headless,
                alert,
                classes,
                roi,
                output_directory,
            )
        else:
            process_stream(
                args.source,
                detector,
                args.backend,
                args.save,
                logger,
                args.headless,
                tracker,
                alert,
                classes,
                roi,
                args.frame_skip,
                output_directory,
            )

        if logger is not None:
            logger.close()
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
