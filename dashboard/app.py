"""Streamlit dashboard for live object detection analytics."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.haar import HaarCascadeDetector
from detector.ssd import SSDMobileNetDetector
from detector.yolo import YOLODetector
from tracker.sort_tracker import SortTracker
from utils.alert import DetectionAlert
from utils.detection import filter_by_classes, parse_classes
from utils.draw import draw_detections

LogRow = dict[str, Any]
HistoryRow = dict[str, Any]


@st.cache_resource(show_spinner=False)
def load_detector(backend: str, conf_threshold: float, weights: str) -> object:
    """Load and cache a detector for dashboard inference."""
    if backend == "yolo":
        return YOLODetector(conf_threshold=conf_threshold, weights=weights)
    if backend == "ssd":
        return SSDMobileNetDetector(conf_threshold=conf_threshold)
    if backend == "haar":
        return HaarCascadeDetector(conf_threshold=conf_threshold)
    raise ValueError(f"Unsupported backend: {backend}")


def available_weight_files() -> list[str]:
    """Return YOLO weight candidates found in common project folders."""
    candidates = ["yolov8n.pt"]
    for pattern in ("*.pt", "runs/**/*.pt", "models/**/*.pt"):
        for path in PROJECT_ROOT.glob(pattern):
            value = str(path.relative_to(PROJECT_ROOT))
            if value not in candidates:
                candidates.append(value)
    return candidates


def open_capture(
    uploaded_video: Any | None, camera_index: int
) -> tuple[cv2.VideoCapture, str | None]:
    """Open webcam capture or a temporary uploaded video file."""
    if uploaded_video is None:
        if platform.system() == "Darwin":
            return cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION), None
        return cv2.VideoCapture(camera_index), None

    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary.write(uploaded_video.read())
    temporary.flush()
    temporary.close()
    return cv2.VideoCapture(temporary.name), temporary.name


def camera_error_message() -> str:
    """Return a platform-aware camera permission message."""
    if platform.system() == "Darwin":
        return (
            "[ERROR] Unable to open webcam. On macOS, grant Camera permission to the app "
            "running Python/Streamlit, usually Terminal, iTerm, or Codex, then restart the dashboard. "
            "Go to System Settings > Privacy & Security > Camera."
        )
    return "[ERROR] Unable to open webcam or uploaded video."


def detections_to_rows(detections: list[dict[str, Any]], frame_index: int) -> list[LogRow]:
    """Convert detections into table rows."""
    rows: list[LogRow] = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    for detection in detections:
        row: LogRow = {
            "time": timestamp,
            "frame": frame_index,
            "label": detection["label"],
            "confidence": round(float(detection["confidence"]), 3),
        }
        if "track_id" in detection:
            row["track_id"] = int(detection["track_id"])
        rows.append(row)
    return rows


def read_uploaded_image(uploaded_image: Any) -> Any | None:
    """Decode an uploaded image to an OpenCV BGR frame."""
    import numpy as np

    image_array = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def render_controls(
    container: Any, prefix: str
) -> tuple[str, float, bool, str, int, Any | None, Any | None, str, str, int, bool]:
    """Render controls in a Streamlit container and return selected values."""
    container.header("Controls")
    backend = container.selectbox("Backend", ["yolo", "ssd", "haar"], key=f"{prefix}_backend")
    conf_threshold = container.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"{prefix}_confidence",
    )
    tracking_enabled = container.toggle("Tracking", value=True, key=f"{prefix}_tracking")
    weights_options = available_weight_files()
    selected_weights = container.selectbox(
        "YOLO weights file", weights_options, key=f"{prefix}_weights_file"
    )
    custom_weights = container.text_input(
        "Custom YOLO weights", value="", key=f"{prefix}_custom_weights"
    )
    weights = custom_weights.strip() or selected_weights
    camera_index = container.number_input(
        "Camera index", min_value=0, max_value=8, value=0, step=1, key=f"{prefix}_camera_index"
    )
    class_filter = container.text_input(
        "Class filter", value="", placeholder="person, car", key=f"{prefix}_class_filter"
    )
    alert_class = container.text_input(
        "Alert class", value="", placeholder="person", key=f"{prefix}_alert_class"
    )
    alert_cooldown = container.number_input(
        "Alert cooldown seconds",
        min_value=1,
        max_value=300,
        value=10,
        step=1,
        key=f"{prefix}_alert_cooldown",
    )
    uploaded_video = container.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv", "webm"], key=f"{prefix}_video"
    )
    uploaded_image = container.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key=f"{prefix}_image",
    )
    running = container.toggle("Run", value=False, key=f"{prefix}_run")
    return (
        backend,
        conf_threshold,
        tracking_enabled,
        weights,
        int(camera_index),
        uploaded_video,
        uploaded_image,
        class_filter,
        alert_class,
        int(alert_cooldown),
        running,
    )


def dashboard_controls() -> (
    tuple[str, float, bool, str, int, Any | None, Any | None, str, str, int, bool]
):
    """Render controls with a sidebar and a main-page fallback."""
    with st.sidebar:
        sidebar_values = render_controls(st, "sidebar")

    with st.expander("Controls", expanded=True):
        st.caption("Use these controls if the sidebar is hidden or collapsed.")
        main_values = render_controls(st, "main")

    return main_values if main_values[-1] else sidebar_values


def render_dashboard() -> None:
    """Render and run the Streamlit object detection dashboard."""
    st.set_page_config(
        page_title="Object Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Object Detection Dashboard")

    (
        backend,
        conf_threshold,
        tracking_enabled,
        weights,
        camera_index,
        uploaded_video,
        uploaded_image,
        class_filter,
        alert_class,
        alert_cooldown,
        running,
    ) = dashboard_controls()
    classes = parse_classes(class_filter)

    video_slot = st.empty()

    if not running:
        video_slot.info("Choose a source in Controls, then enable Run.")
        st.stop()

    metric_columns = st.columns(3)
    with metric_columns[0]:
        fps_slot = st.empty()
    with metric_columns[1]:
        total_slot = st.empty()
    with metric_columns[2]:
        source_slot = st.empty()

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Counts per Class")
        bar_slot = st.empty()
    with chart_columns[1]:
        st.subheader("Total Objects Over Time")
        line_slot = st.empty()

    st.subheader("Last 20 Detections")
    table_slot = st.empty()

    try:
        detector = load_detector(backend, conf_threshold, weights)
    except Exception as exc:
        st.error(f"[ERROR] {exc}")
        return

    tracker = SortTracker() if tracking_enabled else None
    alert = (
        DetectionAlert(alert_class, cooldown_seconds=alert_cooldown)
        if alert_class.strip()
        else None
    )

    if uploaded_image is not None:
        frame = read_uploaded_image(uploaded_image)
        if frame is None:
            st.error("[ERROR] Uploaded image could not be decoded.")
            return
        detections = detector.detect(frame)  # type: ignore[attr-defined]
        detections = filter_by_classes(detections, classes)
        if tracker is not None:
            detections = tracker.update(detections)
        if alert is not None:
            alert.check(detections)
        annotated = draw_detections(frame, detections, backend=backend, fps=0.0)
        video_slot.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True
        )
        fps_slot.metric("FPS", "0.0")
        total_slot.metric("Objects in Image", len(detections))
        source_slot.metric("Source", uploaded_image.name)
        bar_counts = Counter(str(detection["label"]) for detection in detections)
        bar_slot.bar_chart(
            {"class": list(bar_counts.keys()), "count": list(bar_counts.values())},
            x="class",
            y="count",
        )
        line_slot.line_chart(
            [{"second": datetime.now().strftime("%H:%M:%S"), "objects": len(detections)}],
            x="second",
            y="objects",
        )
        table_slot.dataframe(detections_to_rows(detections, 0), use_container_width=True)
        return

    capture, temporary_path = open_capture(uploaded_video, camera_index)
    if not capture.isOpened():
        st.error(
            camera_error_message()
            if uploaded_video is None
            else "[ERROR] Unable to open uploaded video."
        )
        return

    source_slot.metric(
        "Source", f"Webcam {camera_index}" if uploaded_video is None else uploaded_video.name
    )
    class_counts_this_second: Counter[str] = Counter()
    total_history: Deque[HistoryRow] = deque(maxlen=60)
    detection_log: Deque[LogRow] = deque(maxlen=20)
    frame_index = 0
    last_chart_update = 0.0
    previous_tick = cv2.getTickCount()

    try:
        while running:
            ok, frame = capture.read()
            if not ok:
                st.info("[INFO] Stream ended.")
                break

            current_tick = cv2.getTickCount()
            elapsed = (current_tick - previous_tick) / cv2.getTickFrequency()
            previous_tick = current_tick
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            detections = detector.detect(frame)  # type: ignore[attr-defined]
            detections = filter_by_classes(detections, classes)
            if tracker is not None:
                detections = tracker.update(detections)
            if alert is not None:
                alert.check(detections)

            for detection in detections:
                class_counts_this_second[str(detection["label"])] += 1
            for row in detections_to_rows(detections, frame_index):
                detection_log.appendleft(row)

            annotated = draw_detections(frame, detections, backend=backend, fps=fps)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_slot.image(rgb_frame, channels="RGB", use_container_width=True)
            fps_slot.metric("FPS", f"{fps:.1f}")
            total_slot.metric("Objects in Frame", len(detections))

            now = time.monotonic()
            if now - last_chart_update >= 1.0:
                total_history.append(
                    {"second": datetime.now().strftime("%H:%M:%S"), "objects": len(detections)}
                )
                bar_slot.bar_chart(
                    {
                        "class": list(class_counts_this_second.keys()),
                        "count": list(class_counts_this_second.values()),
                    },
                    x="class",
                    y="count",
                )
                line_slot.line_chart(list(total_history), x="second", y="objects")
                table_slot.dataframe(list(detection_log), use_container_width=True)
                class_counts_this_second.clear()
                last_chart_update = now

            frame_index += 1
            time.sleep(0.01)
            running = bool(st.session_state.get("Run", True))
    finally:
        capture.release()
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    render_dashboard()
