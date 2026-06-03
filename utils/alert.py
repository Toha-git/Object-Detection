"""Desktop alert notifications for detections."""

from __future__ import annotations

import time
from typing import Any


class DetectionAlert:
    """Trigger desktop notifications when a target class is detected."""

    def __init__(self, alert_class: str, cooldown_seconds: int = 10) -> None:
        """Initialize an alert rule."""
        self.alert_class = alert_class.lower()
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0

    def check(self, detections: list[dict[str, Any]]) -> None:
        """Trigger an alert if the configured class appears in detections."""
        now = time.monotonic()
        if now - self.last_alert_time < self.cooldown_seconds:
            return

        for detection in detections:
            label = str(detection["label"]).lower()
            if label != self.alert_class:
                continue
            self._notify(str(detection["label"]), float(detection["confidence"]))
            self.last_alert_time = now
            return

    def _notify(self, label: str, confidence: float) -> None:
        """Send a desktop notification, falling back to console output."""
        message = f"Detected {label} with {confidence * 100:.1f}% confidence."
        try:
            from plyer import notification

            notification.notify(
                title="Object Detection Alert",
                message=message,
                app_name="Object Detection",
                timeout=5,
            )
            print(f"[INFO] Alert sent: {message}")
        except Exception as exc:
            print(f"[ERROR] Desktop notification failed: {exc}")
