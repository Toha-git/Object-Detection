"""Tests for alert cooldown behavior."""

from __future__ import annotations

from utils.alert import DetectionAlert


def test_alert_respects_cooldown(monkeypatch) -> None:
    """DetectionAlert should not notify repeatedly inside cooldown."""
    alert = DetectionAlert("person", cooldown_seconds=10)
    calls: list[tuple[str, float]] = []
    times = iter([100.0, 101.0, 111.0])

    monkeypatch.setattr("utils.alert.time.monotonic", lambda: next(times))
    monkeypatch.setattr(
        alert, "_notify", lambda label, confidence: calls.append((label, confidence))
    )

    detections = [{"label": "person", "confidence": 0.9, "box": (0, 0, 1, 1)}]
    alert.check(detections)
    alert.check(detections)
    alert.check(detections)

    assert calls == [("person", 0.9), ("person", 0.9)]
