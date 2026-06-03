"""Simple SORT-style object tracker.

This implementation uses greedy IoU association and constant-position tracks.
It intentionally avoids extra dependencies while preserving the key SORT
behavior needed by this project: persistent IDs across frames and short-lived
track recovery when detections temporarily disappear.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

Box = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass
class Track:
    """Internal state for one tracked object."""

    track_id: int
    label: str
    confidence: float
    box: Box
    age: int = 0
    missed: int = 0
    hits: int = 1
    trail: Deque[Point] = field(default_factory=lambda: deque(maxlen=30))

    def update(self, detection: dict[str, Any]) -> None:
        """Update this track with a matched detection."""
        self.label = str(detection["label"])
        self.confidence = float(detection["confidence"])
        self.box = tuple(int(value) for value in detection["box"])  # type: ignore[assignment]
        self.age += 1
        self.missed = 0
        self.hits += 1
        self.trail.append(center_of_box(self.box))

    def mark_missed(self) -> None:
        """Mark this track as unmatched in the current frame."""
        self.age += 1
        self.missed += 1


def center_of_box(box: Box) -> Point:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection over union for two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class SortTracker:
    """Track detections across frames using SORT-style IoU association."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10) -> None:
        """Initialize the tracker."""
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Associate detections with tracks and return enriched detections."""
        unmatched_detection_indexes = set(range(len(detections)))
        unmatched_track_ids = set(self.tracks.keys())
        matches: list[tuple[int, int]] = []

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            detection_box = tuple(int(value) for value in detection["box"])
            detection_label = str(detection["label"])
            for track_id, track in self.tracks.items():
                if track.label != detection_label:
                    continue
                score = iou(track.box, detection_box)  # type: ignore[arg-type]
                if score >= self.iou_threshold:
                    candidates.append((score, detection_index, track_id))

        for _, detection_index, track_id in sorted(candidates, reverse=True):
            if (
                detection_index not in unmatched_detection_indexes
                or track_id not in unmatched_track_ids
            ):
                continue
            matches.append((detection_index, track_id))
            unmatched_detection_indexes.remove(detection_index)
            unmatched_track_ids.remove(track_id)

        for detection_index, track_id in matches:
            self.tracks[track_id].update(detections[detection_index])

        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            box = tuple(int(value) for value in detection["box"])
            track = Track(
                track_id=self.next_track_id,
                label=str(detection["label"]),
                confidence=float(detection["confidence"]),
                box=box,  # type: ignore[arg-type]
            )
            track.trail.append(center_of_box(track.box))
            self.tracks[self.next_track_id] = track
            matches.append((detection_index, self.next_track_id))
            self.next_track_id += 1

        for track_id in unmatched_track_ids:
            self.tracks[track_id].mark_missed()

        expired = [
            track_id for track_id, track in self.tracks.items() if track.missed > self.max_missed
        ]
        for track_id in expired:
            del self.tracks[track_id]

        tracked_detections: list[dict[str, Any]] = []
        for detection_index, track_id in matches:
            matched_track = self.tracks.get(track_id)
            if matched_track is None:
                continue
            detection = dict(detections[detection_index])
            detection["track_id"] = matched_track.track_id
            detection["trail"] = list(matched_track.trail)
            tracked_detections.append(detection)

        return tracked_detections
