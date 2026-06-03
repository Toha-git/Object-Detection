"""Shared detector interfaces and helpers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

Detection = dict[str, Any]


class Detector(Protocol):
    """Protocol implemented by all object detector backends."""

    conf_threshold: float

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR frame."""
        ...
