"""Detector interfaces and shared result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """Container for a single frame inference result."""

    suspicious: bool
    confidence: float
    label: str
    reason: str
    bbox: Optional[Tuple[int, int, int, int]] = None


class BaseDetector:
    """Base detector contract for all live detector implementations."""

    name = "base"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError
