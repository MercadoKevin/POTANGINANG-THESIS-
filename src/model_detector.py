"""Optional model-backed detector placeholder.

This file is designed so a trained ONNX model can later be plugged into the
same live interface without rewriting the application.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from detector import BaseDetector, DetectionResult


class ModelDetector(BaseDetector):
    """Optional detector wrapper for a trained model file."""

    name = "model"

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.available = False
        self.net = None

        if self.model_path.exists():
            try:
                self.net = cv2.dnn.readNet(str(self.model_path))
                self.available = True
            except Exception:
                self.available = False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.available or self.net is None:
            return DetectionResult(
                suspicious=False,
                confidence=0.0,
                label="Model Unavailable",
                reason="No trained model file loaded",
                bbox=None,
            )

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255.0, size=(224, 224), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward().flatten()
        confidence = float(output[0]) if output.size else 0.0
        suspicious = confidence >= 0.5

        return DetectionResult(
            suspicious=suspicious,
            confidence=confidence,
            label="Model Threat" if suspicious else "Model Safe",
            reason="Prediction from trained detector",
            bbox=None,
        )
