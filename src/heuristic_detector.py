"""Heuristic detector used for early demo and consultation presentation.

This module does not claim true prohibited item recognition. Instead, it provides
an interpretable suspicious-object screening score based on dark dense regions,
edge concentration, and compact contours in the central region of the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from detector import BaseDetector, DetectionResult
from config import (
    DARK_PIXEL_THRESHOLD,
    EDGE_THRESHOLD_HIGH,
    EDGE_THRESHOLD_LOW,
    MORPH_KERNEL_SIZE,
    ROI_H_RATIO,
    ROI_W_RATIO,
    ROI_X_RATIO,
    ROI_Y_RATIO,
    SUSPICION_THRESHOLD,
)


class HeuristicDetector(BaseDetector):
    """Live demo suspicious-item screener for webcam-fed monitor images."""

    name = "heuristic"

    def __init__(self, suspicion_threshold: float = SUSPICION_THRESHOLD) -> None:
        self.suspicion_threshold = suspicion_threshold

    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        rx = int(w * ROI_X_RATIO)
        ry = int(h * ROI_Y_RATIO)
        rw = int(w * ROI_W_RATIO)
        rh = int(h * ROI_H_RATIO)
        roi = frame[ry:ry + rh, rx:rx + rw]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH)

        dark_mask = (gray < DARK_PIXEL_THRESHOLD).astype(np.uint8) * 255
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        merged = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        merged = cv2.bitwise_or(merged, edges)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_box = None
        largest_area = 0
        compact_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(bh, 1)
            fill_ratio = area / max(bw * bh, 1)
            if 0.25 <= aspect <= 4.5 and fill_ratio > 0.22:
                compact_count += 1
                if area > largest_area:
                    largest_area = area
                    largest_box = (x + rx, y + ry, bw, bh)

        dark_ratio = float(np.mean(gray < DARK_PIXEL_THRESHOLD))
        edge_ratio = float(np.mean(edges > 0))
        contour_factor = min(compact_count / 12.0, 1.0)
        size_factor = min(largest_area / max(rw * rh * 0.18, 1), 1.0)

        confidence = (0.35 * dark_ratio) + (0.25 * edge_ratio * 4.0) + (0.20 * contour_factor) + (0.20 * size_factor)
        confidence = max(0.0, min(confidence, 1.0))
        suspicious = confidence >= self.suspicion_threshold

        if suspicious:
            label = "Suspicious Item Pattern"
            reason = "High density / edge concentration detected"
        else:
            label = "No Strong Threat Pattern"
            reason = "Below suspicion threshold"

        return DetectionResult(
            suspicious=suspicious,
            confidence=confidence,
            label=label,
            reason=reason,
            bbox=largest_box,
        )
