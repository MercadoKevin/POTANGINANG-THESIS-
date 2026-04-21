"""Utility helpers for overlays, timing, and audio feedback."""

from __future__ import annotations

import os
import sys
import time
import cv2


def draw_label(frame, text: str, position: tuple[int, int], color: tuple[int, int, int], scale: float = 0.6) -> None:
    """Draw readable text with a dark background block."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 4, y - h - 8), (x + w + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def play_beep() -> None:
    """Best-effort cross-platform beep. Safe if audio support is missing."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1400, 250)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


class CooldownTimer:
    """Simple cooldown helper for alarms."""

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_time = 0.0

    def ready(self) -> bool:
        return (time.time() - self._last_time) >= self.cooldown_seconds

    def trigger(self) -> None:
        self._last_time = time.time()


def ensure_directory(path: str) -> None:
    """Create a directory if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)
