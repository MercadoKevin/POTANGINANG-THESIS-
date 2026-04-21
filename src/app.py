"""Main application for the webcam-based prohibited item detection prototype."""

from __future__ import annotations

import time
import cv2

from config import (
    BEEP_COOLDOWN_SECONDS,
    CAMERA_INDEX,
    COLOR_ALERT,
    COLOR_BOX,
    COLOR_INFO,
    COLOR_SAFE,
    ENABLE_BEEP,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_PATH,
    WINDOW_NAME,
)
from heuristic_detector import HeuristicDetector
from model_detector import ModelDetector
from utils import CooldownTimer, draw_label, play_beep


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

    heuristic_detector = HeuristicDetector()
    model_detector = ModelDetector(MODEL_PATH)
    current_detector = heuristic_detector

    alarm_enabled = ENABLE_BEEP
    alarm_timer = CooldownTimer(BEEP_COOLDOWN_SECONDS)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = current_detector.detect(frame)

        color = COLOR_ALERT if result.suspicious else COLOR_SAFE
        h, w = frame.shape[:2]

        if result.bbox is not None:
            x, y, bw, bh = result.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_BOX, 2)
        else:
            cv2.rectangle(frame, (int(w * 0.18), int(h * 0.12)), (int(w * 0.82), int(h * 0.84)), COLOR_BOX, 1)

        draw_label(frame, f"System: Real-Time Prohibited Item Detection", (20, 32), COLOR_INFO, 0.7)
        draw_label(frame, f"Mode: {current_detector.name.upper()}", (20, 62), COLOR_INFO)
        draw_label(frame, f"Status: {result.label}", (20, 94), color)
        draw_label(frame, f"Confidence: {result.confidence:.2f}", (20, 126), color)
        draw_label(frame, f"Reason: {result.reason}", (20, 158), COLOR_INFO, 0.52)
        draw_label(frame, "Keys: [H]euristic  [M]odel  [A]larm  [Q]uit", (20, h - 20), COLOR_INFO, 0.55)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        draw_label(frame, f"FPS: {fps:.1f}", (w - 130, 32), COLOR_INFO)

        if result.suspicious and alarm_enabled and alarm_timer.ready():
            play_beep()
            alarm_timer.trigger()

        if result.suspicious:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_ALERT, 6)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            alarm_enabled = not alarm_enabled
        if key == ord("h"):
            current_detector = heuristic_detector
        if key == ord("m"):
            current_detector = model_detector

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
