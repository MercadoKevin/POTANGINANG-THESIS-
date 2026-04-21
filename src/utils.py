"""Utility functions for preprocessing, evaluation, and display."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def load_and_prepare_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and convert it into a model-ready tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def probability_to_label(probability: float, threshold: float = 0.50) -> Tuple[str, float]:
    """Convert model probability into a human-readable label.

    Model output is interpreted as probability of the `safe` class.
    """
    if probability >= threshold:
        return "SAFE", probability
    return "PROHIBITED ITEM DETECTED", 1.0 - probability
