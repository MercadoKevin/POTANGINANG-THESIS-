"""Run inference on a single X-ray image."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config import MODEL_PATH, THRESHOLD
from utils import load_and_prepare_image, probability_to_label


def predict(image_path: str | Path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    image_tensor = load_and_prepare_image(image_path)
    safe_probability = float(model.predict(image_tensor, verbose=0)[0][0])
    label, confidence = probability_to_label(safe_probability, threshold=THRESHOLD)

    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict prohibited-item presence in an X-ray image.")
    parser.add_argument("--image", required=True, help="Path to the image file to test")
    args = parser.parse_args()
    predict(args.image)
