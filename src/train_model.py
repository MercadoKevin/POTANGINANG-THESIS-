"""Starter training pipeline for later CNN integration.

This file is included to show the next phase of the project. It is intentionally
lightweight and can be expanded once labeled X-ray image data is ready.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("Dataset folder not found.")
        print("Create a dataset/ directory with class subfolders before training.")
        return

    print("Training scaffold ready.")
    print("Next steps:")
    print("1. Organize labeled X-ray images into class folders.")
    print("2. Add TensorFlow or PyTorch training code.")
    print("3. Export model to ONNX for live webcam inference.")


if __name__ == "__main__":
    main()
