"""Configuration settings for the X-ray prohibited item detector prototype."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = MODEL_DIR / "xray_cnn_model.keras"
CLASS_NAMES = ["prohibited", "safe"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
THRESHOLD = 0.50
