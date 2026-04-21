"""Train a CNN-based X-ray bag classifier using transfer learning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
    TEST_DIR,
    THRESHOLD,
    TRAIN_DIR,
    VAL_DIR,
)


def build_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="safe_probability")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def load_dataset(directory: Path, shuffle: bool = True):
    if not directory.exists():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=shuffle,
    )


def prepare_dataset(ds):
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)


def plot_history(history, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(model: tf.keras.Model, test_ds):
    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        predictions = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().flatten().astype(int).tolist())
        y_pred.extend((predictions.flatten() >= THRESHOLD).astype(int).tolist())

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return report, matrix


def main():
    print("Loading datasets...")
    train_ds = prepare_dataset(load_dataset(TRAIN_DIR, shuffle=True))
    val_ds = prepare_dataset(load_dataset(VAL_DIR, shuffle=True))
    test_ds = prepare_dataset(load_dataset(TEST_DIR, shuffle=False))

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ]

    print("Training model...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history_plot_path = MODEL_DIR / "training_history.png"
    plot_history(history, history_plot_path)

    print("Evaluating model...")
    report, matrix = evaluate_model(model, test_ds)
    metrics_path = MODEL_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump({"classification_report": report, "confusion_matrix": matrix}, file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training plot saved to: {history_plot_path}")
    print(f"Evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
