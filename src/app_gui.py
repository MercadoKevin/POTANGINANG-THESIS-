"""Simple Tkinter desktop demo for the X-ray prohibited item detector."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import tensorflow as tf

from config import IMAGE_SIZE, MODEL_PATH, THRESHOLD
from utils import load_and_prepare_image, probability_to_label


class XrayDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("X-ray Prohibited Item Detector")
        self.root.geometry("900x600")

        self.model = None
        self.image_label = None
        self.result_label = None
        self.confidence_label = None
        self.loaded_image_path = None

        self._build_ui()
        self._load_model_if_available()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Automated Prohibited Item Detector for X-ray Baggage Images",
            font=("Arial", 16, "bold"),
            pady=10,
        )
        title.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        load_btn = tk.Button(button_frame, text="Load X-ray Image", width=20, command=self.load_image)
        load_btn.grid(row=0, column=0, padx=10)

        predict_btn = tk.Button(button_frame, text="Run Detection", width=20, command=self.run_detection)
        predict_btn.grid(row=0, column=1, padx=10)

        self.image_label = tk.Label(self.root, text="No image loaded", width=80, height=20, relief="groove")
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="Result: Waiting for input", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(self.root, text="Confidence: --", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

        note = tk.Label(
            self.root,
            text="Note: This prototype is intended as an AI-assisted screening tool, not a replacement for human inspection.",
            font=("Arial", 10),
            wraplength=800,
            justify="center",
        )
        note.pack(pady=10)

    def _load_model_if_available(self):
        if MODEL_PATH.exists():
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            self.result_label.config(text="Result: Model not found. Train the model first.")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not file_path:
            return

        self.loaded_image_path = Path(file_path)
        image = Image.open(file_path).convert("RGB")
        preview = image.copy()
        preview.thumbnail((600, 350))
        photo = ImageTk.PhotoImage(preview)

        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
        self.result_label.config(text="Result: Image loaded. Ready for detection.")
        self.confidence_label.config(text="Confidence: --")

    def run_detection(self):
        if self.model is None:
            messagebox.showerror("Model Missing", "No trained model was found. Run train_model.py first.")
            return

        if self.loaded_image_path is None:
            messagebox.showwarning("No Image", "Please load an X-ray image first.")
            return

        image_tensor = load_and_prepare_image(self.loaded_image_path)
        safe_probability = float(self.model.predict(image_tensor, verbose=0)[0][0])
        label, confidence = probability_to_label(safe_probability, threshold=THRESHOLD)

        if label == "PROHIBITED ITEM DETECTED":
            self.root.bell()
            self.result_label.config(text=f"Result: {label}", fg="red")
        else:
            self.result_label.config(text=f"Result: {label}", fg="green")

        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    root = tk.Tk()
    app = XrayDetectorApp(root)
    root.mainloop()
