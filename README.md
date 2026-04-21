# Automated Detection of Selected Prohibited Items in X-ray Baggage Images Using CNN

This project is a thesis prototype for detecting selected prohibited items from X-ray baggage images. The prototype is designed to run on a laptop connected to an existing X-ray monitor or using exported X-ray image files.

## Project Goal
Build a CNN-based prototype that classifies X-ray baggage images into:
- **Safe**
- **Prohibited Item Detected**

The system also includes a simple desktop alert interface for demonstration.

## Proposed System Flow
1. Capture or load an X-ray image.
2. Preprocess the image.
3. Pass the image to a CNN-based model.
4. Predict whether a prohibited item is present.
5. Show a visual alert and optional sound alert.

## Current Scope
This prototype is limited to a **binary classification** setup:
- `safe`
- `prohibited`

Future versions may extend to specific object classes such as:
- knives
- scissors
- cutters
- metal tools
- power banks

## Folder Structure
```
xray_prohibited_detector/
├── data/
│   └── sample_placeholder/
├── docs/
│   ├── consultation_notes.md
│   └── system_architecture.md
├── models/
├── src/
│   ├── config.py
│   ├── train_model.py
│   ├── predict_image.py
│   ├── app_gui.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Dataset Structure
Create a dataset folder like this before training:

```
dataset/
├── train/
│   ├── safe/
│   └── prohibited/
├── val/
│   ├── safe/
│   └── prohibited/
└── test/
    ├── safe/
    └── prohibited/
```

## Recommended Public Datasets
You can mention these during consultation:
- GDXray
- SIXray

For an initial proof of concept, a smaller manually collected image set may also be used.

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train_model.py
```

## Predicting a Single Image
```bash
python src/predict_image.py --image path/to/test_image.jpg
```

## Running the Demo App
```bash
python src/app_gui.py
```

## Method Used
- Transfer learning with **MobileNetV2**
- Binary classification output
- Visual alert based on prediction score

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Thesis Positioning
This system is intended as an **AI-assisted screening prototype** and **not a replacement for human inspection**.

## Next Improvements
- collect more X-ray images
- train on a larger balanced dataset
- move from image classification to object detection
- integrate monitor capture pipeline
- improve alarm and logging
