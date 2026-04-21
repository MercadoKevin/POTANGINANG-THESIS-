# Real-Time Prohibited Item Detection Prototype (Webcam-Based)

This repository contains a prototype thesis system for **real-time prohibited item detection** using a **webcam pointed at an X-ray monitor**. The current implementation is a **demo-first proof of concept** designed for consultation, interface demonstration, and early pipeline validation.

## Current Prototype Scope

The prototype currently supports:
- webcam-based live frame capture
- real-time frame preprocessing
- demo suspicious-item detection using heuristic screening
- alert overlay and audible alarm trigger
- modular detector pipeline for future CNN integration
- optional placeholder model loading path for trained classifiers

## Why webcam input?

Direct integration with the X-ray machine was not permitted. To keep the system non-invasive, low-cost, and feasible, the prototype uses a webcam to simulate real-time acquisition from the external X-ray monitor.

## Important note

This prototype is **not yet a validated deployment-ready detector**. The current live demo uses a heuristic detector by default so that the full pipeline can be demonstrated even before a trained CNN model is finalized. A trained model can later be plugged into the same interface.

## Project Structure

```text
xray_realtime_detector/
├── assets/
├── docs/
│   ├── consultation_script.md
│   ├── development_log.md
│   ├── methodology.md
│   ├── scope_and_limitations.md
│   └── system_architecture.md
├── src/
│   ├── app.py
│   ├── config.py
│   ├── detector.py
│   ├── heuristic_detector.py
│   ├── model_detector.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the live demo:

```bash
python src/app.py
```

3. Press:
- `q` to quit
- `a` to toggle alarm on/off
- `h` to switch to heuristic mode
- `m` to switch to model mode (if a trained model exists)

## Demo Logic

The live system captures frames from a webcam, preprocesses them, evaluates them using the active detector, and then displays:
- detector status
- confidence score
- current mode
- live bounding cue over suspicious central region
- audible alarm when the suspicious score exceeds the threshold

## Planned Next Steps

- collect or curate X-ray image dataset
- fine-tune a CNN using transfer learning
- calibrate thresholding for live monitor conditions
- improve suspicious region localization
- test under different distances and lighting conditions
- compare webcam capture vs direct video capture device input

## Consultation Positioning

You can present this as:

> A webcam-based real-time prototype for prohibited item screening that demonstrates the end-to-end pipeline for frame capture, preprocessing, inference, and alerting, with a modular architecture ready for CNN integration.
