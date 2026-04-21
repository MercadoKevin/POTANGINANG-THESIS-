# System Architecture

## Overview

The proposed system is a webcam-based real-time prohibited item detection prototype for X-ray monitor screening. It is composed of four major modules:

1. **Image Acquisition Module**
   - captures live frames using a webcam
   - positions the camera toward the external X-ray monitor
   - provides non-invasive real-time input

2. **Preprocessing Module**
   - frame resizing
   - grayscale preparation
   - region-of-interest extraction
   - edge and density enhancement

3. **Detection Module**
   - current stage: heuristic suspicious-pattern screening
   - future stage: CNN-based prohibited item classifier or detector
   - threshold-based decision logic

4. **Alert and Display Module**
   - displays live feed
   - overlays status, confidence, and detection box
   - triggers audible alert when suspicion threshold is exceeded

## Architecture Flow

```text
External X-ray Monitor
        ↓
Webcam Capture
        ↓
Frame Preprocessing
        ↓
Detector Engine
(Heuristic now / CNN later)
        ↓
Decision Thresholding
        ↓
Live Overlay + Alarm
```

## Deployment Rationale

The design avoids direct modification of the X-ray machine. This makes the prototype safer, lower-cost, and easier to deploy for early experimentation.
