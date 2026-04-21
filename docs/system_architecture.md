# System Architecture

## High-Level Flow
1. X-ray monitor or stored X-ray image provides image input.
2. Laptop-based application receives image.
3. Image is resized and normalized.
4. CNN model predicts the probability of a prohibited item.
5. System displays result and triggers alert if needed.

## IPO Model
### Input
- X-ray baggage image

### Process
- image preprocessing
- CNN-based classification
- threshold-based decision logic

### Output
- safe bag label
- prohibited item detected label
- alert notification

## Modules
### 1. Image Acquisition Module
Loads a bag image from file or future monitor-capture source.

### 2. Preprocessing Module
Resizes image to 224x224 and normalizes pixel values.

### 3. Detection Module
Runs the CNN-based classifier.

### 4. Alert Module
Displays a warning and triggers a bell sound for suspicious images.

### 5. Evaluation Module
Stores training history and classification metrics.

## Future Improvement Path
- real monitor feed capture
- dataset expansion
- prohibited item multiclass detection
- bounding-box object detection
- database logging of alerts
