# Prototype Development Log

This document summarizes the current development status of the thesis prototype.

## Completed Work
### Phase 1: Concept and Scope Definition
- Defined the problem as AI-assisted prohibited item screening in X-ray baggage images.
- Narrowed the task from full item detection to an initial binary classification prototype.
- Chose a laptop-based deployment approach for feasibility.

### Phase 2: System Planning
- Designed the IPO model of the system.
- Identified core modules: image input, preprocessing, detection, alerting, and evaluation.
- Planned dataset folder structure for training, validation, and testing.

### Phase 3: Prototype Development
- Set up Python project structure.
- Implemented transfer-learning based CNN training pipeline.
- Implemented single-image inference script.
- Implemented desktop GUI for image loading and alert display.
- Added evaluation export for metrics and confusion matrix.

### Phase 4: Initial Testing Preparation
- Prepared model storage path.
- Prepared training-history plotting.
- Prepared threshold-based safe/prohibited decision logic.

## Current Status
The system is in the **initial functional prototype stage**. The software pipeline is already prepared, but dataset collection and training using actual X-ray baggage images must still be completed for full testing.

## Remaining Tasks
- collect or organize labeled X-ray image dataset
- train the model on actual bag images
- evaluate model accuracy and error cases
- prepare screenshots of the GUI using sample test images
- explore future object-detection extension
