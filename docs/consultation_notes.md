# Consultation Notes

## Working Thesis Title
Automated Detection of Selected Prohibited Items in X-ray Baggage Images Using Convolutional Neural Networks

## Problem Statement
Manual X-ray bag inspection depends heavily on operator attention, experience, and speed. This may lead to fatigue and inconsistent identification of prohibited items. There is a need for an AI-assisted prototype that can help flag suspicious X-ray images for additional review.

## Proposed Solution
Develop a CNN-based prototype that analyzes X-ray bag images and classifies them as either safe or containing prohibited items. When a suspicious image is detected, the system displays a warning and triggers an alarm sound.

## System Input
- X-ray images captured from an existing monitoring setup or loaded from image files

## System Output
- safe
- prohibited item detected
- visual warning
- optional sound alert

## Initial Method
- binary image classification
- transfer learning using MobileNetV2
- training and validation on labeled X-ray image folders

## Why Laptop-Based Setup
The prototype will run on a laptop connected to the existing X-ray monitor instead of using a Raspberry Pi camera. This reduces cost, simplifies implementation, and keeps the focus on the AI model.

## Scope
The study is limited to a prototype that detects selected prohibited items from X-ray baggage images. It is not intended to fully replace human inspectors or claim airport-level accuracy.

## Suggested Defenses During Consultation
- This is a proof-of-concept prototype.
- The system is AI-assisted, not fully autonomous.
- The initial version is binary classification for feasibility.
- The project can later expand into object detection.

## Progress You Can Honestly Say
- Proposed software architecture completed
- initial codebase prepared
- transfer learning model pipeline prepared
- desktop alert interface prepared
- dataset structure and training workflow organized
