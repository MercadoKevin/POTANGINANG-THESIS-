# Consultation Script

Good day, Sir/Ma'am. Our study focuses on an AI-assisted prototype for detecting prohibited items in X-ray baggage images.

The main idea is to use a laptop-based system connected to the existing X-ray monitoring setup. Instead of adding expensive embedded hardware, the laptop handles image processing and model inference.

At the current stage, we developed an initial CNN-based prototype using transfer learning. The first version is limited to binary classification, where the system determines whether an X-ray image is safe or contains a prohibited item.

The software pipeline already includes image preprocessing, model inference, prediction output, and an alert mechanism through a desktop interface. When the system predicts a suspicious image, it displays a warning and can trigger an alarm sound.

For the next phase, we will train and validate the model using labeled X-ray baggage images and measure performance using accuracy, precision, recall, and F1-score.

This prototype is intended as a decision-support tool to assist human screeners, not replace them.
