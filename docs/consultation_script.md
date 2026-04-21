# Consultation Script

Good day, Sir/Ma'am. Our current prototype focuses on the real-time pipeline of the proposed system.

Because direct integration with the X-ray machine was not allowed, we implemented a non-invasive webcam-based acquisition setup. The webcam is positioned toward the external X-ray monitor so the system can receive continuous visual input without modifying the machine.

At the software level, the prototype already performs live frame capture, preprocessing, suspicious-pattern analysis, and alert triggering. The current detector uses a heuristic screening module so that the real-time workflow can already be demonstrated while the CNN training phase is still being finalized.

The system architecture is modular. This means the current live interface, alert system, and preprocessing pipeline can later accept a trained CNN model without changing the overall application structure.

Our next steps are to finalize the labeled X-ray dataset, train the CNN model, and compare its performance against the current baseline screening method.
