"""Configuration values for the real-time prohibited item detection prototype."""

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Real-Time Prohibited Item Detection Prototype"

# Demo detector parameters
SUSPICION_THRESHOLD = 0.58
DARK_PIXEL_THRESHOLD = 85
EDGE_THRESHOLD_LOW = 70
EDGE_THRESHOLD_HIGH = 160
MORPH_KERNEL_SIZE = 5

# Region of interest where the X-ray monitor is expected to appear more prominently.
ROI_X_RATIO = 0.18
ROI_Y_RATIO = 0.12
ROI_W_RATIO = 0.64
ROI_H_RATIO = 0.72

# Alarm settings
ENABLE_BEEP = True
BEEP_COOLDOWN_SECONDS = 2.0

# Overlay colors in BGR
COLOR_SAFE = (60, 180, 75)
COLOR_ALERT = (0, 0, 255)
COLOR_INFO = (255, 255, 255)
COLOR_BOX = (0, 255, 255)

# Model integration path
MODEL_PATH = "models/prohibited_item_model.onnx"
