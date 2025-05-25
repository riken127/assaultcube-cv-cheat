import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 768
GAME_WIDTH = 1366
GAME_HEIGHT = 768
OFFSET_TOP = 0
OFFSET_LEFT = 0
SENSITIVITY_SCALE = 2.6
AIM_DELAY = 0.03
CONFIDENCE = 0.5
DETECTIONS_DIR = "detections"
