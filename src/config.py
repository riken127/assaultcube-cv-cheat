import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
GAME_WIDTH = 1920
GAME_HEIGHT = 1080
OFFSET_TOP = 0
OFFSET_LEFT = 0
SENSITIVITY_SCALE = 2.6
AIM_DELAY = 0.03
CONFIDENCE = 0.5
DETECTIONS_DIR = "detections"