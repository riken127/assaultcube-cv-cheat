import pyautogui
import mss
import numpy as np
import cv2
from src.config import GAME_WIDTH, GAME_HEIGHT, OFFSET_TOP, OFFSET_LEFT

def capture_screen():
    monitor_width, monitor_height = pyautogui.size()
    top = (monitor_height // 2) - (GAME_HEIGHT // 2) + OFFSET_TOP
    left = (monitor_width // 2) - (GAME_WIDTH // 2) + OFFSET_LEFT

    with mss.mss() as sct:
        monitor = {
            "top": top,
            "left": left,
            "width": GAME_WIDTH,
            "height": GAME_HEIGHT
        }
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, left, top, monitor["width"], monitor["height"]