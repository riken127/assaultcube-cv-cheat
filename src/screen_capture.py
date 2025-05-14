import pyautogui
import mss
import numpy as np
import cv2
from src.config import GAME_WIDTH, GAME_HEIGHT, OFFSET_TOP, OFFSET_LEFT

# Grabs a screenshot of a specific area centered on the screen
# that is defined by the game's width and height, with the option
# to adjust based on offsets.
def capture_screen():
    monitor_width, monitor_height = pyautogui.size()
    top = (monitor_height // 2) - (GAME_HEIGHT // 2) + OFFSET_TOP
    left = (monitor_width // 2) - (GAME_WIDTH // 2) + OFFSET_LEFT

    # open a context for screen capture using mss
    with mss.mss() as sct:

        # define the region to capture
        monitor = {
            "top": top,
            "left": left,
            "width": GAME_WIDTH,
            "height": GAME_HEIGHT
        }

        # take the screenshot
        screenshot = sct.grab(monitor)

        img = np.array(screenshot) # conver the screenshot toa numpy arrray
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, left, top, monitor["width"], monitor["height"]