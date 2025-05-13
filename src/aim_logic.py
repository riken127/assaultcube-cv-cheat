import pyautogui
import time
from src.mouse_control import move_mouse, click_mouse
from src.config import AIM_DELAY, SENSITIVITY_SCALE

def aim(x, y, delay=AIM_DELAY, sensitivity_scale=SENSITIVITY_SCALE):
    current_x, current_y = pyautogui.position()
    dx = int((x - current_x) * sensitivity_scale)
    dy = int((y - current_y) * sensitivity_scale)
    move_mouse(dx, dy)
    time.sleep(delay)

def shoot():
    click_mouse()
    time.sleep(0.1)