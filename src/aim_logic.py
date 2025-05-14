import pyautogui # used here to retrieve the current mouse position
import time
from src.mouse_control import move_mouse, click_mouse
from src.config import AIM_DELAY, SENSITIVITY_SCALE

# This function simulates aiming at a specific screen coordinate.
def aim(x, y, delay=AIM_DELAY, sensitivity_scale=SENSITIVITY_SCALE):
    # uses pyautogui to obtain the current mouse position
    current_x, current_y = pyautogui.position()

    # computes the difference between the current mouse position and the target coordinates.
    # then multiplies these by the sensitivity scale to fine-tune movement.
    # this allows matching the in-game sensitivity.
    # casts the results to an int value because move mouse works on round numbers.
    dx = int((x - current_x) * sensitivity_scale)
    dy = int((y - current_y) * sensitivity_scale)
    
    # calls the win32 based function to move the mouse
    move_mouse(dx, dy) 

    # sleeps for a certain delay
    time.sleep(delay)

# This function simulates a mouse click, firing a weapon inside assault cube.
def shoot():
    # Calls the custom win32 based function.
    click_mouse()
    # gives time for the system or application to register input
    time.sleep(0.1)