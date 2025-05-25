import pydirectinput
from src.config import WALK_STEPS

# Walks forward by pressing 'w' a specified number of times.
# The default number of steps is defined in the configuration.
def walk_forward(steps=WALK_STEPS):
    for _ in range(steps):
        pydirectinput.press('w')

# Determines if the player should walk forward based on the size of detected enemies.
def should_walk(enemy_sizes, min_enemy_size):
    if enemy_sizes:
        return min(enemy_sizes) < min_enemy_size
    return False