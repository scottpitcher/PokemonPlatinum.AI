import os

ACTION_MAPPING = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "none": 8  # Assuming 'none' is a possible action
}

actions_dir = os.listdir('annotated_images/actions')

for action in actions_dir:
    str.replace(map())