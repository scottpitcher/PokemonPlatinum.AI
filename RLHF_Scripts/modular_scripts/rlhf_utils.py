import subprocess
import time
import pyautogui
import torch
from ultralytics import YOLO
from runs.detect.firstRun.weights import best # Update this to final run after training

# Load in annotation model
annotation_model = YOLO(best)

# Action map for the game key vs emulator input
ACTION_MAP = {
    'A': 'x',
    'B': 'z',
    'X': 's',
    'Y': 'a',
    'Up': 'up',
    'Down': 'down',
    'Left': 'left',
    'Right': 'right'
}

# Reward map for certain events
REWARD_MAP = {
    ""
}

#
desmume_executable = '/Applications/DeSmuME.app/Contents/MacOS/DeSmuME'
pokemon_rom = 'Platinum_Randomized.nds'
state_file = '/Users/scottpitcher/Library/Application Support/DeSmuME/0.9.13/States/Platinum_Randomized.ds4'
lua_script = 'env.lua'

def open_emulator():
    """Function to open up the emulator and start the ROM file at the specified point"""
    subprocess.Popen([desmume_executable, pokemon_rom, '--load-lua-script', lua_script])
    time.sleep(5)
    pyautogui.hotkey('fn', 'F')  # Full screen
    time.sleep(15)  # Wait for the emulator to start
    key = '4'
    time.sleep(0.2)
    pyautogui.press('x')
    time.sleep(0.2)
    pyautogui.press(key)
    print(f"Loaded state {key}")

def capture_state():
    """Capture the current state of the game"""
    left = 0
    top = 90
    width = 562
    height = 842

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    return screenshot

def check_route_203(screenshot, annotation_model):
    """Use annotation model to detect whether the Route 203 Location Pop-up is present
        i.e. has the user reached route 203?"""
    annotated_frame = annotation_model.predict(screenshot)

    at_route_203 = False
    for result in annotated_frame:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = box.cls.item()           # Get the class ID
                confidence = box.conf.item()        # Get the confidence score
                class_name = annotation_model.names[class_id]  # Map the class ID to the class name
                if class_name == "route203" and confidence > 0.85:
                    at_route_203 = True
                    break
        if at_route_203:
            break
    
    return at_route_203

def get_feedback(state, action):

    if check_route_203(state, annotation_model = annotation_model):
        return 100, True  # High reward and end episode
    return -1, False  # Small penalty for each step

# def perform_action():
#     None
