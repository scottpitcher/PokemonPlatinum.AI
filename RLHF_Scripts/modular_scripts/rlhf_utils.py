# rlhf_utils.py

# Importing Packages
import subprocess
import time
import pyautogui
import torch
from ultralytics import YOLO
from PIL import Image
import sys
from torchvision import transforms
import os

device = 'cpu'

# Load in annotation model
best = "runs/detect/firstRun/weights/best.pt"
annotation_model = YOLO(best)

# Variable definitions
# Converts emulator input to model input
ACTION_MAPPING = {
    "x": 0,
    "z": 1,
    "s": 2,
    "a": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "none": 8
}

# Converts model output to emulator input
REVERSED_ACTION_MAPPING = {
    0: "x",
    1: "z",
    2: "s",
    3: "a",
    4: "up",
    5: "down",
    6: "left",
    7: "right",
    8: "none"
}

# Converts emulator input into action known to user
ACTION_MAP_DIALOGUE = {
    "x":'a',
    "z":"b",
    "s":"x",
    "a":"y",
    "up":"up",
    "down":"down",
    "left":"left",
    "right":"right",
    "none":"none"
}

# Converts back from emulator input for human feedback
REVERSED_ACTION_MAP_DIALOGUE = {
    'a': "x",
    'b': "z",
    'x': "s",
    'y': "a",
    'up': "up",
    'down': "down",
    'left': "left",
    'right': "right",
    'none': "none"
}


## Filepaths for loading in the emulator and ROM/state
desmume_executable = '/Applications/DeSmuME.app/Contents/MacOS/DeSmuME'
pokemon_rom = '/Users/scottpitcher/Downloads/PokemonRandomizer_1.10.3/Platinum_Randomized.nds'
# Functions

## Misc. Functions for Gameplay/Feedback
def open_emulator():
    """Function to open up the emulator and start the ROM file at the specified point"""
    subprocess.Popen([desmume_executable, pokemon_rom])
    time.sleep(4)
    pyautogui.hotkey('fn', 'F')  # Full screen
    time.sleep(0.5)
    key = '5'                    # Load correct game state
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

def detection_feedback():
    """Function to check if the the model detected correctly that the agent has 
       finished the current episode."""
    user_input = ''
    while user_input not in ['yes','no']:
        user_input = input("has the episode completed?")
        if user_input =='yes':
            return True
        elif user_input =='no':
            return False
        
def get_human_feedback(action):
    """Ask for human feedback after each action and get the better action if applicable."""
    feedback = ''
    while feedback not in ['good', 'bad', 'terrible']:
        feedback = input(f"Was action: {action} good, bad, or terrible? (good/bad/terrible): ")

    if feedback == 'good':
        return 0, None  # No penalty, no better action needed
    else:
        # Ask for the better action if the move was bad
        better_action = ''
        while better_action not in ACTION_MAP_DIALOGUE.values():
            better_action = input("What would have been the better action?(a/b/x/y/up/down/left/right): ")
            better_action = REVERSED_ACTION_MAP_DIALOGUE[better_action]
        
        return -20 if feedback == 'bad' else -50, better_action  # Return penalty and better action


def perform_action(action):
    """Function to utilise pyautgui to perform an action.
       Takes in the action, returns None."""
    pyautogui.press(action)
    return None

## Reward Functions
        ### Phase 1 Reward(s)
def phase1_reward(screenshot, annotation_model=annotation_model):
    """Use annotation model to detect whether the Route 203 Location Pop-up is present."""
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to a shape divisible by 32
    ])
    # Getting current state
    screenshot = transform(screenshot).to('cpu')
    annotated_frame = annotation_model.predict(screenshot)
    # Initialising variables
    reward = -1
    done = False
    for result in annotated_frame:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = box.cls.item()                      # Get the class ID
                confidence = box.conf.item()                   # Get the confidence score
                class_name = annotation_model.names[class_id]  # Map the class ID to the class name
                print(f"{class_name} has confidence {confidence}")
                # For Phase 1, reward is only detecting Route 203
                if class_name == "route203" and confidence > 0.40:
                    print("Route 203 Reached!")
                    reward, done = 100, True
                    break
        # If done, stops checking results
        if not done:
            reward, done = -1, False
        else:
            break
    return reward, done


    ### Phase 2 Reward(s)
def phase2_reward(screenshot, annotation_model=annotation_model):
    """Use annotation model to detect current state for reward and whether or not the episode is finished."""
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to a shape divisible by 32
    ])
    # Getting current state
    screenshot = transform(screenshot).to('cpu')
    annotated_frame = annotation_model.predict(screenshot)
    # Initialising variables
    reward = -1
    done = False
    for result in annotated_frame:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = box.cls.item()                      # Get the class ID
                confidence = box.conf.item()                   # Get the confidence score
                class_name = annotation_model.names[class_id]  # Map the class ID to the class name
                print(f"{class_name} has confidence {confidence}")
                # For Phase 2, rewards are the following
                if class_name == "route203" and confidence > 0.8:
                    print("Route 203 detected.")
                    reward, done = 20, False
                elif class_name == "attackUsed" and confidence > 0.8:
                    print("User used an attack!")
                    reward, done = 30, False
                elif class_name == "opponentFaint" and confidence > 0.8:
                    print("Opponent Pokémon Defeated!")
                    reward, done = 50, False
                elif class_name == "opponentDefeated" and confidence > 0.8:
                    print("Opponent was Defeated!")
                    reward, done = 200, True
        # If done, stops checking results
        if not done:
            reward, done = -1, False
        else:
            break
    return reward, done

    ### Phase 3 Reward(s)
def phase3_reward(screenshot, annotation_model=annotation_model):
    """Use annotation model to detect current state for reward and whether or not the episode is finished."""
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to a shape divisible by 32
    ])
    # Getting current state
    screenshot = transform(screenshot).to('cpu')
    annotated_frame = annotation_model.predict(screenshot)
    # Initialising variables
    reward = -1
    done = False
    for result in annotated_frame:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = box.cls.item()                      # Get the class ID
                confidence = box.conf.item()                   # Get the confidence score
                class_name = annotation_model.names[class_id]  # Map the class ID to the class name
                print(f"{class_name} has confidence {confidence}")
                # For Phase 3, rewards are the following
                if class_name == "route203" and confidence > 0.8:
                    print("Route 203 detected.")
                    reward, done = 15, False
        # If done, stops checking results
        if not done:
            reward, done = -1, False
        else:
            break
    return reward, done

