# pokemon_platinum_env.py

import gym
from gym import spaces
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import pyautogui
import time
import subprocess
import sys

# Import your utility functions
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.rlhf_utils import (
    open_emulator, capture_state, phase1_reward
)

class PokemonPlatinumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PokemonPlatinumEnv, self).__init__()
        
        # Action mappings
        self.ACTION_MAPPING = {
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
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.ACTION_MAPPING))
        
        # Observation space: images resized to (640, 640, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(640, 640, 3), dtype=np.uint8
        )
        
        # Load annotation model
        self.annotation_model = YOLO("runs/detect/firstRun/weights/best.pt")
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to match model input
            transforms.ToTensor()
        ])
        
        # Open emulator
        print('Opening emulator...')
        open_emulator()
        print('Emulator opened!')
        time.sleep(2)  # Ensure emulator is ready

        self.done = False
        self.state = self._get_state()

    def _get_state(self):
        """Capture and preprocess the current state."""
        screenshot = capture_state()
        screenshot = screenshot.convert('RGB').resize((640, 640))
        state = np.array(screenshot)
        return state

    def step(self, action):
        # Map action index to action key
        action_key = self.ACTION_MAPPING[action]
        
        # Perform the action
        pyautogui.keyDown(action_key)
        pyautogui.keyUp(action_key)
        time.sleep(0.1)  # Small delay to allow the emulator to process the input

        # Capture the new state
        state_image = self._get_state()
        self.state = state_image  # Update the current state
        
        # Calculate reward and done flag
        reward, self.done = phase1_reward(screenshot=Image.fromarray(state_image))
        
        # Optionally, you can include additional info
        info = {}
        
        return state_image, reward, self.done, info

    def reset(self):
        # Reset the emulator to the initial state
        pyautogui.press('5')  # Assuming '5' reloads the state
        time.sleep(1)
        self.done = False
        # Capture the initial state
        self.state = self._get_state()
        return self.state

    def render(self, mode='human'):
        # Optional: Implement rendering if needed
        pass

    def close(self):
        # Close the emulator
        pyautogui.hotkey('command', 'q')  # Adjust as per your OS
        time.sleep(1)
