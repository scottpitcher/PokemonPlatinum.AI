import os
import time
import json
import subprocess
import pyautogui
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import deque

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Hyperparameters
epsilon = 1.0                        # Initial exploration rate (probability of choosing a random action)
epsilon_decay = 0.995                # Decay rate for the exploration probability after each episode
min_epsilon = 0.01                   # Minimum exploration rate to ensure some exploration continues
gamma = 0.99                         # Discount factor for future rewards in Q-learning
replay_buffer = deque(maxlen=10000)  # Buffer to store past experiences for training
batch_size = 32                      # Number of experiences sampled from the replay buffer for training


# Load in initialised model
from modular_scripts.load_model import load_phase_1
model = load_phase_1()
model.eval()
print("Gameplay model successfully loaded!")

# Open Gameplay Functions
from modular_scripts.rlhf_utils import open_emulator, ACTION_MAP, capture_state, check_route_203

# Open the emulator
open_emulator()

