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

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
from models.PokemonModelLSTM import PokemonModelLSTM
# Setting Hyperparameters
num_actions = 9  # (Total Number of Actions: [A, B, X, Y, Up, Down, Left, Right, None]) (Excluding Start, Select, L, R to reduce model complexity)
input_size = 32 * 160 * 160
hidden_size = 128
num_layers = 2
num_epochs = 20

# Initialising model
from modular_scripts.load_model import load_phase_1
model = load_phase_1()
print("Gameplay model successfully loaded!")

# Open Gameplay Functions
from modular_scripts.rlhf_utils import open_emulator, ACTION_MAP, get_feedback

# Open the emulator
open_emulator()