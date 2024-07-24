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
model = PokemonModelLSTM(input_size, hidden_size, num_layers, num_actions).to(device)

# Load the trained model
state_dict = torch.load("models/pokemon_model_lstm.pth")
model.load_state_dict(state_dict)

print("Gameplay model successfully loaded!")
# gameplay_model.eval()