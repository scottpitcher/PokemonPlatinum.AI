# rlhf_phase2.py
print("Script Starting...")
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
import numpy as np
import random
from PIL import Image

device = 'cpu'
print(f"Device: {device}")

# Hyperparameters
epsilon = 0.99                       # Initial exploration rate (probability of choosing a random action)
epsilon_decay = 0.995                # Decay rate for the exploration probability after each episode
min_epsilon = 0.05                   # Minimum exploration rate to ensure some exploration continues
gamma = 0.99                         # Discount factor for future rewards in Q-learning
replay_buffer = deque(maxlen=10000)  # Buffer to store past experiences for training
batch_size = 32                      # Number of experiences sampled from the replay buffer for training

# Reward map for phase 2
REWARD_MAP = {
    "Route 203": 50,
    "Battle Start": 10,
    "Attack Used":25,
    "Barry Defeated":200
}

print("Model parameters set!")

# Load in initialised model
import sys
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.rlhf_utils import open_emulator, ACTION_MAP, perform_action, capture_state, check_route_203
from RLHF_Scripts.modular_scripts.load_model import load_phase_2

model = load_phase_2()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss()
model.eval()
print("Gameplay model successfully loaded!")

# Open Gameplay Functions
from RLHF_Scripts.modular_scripts.rlhf_utils import open_emulator, perform_action, capture_state, phase2_reward
ACTION_MAPPING = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "none": 8
}

REVERSED_ACTION_MAPPING = {
    0: "a",
    1: "b",
    2: "x",
    3: "y",
    4: "up",
    5: "down",
    6: "left",
    7: "right",
    8: "none"
}

# Open the emulator
open_emulator()
print('Emulator opened!')

# Training loop:
# 1. Loop through episodes
# 2. Captures current state/annotates
# 3. Determines action (epsilon greedy policy)
# 4. Performs action
# 5. Captures next state, checks if done
# 6. Appends replay buffer
# 7. Sample from replay buffer for stabilization
# 8. Epsilon Decay

num_episodes = 1000                   # Number of training loops to reach Done
for episode in range(num_episodes):   # Looping over num_episodes
    # Get current state of gameplay
    state = capture_state().convert('RGB').resize((640, 640))
    # Make current state readable to model
    state = transforms.ToTensor()(state).unsqueeze(0).to(device)
    done = False
    episode_reward = 0
    j = 0                             # Action Counter to delay replay buffer until every 10th action

    pyautogui.press('5') # Restart from Jubilife Cityx
    while not done:
        j+=1
        print("Looping...")
        time.sleep(0.1)
        # Select action using epsilon-greedy policy
        print(f"Epsilon {epsilon}")
        if np.random.rand() <= epsilon:
            # Exploration: Choosing an action randomly
            print('using exploration')
            action = np.random.choice(list(ACTION_MAP.keys())) 
        else:
            # Exploitation: Utiling model knowledge for action choice
            print('using exploitation')
            model_state = state
            model_state = model_state.unsqueeze(0)
            q_values = model(model_state)

            action = torch.argmax(q_values).item()
            action = REVERSED_ACTION_MAPPING[action]
            

        # Perform the action and capture the next state
        perform_action(action)
        print(f"Action:{action}")
        next_state = capture_state().convert('RGB').resize((640, 640))
        next_state = transforms.ToTensor()(next_state).unsqueeze(0).to(device)
        # Getting reward
        reward, done = check_route_203(screenshot=next_state)
        episode_reward += reward
        # Store experience in replay buffer: (S0, A0, R1, S1, done)
        replay_buffer.append((state, action, reward, next_state, done))

        # Update state
        state = next_state

        # Sample a batch from the replay buffer and train the model
        if len(replay_buffer) > batch_size and j>=25:
            print("Entering replay buffer") # Problem point: code stops executing emulator actions; solution: checking problem location
            minibatch = random.sample(replay_buffer, batch_size)
            print("Minibatch Sampled")
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                print(f"Minibatch item {i}/{batch_size} processing...")
                state, next_state = state.unsqueeze(0), next_state.unsqueeze(0)
                target = reward
                if not done:
                    target += gamma * torch.max(model(next_state))
                target_f= model(state)                
                # Convert action to integer index
                action_index = ACTION_MAPPING[action]

                target_f[0][action_index] = target

                # Train the modelf
                optimizer.zero_grad()
                loss = criterion(model(state), target_f)
                loss.backward()
                optimizer.step()
                # Reset action counter
                j=0

    # Decay epsilon at episode conclusion
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

# Save the trained model
torch.save(model.state_dict(), 'models/route_203_models/route203_model.pth')