# rlhf_phase1.py
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
import mlflow
import mlflow.pytorch
import sys

device = 'cpu'
print(f"Device: {device}")

# Hyperparameters
epsilon = 0.99                       # Initial exploration rate (probability of choosing a random action)
epsilon_decay = 0.995                # Decay rate for the exploration probability after each episode
min_epsilon = 0.05                   # Minimum exploration rate to ensure some exploration continues
gamma = 0.99                         # Discount factor for future rewards in Q-learning
replay_buffer = deque(maxlen=10000)  # Buffer to store past experiences for training, (Max 10,000 experiences)
batch_size = 32                      # Number of experiences sampled from the replay buffer for training
num_episodes = 500                   # Amount of times the model will go through the training loop
print("Model parameters set!")

# Set system path to project
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.rlhf_utils import open_emulator, perform_action, capture_state, phase1_reward
from RLHF_Scripts.modular_scripts.load_model import load_phase_1

model, optimizer = load_phase_1()
criterion = nn.MSELoss()
model.eval()
print("Gameplay model successfully loaded!")

# Open Gameplay Functions
from RLHF_Scripts.modular_scripts.rlhf_utils import open_emulator, perform_action, capture_state, phase1_reward

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

# Open the emulator
print('Opening emulator...')
open_emulator()
print('Emulator opened!')

# Training loop:
# 1. Loop through episodes
# 2. Captures current state/annotatesfri
# 3. Determines action (epsilon greedy policy)
# 4. Performs action
# 5. Captures next state, checks if done
# 6. Appends replay buffer
# 7. Sample from replay buffer for stabilization
# 8. Epsilon Decay

# Slight delay to ensure emulator is in correct window
time.sleep(2)

# Start MLflow experiment
mlflow.set_experiment("Pokemon_Platinum_AI_Phase1")

print("Beginning training...")
# Set MLflow tracking
with mlflow.start_run():
    mlflow.log_param("epsilon", epsilon)
    mlflow.log_param("epsilon_decay", epsilon_decay)
    mlflow.log_param("min_epsilon", min_epsilon)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_episodes", num_episodes)

    # Loop over num_episodes
    for episode in range(num_episodes):   # Looping over num_episodes
        state = capture_state().convert('RGB').resize((640, 640))
        state = transforms.ToTensor()(state).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        j = 0

        pyautogui.press('5') # Restart from Jubilife City
        while not done:
            j += 1
            if np.random.rand() <= epsilon:
                print("Using exploration")
                # Chooses from the 
                action = np.random.choice(list(ACTION_MAP_DIALOGUE.keys())) 
            else:
                print("Using exploitation")
                q_values = model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()
                action = REVERSED_ACTION_MAPPING[action]
            
            # Map the key onto what the user would understand in actual console gameplay
            print(f"Action: {ACTION_MAP_DIALOGUE[action]}")

            # Use .keyDown/.keyUp as opposed to .press as emulator might not detect input
            pyautogui.keyDown(action)
            pyautogui.keyUp(action)

            next_state = capture_state().convert('RGB').resize((640, 640))
            next_state = transforms.ToTensor()(next_state).unsqueeze(0).to(device)
            reward, done = phase1_reward(screenshot=next_state)
            episode_reward += reward

            # Mapping back to action index as this is the format the model understands
            action_index = ACTION_MAPPING[action]

            replay_buffer.append((state, action_index, reward, next_state, done))
            state = next_state

            if len(replay_buffer) > batch_size and j>=50:
                print("Entering replay buffer") # Problem point: code stops executing emulator actions; solution: checking problem location
                minibatch = random.sample(replay_buffer, batch_size)
                print("Minibatch Sampled")
                for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                    try:
                        print(f"Minibatch item {i}/{batch_size} processing...")
                        state, next_state = state.unsqueeze(0), next_state.unsqueeze(0)
                        target = reward
                        if not done:
                            target += gamma * torch.max(model(next_state))

                        target_f= model(state)   
                        target_f[0][action_index] = target

                        # Train the modelf
                        optimizer.zero_grad()
                        loss = criterion(model(state), target_f)
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error processing minibatch item {i}: {e}")
                
                # Reset action counter at end of replay buffer
                j=0

        # Decay epsilon at end of each episode
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        mlflow.log_metric("epsilon", epsilon, step=episode)

    # Save the model to MLflow
    model_path = "models/route203_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")

print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")