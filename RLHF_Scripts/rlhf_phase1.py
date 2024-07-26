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
from modular_scripts.rlhf_utils import open_emulator, ACTION_MAP, perform_action, capture_state, get_feedback, check_route_203

# Open the emulator
open_emulator()

# Training loop
num_episodes = 1000
for episode in range(num_episodes):

    # Get current state of gameplay
    state = capture_state()
    state = transforms.ToTensor()(state).unsqueeze(0).to(device)
    done = False
    episode_reward = 0

    while not done:
        # Select action using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = np.random.choice(list(ACTION_MAP.values()))  # Exploration
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()  # Exploitation

        # Perform the action and capture the next state
        perform_action(action)
        next_state = capture_state()
        next_state = transforms.ToTensor()(next_state).unsqueeze(0).to(device)
        
        if check_route_203():
            # Get reward and check if the episode is done
            reward, done = get_feedback(next_state, action)
            episode_reward += reward

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Update state
        state = next_state

        # Sample a batch from the replay buffer and train the model
        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target += gamma * torch.max(model(next_state))
                target_f = model(state)
                target_f[0][action] = target

                # Train the model
                optimizer.zero_grad()
                loss = criterion(model(state), target_f)
                loss.backward()
                optimizer.step()

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

# Save the trained model
torch.save(model.state_dict(), 'models/pokemon_model_lstm.pth')