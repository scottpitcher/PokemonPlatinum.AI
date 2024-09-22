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
epsilon = 0.75                       # Initial exploration rate (probability of choosing a random action); low due to human feedback mechanisms
epsilon_decay = 0.995                # Decay rate for the exploration probability after each episode
min_epsilon = 0.05                   # Minimum exploration rate to ensure some exploration continues
gamma = 0.99                         # Discount factor for future rewards in Q-learning
replay_buffer = deque(maxlen=10000)  # Large buffer to stabilize learning
short_term_buffer = deque(maxlen=20) # Short-term buffer to prioritize recent experiences
batch_size = 4                       # Number of experiences sampled from the replay buffer for training
num_episodes = 500                   # Amount of times the model will go through the training loop
print("Model hyperparameters set!")

# Set system path to project
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.rlhf_utils import (
    open_emulator, capture_state, phase1_reward, ACTION_MAPPING, REVERSED_ACTION_MAPPING, ACTION_MAP_DIALOGUE,get_human_feedback
)
from RLHF_Scripts.modular_scripts.load_model import load_phase_1

# Setting directory for human reviewing
review_dir = "RLHF_Scripts/human_review_logs"

# Loading in model state from the model pretraining on game data
model, optimizer = load_phase_1()
criterion = nn.MSELoss()
model.eval()
print("Gameplay model state successfully loaded!")

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


# TRAINING LOOP
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
    for episode in range(num_episodes):

        # Capture initial state
        state = capture_state().convert('RGB').resize((640, 640))
        state = transforms.ToTensor()(state).unsqueeze(0).to(device)
        
        # Initialise episode variables
        done = False
        episode_reward = 0
        j = 0
        pyautogui.press('5') # Restart from Jubilife City

        # Episode Start
        while not done:
            # Step action counter
            j += 1

            # ACTION
            ## Exploration vs Exploitation
            if np.random.rand() <= epsilon:
                print("Using exploration") # Chooses from the action map keys
                action = np.random.choice(list(ACTION_MAP_DIALOGUE.keys())) 
            else:
                print("Using exploitation")
                q_values = model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()
                action = REVERSED_ACTION_MAPPING[action]
            
            ## Map the key onto what the user would understand in actual console gameplay
            print(f"Action: {ACTION_MAP_DIALOGUE[action]}")

            ## Window Focus: Bring DeSmuME back into focus for next action
            os.system("osascript -e 'tell application \"DeSmuME\" to activate'")
            ## Execute Action: Use .keyDown/.keyUp as opposed to .press as emulator might not detect input
            pyautogui.keyDown(action)
            pyautogui.keyUp(action)

            # STATE
            next_state = capture_state().convert('RGB').resize((640, 640))
            next_state = transforms.ToTensor()(next_state).unsqueeze(0).to(device)

            # HUMAN FEEDBACK/REWARD
            reward, done = phase1_reward(screenshot=next_state)
            penalty, better_action = get_human_feedback(ACTION_MAP_DIALOGUE[action])
            
            ## Use immediate training from human feedback, if negative
            if penalty < 0:
                print(f"Human disapproved the action! Applying penalty: {penalty}")
                reward += penalty  # Override/adjust reward if action was bad
                
                # If the action was bad, terrible, then the better_action will be immediately trained on ahead of the replay buffer
                if better_action:
                    # Map the better action to the model's index
                    better_action_index = ACTION_MAPPING[better_action]
                    try:
                        # Get the Q-values for the current state
                        q_values = model(state.unsqueeze(0))  # Get current Q-values for the state
                        
                        # Correct the Q-value for the better action
                        target_f = q_values.clone()  # Clone current Q-values
                        target_f[0][better_action_index] = reward  # Set the Q-value for the better action

                        # Train the model on the corrected Q-values
                        optimizer.zero_grad()
                        loss = criterion(q_values, target_f)
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error processing current item: {e}")
            
            ## Update episode reward
            episode_reward += reward

            # REPLAY BUFFER
            ## Mapping back to action index as this is the format the model understands
            action_index = ACTION_MAPPING[action]

            ## Append both replay buffers with current state
            replay_buffer.append((state, action_index, reward, next_state, done))
            short_term_buffer.append((state, action_index, reward, next_state, done))
            state = next_state

            ## Enter replay_buffer when there are enough examples, then, every 5th step
            if len(replay_buffer) > batch_size and j>=5:
                print("Entering replay buffer") 

                # Randomly choose between the smaller replay buffer (more recent memories) and larger (entire episode)
                if random.random() < 0.5:
                    minibatch = random.sample(short_term_buffer, batch_size)
                else:
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
        
        # For each episode the final state and action pair will be recorded and used for human feedback
        state_image_path = os.path.join(review_dir, f"states/final_state_episode_{episode}.png")
        state.save(state_image_path)  # Save the final state image

        log_path = os.path.join(review_dir, "final_review_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write(f"Episode {episode}, Final Action: {ACTION_MAP_DIALOGUE[action]}, Done: {done}\n")
        
        # Loggin the episode metrics to MLflow
        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        mlflow.log_metric("epsilon", epsilon, step=episode)

    # Save the model to MLflow once all episodes have been completed
    model_path = "models/route203_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")

print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")