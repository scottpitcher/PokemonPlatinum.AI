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
epsilon_decay = 0.995                # Decay rate for the exploration probability after each episode
min_epsilon = 0.05                   # Minimum exploration rate to ensure some exploration continues
gamma = 0.99                         # Discount factor for future rewards in Q-learning
batch_size = 4                       # Number of experiences sampled from the replay buffer for training
num_episodes = 500                   # Amount of times the model will go through the training loop
print("Model hyperparameters set!")

# Set system path to project
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.rlhf_utils import (
    open_emulator, capture_state, phase1_reward, ACTION_MAPPING, REVERSED_ACTION_MAPPING, 
    ACTION_MAP_DIALOGUE, get_human_feedback, list_checkpoint_files,load_training_state, save_training_state,
    annotation_model_fn
)
from RLHF_Scripts.modular_scripts.load_model import load_phase_1

## Problem Point: Action space for this phase is too large for the complex environment
## Solution: Reduce action space for this phase
PRUNED_ACTIONS = ["up", "down", "left", "right"]

# Setting directory for human reviewing
review_dir = "RLHF_Scripts/human_review_logs"

# Loading in model state from the model pretraining on game data
model, optimizer = load_phase_1()
criterion = nn.MSELoss()
model.eval()
print("Gameplay model state successfully loaded!")

# Load in annotation_model
annotation_model = annotation_model_fn()

# Open the emulator
print('Opening emulator...')
open_emulator()
print('Emulator opened!')
time.sleep(2)               # Pause to allow emulator to open

# Training loop:
## 1.  Loop through episodes (starting fresh or from a checkpoint)
## 2.  Capture current state and preprocess
## 3.  Determine action (exploration or exploitation)
## 4.  Perform action via emulator input
## 5.  Capture next state and check if the episode is done
## 6.  Get human feedback; immediate training if penalties
## 7.  Append both replay buffers (short-term and long-term)
## 8.  Sample minibatch from replay buffers for model stabilization
## 9.  Epsilon Decay
## 10. Save model, buffers, and logs periodically or upon pause


# Initialize episode, buffers, and epsilon
start_episode = 0
replay_buffer = deque(maxlen=10000)  # Large buffer to stabilize learning
short_term_buffer = deque(maxlen=20) # Short-term buffer to prioritize recent experiences
epsilon = 0.75                       # Initial exploration rate (probability of choosing a random action); low due to human feedback mechanisms

# Prompt to start back training from a checkpoint
resume = ''
while resume not in ['yes','no']:
    resume = input("Would you like to resume from a checkpoint? (yes/no): ").strip().lower()


# If the user chooses to resume, list the available checkpoints
if resume == "yes":
    checkpoint_files = list_checkpoint_files()  # Get a list of checkpoint files
    if checkpoint_files:
        print("Available checkpoints:")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{i}. {file}")  # Print out each checkpoint with its corresponding index

        # Keep asking the user for valid input until they provide a correct number
        while True:
            try:
                choice = int(input("Enter the number of the checkpoint to resume from: ").strip())
                if 1 <= choice <= len(checkpoint_files):
                    selected_checkpoint = checkpoint_files[choice - 1]
                    print(f"Resuming from checkpoint: {selected_checkpoint}")
                    start_episode, replay_buffer, short_term_buffer, epsilon = load_training_state(selected_checkpoint)
                    break  # Exit the loop after a valid input
                else:
                    print("Error: Episode number out of range. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    else:
        print("No checkpoint files found. Starting from episode 0.")
else:
    print("Starting new training session from episode 0.")

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
    for episode in range(start_episode, num_episodes):
        print(f"Starting episode {episode}...")

        # Capture initial state
        state = capture_state().convert('RGB').resize((640, 640))
        state = transforms.ToTensor()(state).unsqueeze(0).to(device)
        anotations = annotation_model(state)
        
        # Initialise episode variables
        done = False            ## Rese5t Done status
        episode_reward = 0      ## Reset episode reward
        j = 0                   ## Reset action counter
        episode_data = []       ## Reset episode data for Human Feedback
        if episode > 0:         ## Restart env
            print('Restarting env...')
            os.system("osascript -e 'tell application \"DeSmuME\" to activate'")
            pyautogui.keyDown('5')
            pyautogui.keyUp('5')
        
        # EPISODE START
        while not done:
            # Step action counter
            j += 1

            # ACTION
            ## While loop to handle model processing errors
            while True:
                    # Decide once whether to use exploration or exploitation
                    if np.random.rand() <= epsilon:
                        print("Using Exploration!")
                        action = np.random.choice(PRUNED_ACTIONS)  #f Random directional action
                        break  # Exploration always succeeds, so we break out of the loop
                    else:
                        print("Using Exploitation!")
                        try:
                            q_values = model(state.unsqueeze(0))          # Assuming annotations are no longer used
                            action_index = torch.argmax(q_values).item()  # Best action based on model

                            # Map back to the pruned action space (only directional actions)
                            action = REVERSED_ACTION_MAPPING[action_index]
                            print(f"Chosen action (exploitation): {action}")

                            # Ensure that only directional actions are selected
                            if action in PRUNED_ACTIONS:
                                break  # If a valid action is chosen, break out of the loop
                            else:
                                print(f"Action '{action}' not in pruned action space, retrying...")

                        except Exception as e:
                            print(f"Error during action selection: {e}. Retrying...")
                                
            ## Map the key onto what the user understands in actual console gameplay
            print(f"Action: {ACTION_MAP_DIALOGUE[action]}")

            ## Window Focus: Bring DeSmuME back into focus for next action
            os.system("osascript -e 'tell application \"DeSmuME\" to activate'")

            ## Execute Action: Use .keyDown/.keyUp as opposed to .press as emulator might not detect input
            pyautogui.keyDown(action)
            pyautogui.keyUp(action)
            time.sleep(0.75) # Wait for state update
            # STATE
            next_state = capture_state().convert('RGB').resize((640, 640))
            next_state = transforms.ToTensor()(next_state).unsqueeze(0).to(device)

            # REWARD
            reward, done = phase1_reward(screenshot=next_state, annotation_model=annotation_model)
            
            ## Update episode reward
            episode_reward += reward

            # REPLAY BUFFER
            ## Mapping back to action index as this is the format the model understands
            action_index = ACTION_MAPPING[action]

            ## Append both replay buffers with current state
            replay_buffer.append((state, action_index, reward, next_state, done))
            short_term_buffer.append((state, action_index, reward, next_state, done))

            ## Append the episode data (p=0.1) [10% of episode data to be reviewed by human]
            if np.random.rand()<=0.1:
                episode_data.append((state, action, reward, next_state, done))

            ## Set next_state to state for next step
            state = next_state

            ## Enter replay_buffer when there are enough examples, then, every 30th step
            if len(replay_buffer) > batch_size and j>=30:
                print("Entering replay buffer") 

                # Randomly choose between the smaller replay buffer (more recent memories) and larger (entire episode)
                if random.random() < 0.5:
                    print("Sampling from Short Term Buffer")
                    minibatch = random.sample(short_term_buffer, batch_size)
                else:
                    print("Sampling from Long Term Buffer")
                    minibatch = random.sample(replay_buffer, batch_size)

                print("Minibatch Sampled")
                for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                    try:
                        print(f"Minibatch item {i+1}/{batch_size} processing...")
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
        # EPISODE COMPLETED
        print(f"Episode {episode} completed, moving onto Human Feedback!")
        ## Decay epsilon at end of each episode
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
        
        print("Writing final state and action")
        ## For each episode the final state and action pair will be recorded and used for human feedback
        state_image_path = os.path.join(review_dir, f"states/final_state_episode_{episode}.png")
        ## Convert the tensor back to a PIL image and save
        state_image = transforms.ToPILImage()(state.squeeze(0))  # Remove batch dimension and convert to PIL
        state_image.save(state_image_path)                       # Save the final state image

        log_path = os.path.join(review_dir, "final_review_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write(f"Episode {episode}, Final Action: {ACTION_MAP_DIALOGUE[action]}, Done: {done}\n")
        
        print("Logging to MLflow")
        ## Loggin the episode metrics to MLflow
        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        mlflow.log_metric("epsilon", epsilon, step=episode)

        ## Save current training checkpoint
        save_training_state(episode, model, optimizer, replay_buffer, short_term_buffer, epsilon, phase='phase1')

        # HUMAN FEEDBACK
        ## Use immediate training from human feedback, if negative
        print(f"Reviewing sampled actions for human feedback (Episode {episode})")
        for i, (state, action, reward, next_state, done) in enumerate(episode_data):
            # Remove unnecessary dimensions (batch and sequence dimension) and move the channel dimension to the last position
            state_img = state.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: (640, 640, 3)
            
            # Display the current state for feedback
            plt.imshow(state_img)
            plt.show()

            # Get human feedback
            penalty, better_action = get_human_feedback(ACTION_MAP_DIALOGUE[action])

            if penalty < 0:
                print(f"Human disapproved the action! Applying penalty: {penalty}")
                reward += penalty  # Override/adjust reward if action was bad
                
                # If the action was bad, terrible, then the better_action will be immediately trained on ahead of the replay buffer
                if better_action:
                    # Map the better action to the model's index
                    better_action_index = ACTION_MAPPING[better_action]
                    print("Training on immediate action...")
                    try:
                        # Get the Q-values for the current state
                        q_values = model(state.unsqueeze(0))       # Get current Q-values for the state
                        
                        # Correct the Q-value for the better action
                        target_f = q_values.clone()                # Clone current Q-values
                        target_f[0][better_action_index] = reward  # Set the Q-value for the better action

                        # Train the model on the corrected Q-values
                        optimizer.zero_grad()
                        loss = criterion(q_values, target_f)
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error processing current item: {e}")
        save_training_state(episode, model, optimizer, replay_buffer, short_term_buffer, epsilon, phase='phase1/human_feedback')





    # Save the model to MLflow once all episodes have been completed
    model_path = "models/phase1/route203_finalmodel.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")

print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")