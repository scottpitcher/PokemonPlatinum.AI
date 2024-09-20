# rlhf_phase1.py

print("Script Starting...")
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import mlflow
import mlflow.pytorch
import sys

device = 'cpu'
print(f"Device: {device}")

# Hyperparameters
epsilon = 0.99
epsilon_decay = 0.995
min_epsilon = 0.05
gamma = 0.99
replay_buffer = deque(maxlen=10000)
batch_size = 32
num_episodes = 500
learning_rate = 1e-4
print("Model parameters set!")

# Set system path to project
sys.path.append('/Users/scottpitcher/Desktop/python/Github/PokemonPlatinum.AI/')
from RLHF_Scripts.modular_scripts.pokemon_platinum_env import PokemonPlatinumEnv  # Import the custom Gym environment
from RLHF_Scripts.modular_scripts.load_model import load_phase_1

# Load your pretrained model and optimizer
model, optimizer = load_phase_1()
criterion = nn.MSELoss()
model.eval()
print("Gameplay model successfully loaded!")

# Open the Gym environment
env = PokemonPlatinumEnv()

# Start MLflow experiment
mlflow.set_experiment("Pokemon_Platinum_AI_Phase1")

print("Beginning training...")
with mlflow.start_run():
    mlflow.log_param("epsilon", epsilon)
    mlflow.log_param("epsilon_decay", epsilon_decay)
    mlflow.log_param("min_epsilon", min_epsilon)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_episodes", num_episodes)
    mlflow.log_param("learning_rate", learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        j = 0

        while not done:
            j += 1
            if np.random.rand() <= epsilon:
                # Random action
                action = env.action_space.sample()
            else:
                # Use your pretrained model to predict the action
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done, info = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            episode_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor

            if len(replay_buffer) > batch_size and j >= 50:
                minibatch = random.sample(replay_buffer, batch_size)
                for i, (s, a, r, s_next, d) in enumerate(minibatch):
                    try:
                        s = s.to(device)
                        s_next = s_next.to(device)
                        target = r
                        if not d:
                            target += gamma * torch.max(model(s_next)).item()
                        target_f = model(s)
                        target_val = target_f.clone()
                        target_val[0][a] = target

                        optimizer.zero_grad()
                        loss = criterion(target_f, target_val)
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error processing minibatch item {i}: {e}")
                j = 0

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # Logging
        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        mlflow.log_metric("epsilon", epsilon, step=episode)

        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

    # Save the model
    model_path = "models/pokemon_platinum_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")

env.close()