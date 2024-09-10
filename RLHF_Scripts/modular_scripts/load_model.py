# load_model.py
from models.PokemonModelLSTM import PokemonModelLSTM
import torch
import torch.optim as optim

# Setting Hyperparameters
num_actions = 9  # (Total Number of Actions: [A, B, X, Y, Up, Down, Left, Right, None]) (Excluding Start, Select, L, R to reduce model complexity)
input_size = 32 * 160 * 160
hidden_size = 128
num_layers = 2
num_epochs = 20
learning_rate = 0.001

# Initialising model
model = PokemonModelLSTM(input_size, hidden_size, num_layers, num_actions)


def load_phase_1(model=model):
    """This will load the checkpoint for the model after pretraining to be used in Phase 1 training"""
    # Load the saved checkpoint
    checkpoint = torch.load("models/pokemon_model_lstm_epoch_2.pth", map_location=torch.device('cpu'))
    # Load the model state from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # To continue training from this checkpoint, restore the optimizer and loss:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def load_phase_2(model=model):
    """This will load the state_dict for the model after Phase 1 to be used in Phase 2 training"""
    # Load the trained model
    state_dict = torch.load("NA")
    model.load_state_dict(state_dict)
    return model

def load_phase_3(model=model):
    """This will load the state_dict for the model after Phase 2 to be used in Phase 3 training"""
    # Load the trained model
    state_dict = torch.load("NA")
    model.load_state_dict(state_dict)
    return model