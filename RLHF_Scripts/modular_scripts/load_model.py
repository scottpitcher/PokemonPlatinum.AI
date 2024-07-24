from models.PokemonModelLSTM import PokemonModelLSTM
import torch

# Setting Hyperparameters
num_actions = 9  # (Total Number of Actions: [A, B, X, Y, Up, Down, Left, Right, None]) (Excluding Start, Select, L, R to reduce model complexity)
input_size = 32 * 160 * 160
hidden_size = 128
num_layers = 2
num_epochs = 20

# Initialising model
model = PokemonModelLSTM(input_size, hidden_size, num_layers, num_actions)

def load_phase_1(model):
    """This will load the state_dict for the model after pretraining to be used in Phase 1 training"""
    # Load the trained model
    state_dict = torch.load("models/pokemon_model_lstm.pth")
    model.load_state_dict(state_dict)
    return model

def load_phase_2(model):
    """This will load the state_dict for the model after Phase 1 to be used in Phase 2 training"""
    # Load the trained model
    state_dict = torch.load("NA")
    model.load_state_dict(state_dict)
    return model

def load_phase_3(model):
    """This will load the state_dict for the model after Phase 2 to be used in Phase 3 training"""
    # Load the trained model
    state_dict = torch.load("NA")
    model.load_state_dict(state_dict)
    return model