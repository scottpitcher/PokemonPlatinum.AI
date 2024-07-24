# pokemon_model.py

import torch.nn as nn
import torch.nn.functional as F

class PokemonModelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions):
        super(PokemonModelLSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTM(input_size=32*160*160, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x, annotations=None):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
