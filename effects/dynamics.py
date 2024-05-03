import torch
import torch.nn as nn

class LearnableASR(torch.nn.Module):
    def __init__(self, asr=None):
        super().__init__()
        self.asr = asr
        self.asr = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        # Initialize biases to 0
        self.asr[0].bias.data.fill_(0)
    def forward(self, x):
        time_grid = torch.arange(x.size(0)).unsqueeze(1).float()
        time_grid = time_grid / time_grid.max()
        # Reverse time grid
        time_grid = 1 - time_grid
    
        envelope = self.asr(time_grid)
        envelope = envelope / envelope.max()
        return envelope.squeeze(1) * x