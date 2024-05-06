import torch
import torch.nn as nn
from effects.decorator import effect

@effect("Envelope")
class LearnableASR(torch.nn.Module):
    def __init__(self, stages=3):
        super().__init__()
        self.asr = nn.Sequential(
            nn.Linear(1, stages),
            nn.ReLU(),
            nn.Linear(stages, 1),
        )
        # Initialize biases to 0
        self.asr[0].bias.data.fill_(0)
    def forward(self, x, t):
        # TODO: Hardcoded sample rate
        time_grid = torch.linspace(t, t+x.size(0)/44100, x.size(0)).unsqueeze(1).float().to(x.device)
        # Reverse time grid
        
        time_grid = 1 - time_grid
    
        envelope = self.asr(time_grid)
        envelope = envelope / envelope.max()
        return envelope.squeeze(1) * x