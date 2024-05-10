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
        t = t.clamp(0, 1)
        t_float = t.item()
        time_grid = torch.linspace(t_float, t_float+x.size(0)/44100, x.size(0)).unsqueeze(1).float().to(x.device)
        # Reverse time grid (max is not 1!)
        time_grid = time_grid.flip(1)
    
        envelope = self.asr(time_grid)
        #envelope = envelope.clamp(0, 1)
      #  envelope = envelope.abs()
       # envelope = envelope / envelope.abs().max()
        
        return envelope.squeeze(1) * x