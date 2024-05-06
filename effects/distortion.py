import torch
import torch.nn as nn
from effects.decorator import effect

@effect("SoftClipping")
class SoftClipping(nn.Module):
    def __init__(self, gain=1.0, blend=0.99):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([gain]))
        self.color = nn.Parameter(torch.tensor([0.0]))
        self.blend = nn.Parameter(torch.tensor([blend]))
    def forward(self, x, t):
        out = torch.tanh(self.gain * x + self.color)
        return (1-self.blend)*out + self.blend*x
    
@effect("HardClipping")
class HardClipping(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([threshold]))

    def forward(self, x):
        return torch.clamp(x, min=-self.threshold, max=self.threshold)
    
@effect("Gain")
class Gain(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([gain]))

    def forward(self, x):
        return self.gain * x

@effect("ToneKnob")
class ToneKnob(nn.Module):
    # Implement a lowpass with pytorch
    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = nn.Parameter(torch.tensor([cutoff]))
        self.filter = torch.tensor([1, 0])
        self.filter = self.filter / torch.sum(self.filter)
        