import torch
import torch.nn as nn

class SoftClipping(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([gain]))

    def forward(self, x):
        print("gain: ", self.gain)
        return torch.tanh(self.gain * x)


class HardClipping(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([threshold]))

    def forward(self, x):
        return torch.clamp(x, min=-self.threshold, max=self.threshold)