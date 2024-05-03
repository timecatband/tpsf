from synth.subtractive import SubtractiveNoiseSynth
import torch
import torch.nn as nn
from effects.decorator import effect

@effect("SubtractiveSynth")
class SubtractiveSynthAsEffect(nn.Module):
    def __init__(self, sr):
        super().__init__()
        self.synth = SubtractiveNoiseSynth(sr)
    def forward(self, x):
        x_length_samples = x.size(0)
        return self.synth(x_length_samples)+x