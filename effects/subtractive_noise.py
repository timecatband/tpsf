from synth.subtractive import SubtractiveNoiseSynth
import torch
import torch.nn as nn
from effects.decorator import effect

@effect("SubtractiveSynth")
class SubtractiveSynthAsEffect(nn.Module):
    def __init__(self, sr):
        super().__init__()
        self.synth = SubtractiveNoiseSynth(sr)
        self.blend = nn.Parameter(torch.tensor([0.9]))
    def forward(self, x, t):
        # TODO: t might be needed...
        x_length_samples = x.size(0)
        synth_out = self.synth(x, t)
        
        return (1-self.blend)*synth_out + self.blend*x