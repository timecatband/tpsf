from synth.subtractive import SubtractiveNoiseSynth
import torch
import torch.nn as nn
from effects.decorator import effect

@effect("SubtractiveSynth")
class SubtractiveSynthAsEffect(nn.Module):
    def __init__(self, sr, exp_decay=True):
        super().__init__()
        self.synth = SubtractiveNoiseSynth(sr)
        self.blend = nn.Parameter(torch.tensor([0.1]))
        self.exp_decay = exp_decay
        
    def forward(self, x, t):
        # TODO: t might be needed...
        x_length_samples = x.size(0)
        synth_out = self.synth(x, t)
        
        decay = torch.ones(x_length_samples)
        if self.exp_decay:
            # Create a tensor of same length as x which exponentially decays from 1 to 0
            decay = torch.exp(torch.linspace(1, 0, x_length_samples)) - 1.0
            decay = decay / decay.max()
            
        
        return (1-self.blend)*synth_out*decay + x