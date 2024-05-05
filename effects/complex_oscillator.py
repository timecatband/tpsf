from synth.complex_oscillator import LearnableSineOscillator
from effects.dynamics import LearnableASR
import torch.nn as nn
import torch
from effects.decorator import effect

@effect("ComplexOscillator")
class ComplexOscillatorAsEffect(nn.Module):
    def __init__(self, starting_freq, sr):
        super(ComplexOscillatorAsEffect, self).__init__()
        self.osc = LearnableSineOscillator(starting_freq, sr)
        self.sr = sr
        self.envelope = LearnableASR()
        self.gain = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, t):
        gain = torch.clamp(self.gain, 0.01, 5)
        return self.envelope(self.osc(x, t), t)*gain+x