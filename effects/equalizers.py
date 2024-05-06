import torchaudio
import torch
import torch.nn as nn
from effects.decorator import effect

@effect("BiquadEq")
class BiquadEq(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.center_freq = nn.Parameter(torch.tensor([0.0]))
        self.q = nn.Parameter(torch.tensor([1.0]))
        self.gain = nn.Parameter(torch.tensor([1.0]))
        #self.q = torch.tensor([1.0])
        #self.gain = torch.tensor([5.0])
    def forward(self, x, t):
        gain = torch.clamp(self.gain, 0.1, 10)
        return torchaudio.functional.equalizer_biquad(x, self.sample_rate, self.center_freq, self.q, gain)
    def print(self):
        print("center_freq: ", self.center_freq)
        print("q: ", self.q)
        print("gain: ", self.gain)
   
@effect("NotchFilter")     
class NotchFilter(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.filter = nn.Conv1d(2, 2, 5, padding=2, bias=False)
        self.more_filters = nn.Sequential(  
            nn.Conv1d(1, 16, 3, padding=1, bias=False),
           nn.Conv1d(16, 16, 7, padding=3, bias=False),
            nn.Conv1d(16, 1, 3, padding=2, bias=False),
        )
        
        self.sample_rate = sample_rate
    def forward(self, x, t):
        x = x.unsqueeze(0)
        output = self.more_filters(x)
        # Clip output shape to match input
        output = output[:, :x.size(1)]
        return output.squeeze(0)
    def print(self):
        print("filter: ", self.filter)