
import torch
import torch.nn as nn

class LearnableSineOscillator(nn.Module):
    def __init__(self, freq_rad, sr):
        super(LearnableSineOscillator, self).__init__()
        self.freq_rad = nn.Parameter(torch.tensor([freq_rad]))
        self.sr = sr
        self.phase = nn.Parameter(torch.tensor([0.0]))
        self.phase.requires_grad = True
        self.amplitude_parameters = nn.Parameter(torch.tensor([1.0,0.5,0.25,.125,0.0625,0.03125,0.015625]))
    def forward(self, num_samples):
        time = torch.linspace(0, num_samples / self.sr, num_samples)
        x = self.freq_rad * time * self.sr
        waveform = torch.sin(x+self.phase)
        # Add harmonics
        waveform += torch.sin(2 * x+self.phase) * self.amplitude_parameters[0].clamp(0, 1)
        waveform += torch.sin(3 * x+self.phase) * self.amplitude_parameters[1].clamp(0, 1)
        waveform += torch.sin(4 * x+self.phase) * self.amplitude_parameters[2].clamp(0, 1)
        waveform += torch.sin(5 * x+self.phase) * self.amplitude_parameters[3].clamp(0, 1)
        waveform += torch.sin(6 * x+self.phase) * self.amplitude_parameters[4].clamp(0, 1)
        waveform += torch.sin(7 * x+self.phase) * self.amplitude_parameters[5].clamp(0, 1)
        waveform += torch.sin(8 * x+self.phase) * self.amplitude_parameters[6].clamp(0, 1)
       # waveform = waveform / waveform.abs().max()
        return waveform
    def print(self):
        print("freq_rad: ", self.freq_rad)
        print("phase: ", self.phase)
        freq_hz = self.freq_rad * self.sr / (2 * 3.14159)
        print("freq_hz: ", freq_hz)