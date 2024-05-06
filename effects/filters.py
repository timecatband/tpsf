import torch
import torch.nn as nn
import torchaudio
from effects.decorator import effect

class LearnableToneKnob(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_cutoff_freq):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.cutoff_freq = nn.Parameter(torch.tensor(0.5))  # Initialize mid-range
        self.max_cutoff_freq = max_cutoff_freq 

    def design_lowpass_kernel(self, cutoff_freq, kernel_size):
        # Example: A more flexible Gaussian-based window design
        center = kernel_size // 2
        std = center / cutoff_freq  # Control width based on cutoff
        kernel = torch.exp(-0.5 * torch.arange(kernel_size).pow(2) / std)
        kernel = kernel / kernel.sum()  # Normalize
        return kernel

    def forward(self, x):
        cutoff_freq = self.cutoff_freq.sigmoid() * self.max_cutoff_freq  # Scale to range
        kernel = self.design_lowpass_kernel(cutoff_freq, self.conv.kernel_size[0])
        self.conv.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0)) 
        return self.conv(x)

#TODO: Make sure we didn't break grads with .item()
@effect("Lowpass")
class LearnableLowpass(nn.Module):
    def __init__(self, sample_rate, initial_freq=15000.0):
        super().__init__()
        self.sample_rate = sample_rate

        # Filter components
    
        self.filter_freq = nn.Parameter(torch.tensor([initial_freq]))
        self.filter_q = nn.Parameter(torch.tensor([0.9]))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        filter_freq = self.filter_freq.clamp(100, self.sample_rate / 2 - 1)
        q = self.filter_q.clamp(0.1, 10)
        out = torchaudio.functional.lowpass_biquad(
            x,
            self.sample_rate,    # Sample rate
            filter_freq,
            q
        )

        return out
    def print(self):
        print("filter_freq: ", self.filter_freq)
        print("filter_q: ", self.filter_q)
    
@effect("Highpass")
class LearnableHighpass(nn.Module):
    def __init__(self, sample_rate, initial_freq=100.0):
        super().__init__()
        self.sample_rate = sample_rate

        # Filter components
    
        self.filter_freq = nn.Parameter(torch.tensor([initial_freq]))
        self.filter_q = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        filter_freq = self.filter_freq.clamp(100, self.sample_rate / 2 - 1)
        filter_q = self.filter_q.clamp(0.1, 10)
        out = torchaudio.functional.highpass_biquad(
            x,
            self.sample_rate,    # Sample rate
            filter_freq, filter_q
        )

        return out
