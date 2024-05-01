import torch
import torch.nn as nn

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
