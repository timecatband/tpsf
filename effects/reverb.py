import torchaudio
import torch
import torch.nn as nn

class IRReverb(nn.Module):
    def __init__(self, ir):
        super().__init__()
        self.ir = ir
    def forward(self, x):
        return torchaudio.functional.fftconvolve(x, self.ir, mode='full')
    
class ParametricEarlyReflections(nn.Module):
    def __init__(self, num_reflections, base_delay, decay_factor, filter_cutoff=None):
        super().__init__()
        self.num_reflections = nn.Parameter(torch.tensor(num_reflections, dtype=torch.int))
        self.base_delay = nn.Parameter(torch.tensor(base_delay))
        self.decay_factor = nn.Parameter(torch.tensor(decay_factor))

        if filter_cutoff is not None:
            self.filter = nn.Linear(1, 1)  # Simple low-pass filter
            self.filter.weight.data.fill_(filter_cutoff) 
            self.filter.bias.data.zero_()

    def forward(self, x):
        delays = torch.arange(self.num_reflections.item()) * self.base_delay
        amplitudes = self.decay_factor ** delays
        output = torch.zeros_like(x)

        for i in range(self.num_reflections.item()):
            delayed_x = torch.roll(x, shifts=int(delays[i]), dims=-1)  # Circular delay
            if hasattr(self, 'filter'):
                delayed_x = self.filter(delayed_x.unsqueeze(1)).squeeze(1)
            output += delayed_x * amplitudes[i]

        return output
