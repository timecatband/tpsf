import torchaudio
import torch
import torch.nn as nn
from effects.dynamics import LearnableASR

class IRReverb(nn.Module):
    def __init__(self, ir):
        super().__init__()
        self.ir = ir
    def forward(self, x):
        return torchaudio.functional.fftconvolve(x, self.ir, mode='full')
    
class SinAsModule(nn.Module):
    def __init__(self, freq=0.1):
        super().__init__()
        self.freq = freq
    def forward(self, x):
        return torch.sin(x*self.freq)

class LearnableIRReverbSinusoidal(nn.Module):
    def __init__(self, ir_length, num_sinusoids=32):
        super().__init__()
        self.ir_length = ir_length
        self.time_grid = nn.Parameter(torch.linspace(0, 1, ir_length), requires_grad=False)
        self.sinusoidal_net = nn.Sequential(
            nn.Linear(1, num_sinusoids),  # Input: time coordinate
            SinAsModule(),  # Sinusoidal activation
            nn.Linear(num_sinusoids, 1)   # Output: single IR value
        )    
        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        ir = self.sinusoidal_net(self.time_grid.unsqueeze(1)).squeeze(1)
        out = torchaudio.functional.fftconvolve(x, ir, mode='full')
        return out[:x.shape[0]] * self.blend + x *(1 - self.blend) 

class LearnableIRReverb(nn.Module):
    def __init__(self, ir_length):
        super().__init__()
        self.ir = nn.Parameter(torch.randn(ir_length))
        self.blend = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        out = torchaudio.functional.fftconvolve(x, self.ir, mode='full')
        # Clip to input length
        return out[:x.shape[0]]*self.blend + x*(1-self.blend)

class LearnableNoiseReverb(nn.Module):
    def __init__(self, ir_length):
        super().__init__()
        self.reverb = LearnableIRReverb(ir_length)
        self.envelope = LearnableASR()
    def forward(self, x):
        noise = torch.randn_like(x)
        reverb = self.reverb(noise)
        reverb = self.envelope(reverb)
        return reverb*0.01+x

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
