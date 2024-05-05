import torch
import torch.nn as nn
import math
from effects.decorator import effect

@effect("PhaseDistortion")
class LearnablePhaseDistortion(nn.Module):
    def __init__(self, order=1, sample_rate=44100):
        super().__init__()
        self.order = order
        self.sample_rate = sample_rate

        # Initialize filter coefficients (we'll design these below)
        self.a = nn.Parameter(torch.rand(order))  
        self.b = nn.Parameter(torch.rand(order))
        self.gain = 1 #nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        y = x.clone()  # Start with a copy of the input 

        # Implement all-pass filter logic (order determines complexity)
        for i in range(self.order):
            y = (self.b[i] * x) + (self.a[i] * y) - (self.b[i] * y.roll(-1, dims=-1))
            x = x.roll(-1, dims=-1) 

        return torch.tanh(self.gain*y) 

@effect("TimeDependentAllPassFilter")
class TimeDependentAllPassFilter(nn.Module):
    def __init__(self, order=1, sample_rate=44100):
        super().__init__()
        self.order = 3
        order = 3
        self.sample_rate = sample_rate

        # Initialize learnable coefficients
        self.a = nn.Parameter(torch.rand(order))  
        self.b = nn.Parameter(torch.rand(order))
        self.frequency = nn.Parameter(torch.tensor([0.2]))
        size = 256
        self.t_modulator = nn.Sequential(
            nn.Linear(1, size),
            nn.ReLU(),
            nn.Linear(size, 1)
        )

    def forward(self, x, t):
        # Normalize time for stable learning
       # t_normalized = t / self.sample_rate
        t_normalized = self.t_modulator(t.unsqueeze(1)).squeeze(1)

        y = x.clone()  

        # Time-dependent all-pass filter
        for i in range(self.order):
            # Example modulation - sinusoidal variation 
            #modulation = 0.5 * torch.sin(2 * math.pi * t_normalized * self.frequency)  
            modulation = t_normalized
            # Apply modulated coefficients
            a_mod = self.a[i] * (1 + modulation)
            b_mod = self.b[i] * (1 + modulation)

            y = (b_mod * x) + (a_mod * y) - (b_mod * y.roll(-1, dims=-1))
            x = x.roll(-1, dims=-1) 

        return y 
    

@effect("PeriodicAllPassFilter")
class PeriodicAllPassFilter(nn.Module):
    def __init__(self, order=1, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate

        # Initialize learnable coefficients
        self.a = nn.Parameter(torch.rand(order))  
        self.b = nn.Parameter(torch.rand(order))
        # Initialize a freq param for each order
        self.frequency = nn.Parameter(torch.rand(order))
        self.order = order
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

    def forward(self, x, t = None):
        #if t is None:
        # TODO AU:  We should probably do something with T here
        t = torch.arange(x.size(0)).to(self.dev)
        t_normalized = t /  t.max()


        y = x.clone()  

        # Time-dependent all-pass filter
        for i in range(self.order):
            # Example modulation - sinusoidal variation 
            modulation = 0.5 * torch.sin(2 * math.pi * t_normalized * self.frequency[i])  
            modulation = t_normalized
            # Apply modulated coefficients
            a_mod = self.a[i] * (1 + modulation)
            b_mod = self.b[i] * (1 + modulation)

            y = (b_mod * x) + (a_mod * y) - (b_mod * y.roll(-1, dims=-1))
            x = x.roll(-1, dims=-1) 

        return y 