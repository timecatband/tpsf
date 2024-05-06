import torch
import torch.nn as nn
import math

def complex_oscillator(
    z,
    initial_phase,
    t: torch.Tensor,
    N: int = 2048,
    constrain: bool = False,
    reduce: bool = False,
):
    """Generates an exponentially decaying sinusoid from a complex number."""

    if initial_phase is None:
        # If no initial phase is provided, use zero phase.
        # Note that in the complex representation, zero phase is equivalent to a real number.
        initial_phase = torch.ones_like(z)
    
    if constrain:
        # Limit the magnitude of z to 1. Note that tanh is used in lieu of sigmoid to 
        # avoid vanishing gradients as magnitude approaches zero.
        mag = torch.abs(z)
        z = z * torch.tanh(mag) / mag
        
    # We calculate the oscillation in batches
    z = z[..., None].expand(N - 1)
    z = torch.cat([initial_phase.unsqueeze(-1), z], dim=-1)
    
    y = z.cumprod(dim=-1).real

    if reduce:
        y = y.sum(dim=-2)

    return y

class LearnableSineOscillator(nn.Module):
    def __init__(self, starting_freq_rad, sr):
        super().__init__()
        self.sr = sr
        starting_freq_rad = torch.tensor(starting_freq_rad)
        self.freq = nn.Parameter(torch.tensor(starting_freq_rad))
        self.predicted_z = nn.Parameter(torch.exp(1j * starting_freq_rad))
        self.initial_phase = nn.Parameter(torch.ones_like(self.predicted_z))
        self.prepared_output = torch.zeros(self.sr)
    def forward(self, x, t: float):
        # Training
        if t == 0:
            return complex_oscillator(self.predicted_z, self.initial_phase, torch.tensor([0.0]), N=x.shape[0], constrain=True)
        else:
            # TODO: How to cause this path to get traced
            if self.prepared_output is None:
                self.prepared_output = complex_oscillator(self.predicted_z, self.initial_phase, torch.tensor([t]), N=self.sr, constrain=True)
            else:
                t = t * self.sr
                tt = torch.tensor([t]).long()
                tt = tt.clamp(0, self.sr-x.shape[0]).item()
                return self.prepared_output[tt:tt+x.shape[0]]
    def print(self):
        print(f"Predicted frequency: {self.predicted_z.angle().abs().item():.3f}")
        frequency_hz = (self.predicted_z.angle().abs().item() / (2 * math.pi)) * self.sr
        print(f"Predicted frequency in Hz: {frequency_hz:.3f}")

    
    
