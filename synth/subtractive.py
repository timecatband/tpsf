import torch
import torch.nn as nn
import torchaudio
from effects.filters import LearnableLowpass, LearnableHighpass
from effects.dynamics import LearnableASR
from effects.distortion import SoftClipping
class SubtractiveNoiseSynth(nn.Module):
    def __init__(self, sample_rate, noise_type='white'):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_type = noise_type

        # Filter components
    
        self.filter1 = LearnableLowpass(sample_rate, initial_freq=12000.0)
        self.filter2 = LearnableHighpass(sample_rate, initial_freq=8000.0)
        # Amplitude envelope
        self.envelope = LearnableASR()
        self.distortion = SoftClipping()
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

    def forward(self, x, t):
        duration_samples = x.size(0)
        # Generate noise
        noise = torch.rand(int(duration_samples)).to(self.dev)
        # Add other noise types (pink, brown) if desired

        # Apply filter
        # Turning both passes on triggers divergence
        noise = self.filter1(noise, t)
        filtered_noise = self.filter2(noise, t)
        filtered_noise = filtered_noise / torch.max(torch.abs(filtered_noise))
        output = self.envelope(filtered_noise, t)
        
        output = self.distortion(output, t)
        

        return output
