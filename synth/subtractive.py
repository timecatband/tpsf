import torch
import torch.nn as nn
import torchaudio
from effects.filters import LearnableLowpass
from effects.dynamics import LearnableASR
from effects.distortion import SoftClipping
class SubtractiveNoiseSynth(nn.Module):
    def __init__(self, sample_rate, noise_type='white'):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_type = noise_type

        # Filter components
    
        self.filter1 = LearnableLowpass(sample_rate, initial_freq=1000.0)
        self.filter2 = LearnableLowpass(sample_rate, initial_freq=3000.0)
        # Amplitude envelope
        self.envelope = LearnableASR()
        self.distortion = SoftClipping()

    def forward(self, duration_samples):
        # Generate noise
        if self.noise_type == 'white':
            noise = torch.rand(int(duration_samples))
        # Add other noise types (pink, brown) if desired

        # Apply filter
        filtered_noise = self.filter2(self.filter1(noise))

        output = self.envelope(filtered_noise)
        output = self.distortion(output)

        return output
