
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
        
class LearnableHarmonicSynth(nn.Module):
    def __init__(self, sr, num_harmonics, enable_amplitude_scaling=True, time_latent_size=2):
        super(LearnableHarmonicSynth, self).__init__()
        self.sr = sr
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.harmonic_amplitudes = nn.Parameter(torch.ones(num_harmonics))
    #    if torch.backends.mps.is_available():
     #       self.device = torch.device("mps")
        if torch.cuda is not None and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.amplitude_scaler = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_harmonics)
        )
        self.map_time_latent = nn.Sequential(
            nn.Linear(time_latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_harmonics)
        )
        self.map_amplitude = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8,num_harmonics)
        )
        self.map_fundamental_amplitude = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
            
        # Rescale harmonic amplitudes to decay
       # self.harmonic_amplitudes = harmonic_amplitudes / torch.arange(1, num_harmonics+1).float()
        
    # TODO Modify this to accept amplitude latent
    def forward(self, freq, output_length_samples, time_latent=None, amplitude=None):
        time = torch.linspace(0, output_length_samples / self.sr, output_length_samples).to(self.device)
        x = freq * time * self.sr
        waveform = torch.sin(x+self.phase)
        for i in range(1, len(self.harmonic_amplitudes)):
            # Convert frequency to hz and check if it is above half the sampling rate
            freq_hz = freq * self.sr / (2 * 3.14159)
            scale = 1.0
            if amplitude is not None:
                scale = self.map_fundamental_amplitude(torch.tensor([amplitude]).unsqueeze(0).to(self.device)).squeeze(0)
            
            if freq_hz * (i+1) > self.sr / 2:
                scale = 1e-4
            hamps = self.harmonic_amplitudes
            if self.amplitude_scaler is not None:
                hamps = hamps*self.amplitude_scaler(torch.tensor([freq]).unsqueeze(0).to(self.device))
            if time_latent is not None:
                hamps = hamps*self.map_time_latent(time_latent.unsqueeze(0).to(self.device)).squeeze(0)
            if amplitude is not None:
                hamps = hamps * self.map_amplitude(torch.tensor([amplitude]).unsqueeze(0).to(self.device)).squeeze(0)
            waveform += scale * torch.sin((i+1) * x+self.phase) * self.harmonic_amplitudes[i]
        # TODO...maybe this is bad
        waveform = waveform / waveform.abs().max()
        return waveform
    
class NullSynth(nn.Module):
    def __init__(self):
        super(NullSynth, self).__init__()
    def forward(self, x):
        return x