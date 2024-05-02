from midi.parser import process_midi
sample_rate = 44100
import sys
import torch
import torchaudio
from synth.complex_oscillator import LearnableSineOscillator
import math
import torch.nn as nn

class LearnableSineOscillatorNonComplex(nn.Module):
    def __init__(self, freq_rad, sr):
        super(LearnableSineOscillatorNonComplex, self).__init__()
        self.freq_rad = freq_rad
        self.sr = sr
        self.phase = nn.Parameter(torch.tensor([0.0]))
        self.phase.requires_grad = True
        self.amplitude_parameters = nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
    def forward(self, num_samples):
        time = torch.linspace(0, num_samples / self.sr, num_samples)
        x = self.phase + self.freq_rad * time * sample_rate
        waveform = torch.sin(x)
        # Add harmonics
        waveform += torch.sin(2 * x) * self.amplitude_parameters[0].clamp(0, 1)
        waveform += torch.sin(3 * x) * self.amplitude_parameters[1].clamp(0, 1)
        waveform += torch.sin(4 * x) * self.amplitude_parameters[2].clamp(0, 1)
        waveform += torch.sin(5 * x) * self.amplitude_parameters[3].clamp(0, 1)
        waveform += torch.sin(6 * x) * self.amplitude_parameters[4].clamp(0, 1)
        waveform += torch.sin(7 * x) * self.amplitude_parameters[5].clamp(0, 1)
        waveform += torch.sin(8 * x) * self.amplitude_parameters[6].clamp(0, 1)

        return waveform
    def print(self):
        print("freq_rad: ", self.freq_rad)
        print("phase: ", self.phase)

def initialize_oscillators(note_events, sr):
    oscillators = {}
    for freq_rad, _, _ in note_events:
        if freq_rad not in oscillators:  
            print("Initializing oscillator for freq", freq_rad)
            # Print frequency in hz
            frequency_hz = (freq_rad / (2 * math.pi)) * sr
            print(f"Frequency in Hz: {frequency_hz:.3f}")
            oscillators[freq_rad] = LearnableSineOscillatorNonComplex(freq_rad, sr)
    return oscillators

def render_audio(note_events, oscillators, sr, duration_samples):
    output = torch.zeros(duration_samples)
    for freq_rad, start_sample, end_sample in note_events:
        if start_sample >= duration_samples:  
            continue

        # Get the oscillator (already initialized)
        oscillator = oscillators[freq_rad]

        output_length = min(end_sample, duration_samples) - start_sample
       # segment = torch.tanh(oscillator.forward(output_length))
        segment = oscillator.forward(output_length)
        output[start_sample:start_sample + output_length] += segment
    return output


midi_path = sys.argv[1]

midi_events = process_midi(midi_path, sample_rate)
oscillators = initialize_oscillators(midi_events, sample_rate)

duration_seconds = 10
duration_samples = int(duration_seconds * sample_rate) 
audio = render_audio(midi_events, oscillators, sample_rate, duration_samples)
# Save audio using torchaudio
#audio = audio/torch.max(torch.abs(audio))
audio = audio.unsqueeze(0)
torchaudio.save("dry_output.wav", audio.detach(), sample_rate)
dry_audio = audio.detach()
dry_audio_sum = dry_audio.abs().sum()
parameters_to_optimize = []
for oscillator in oscillators.values():
    parameters_to_optimize.extend(oscillator.parameters())

optimizer = torch.optim.Adam(parameters_to_optimize, lr=1e-2) 
num_steps = 1000
for i in range(num_steps):
    optimizer.zero_grad()
    audio = render_audio(midi_events, oscillators, sample_rate, duration_samples)
    energy = torch.abs(audio).sum()-dry_audio_sum
    loss = -energy
    print("Step", i, "Loss", loss.item())
    loss.backward()
    optimizer.step()
    
audio = render_audio(midi_events, oscillators, sample_rate, duration_samples)
# Save audio using torchaudio
#audio = audio/torch.max(torch.abs(audio))
audio = audio.unsqueeze(0)
torchaudio.save("output.wav", audio.detach(), sample_rate)

for oscillator in oscillators.values():
    print("Harmonic content")
    print(oscillator.amplitude_parameters)