from midi.parser import process_midi
sample_rate = 44100
import sys
import torch
import torchaudio
from synth.oscillator import LearnableSineOscillator
import math
import torch.nn as nn
from effects.distortion import SoftClipping, HardClipping, Gain
from effects.reverb import LearnableIRReverb, LearnableIRReverbSinusoidal, LearnableNoiseReverb, LearnableParametricIRReverb
from goals.spectral_losses import SpectrogramLoss
from effects.waveform_phase import PeriodicAllPassFilter
from effects.equalizers import NotchFilter
from effects.dynamics import LearnableASR
from effects.subtractive_noise import SubtractiveSynthAsEffect
from effects.complex_oscillator import ComplexOscillatorAsEffect
from effects.filters import LearnableLowpass
def initialize_oscillators(note_events, sr):
    oscillators = {}
    for freq_rad, _, _ in note_events:
        if freq_rad not in oscillators:  
            print("Initializing oscillator for freq", freq_rad)
            # Print frequency in hz
            frequency_hz = (freq_rad / (2 * math.pi)) * sr
            print(f"Frequency in Hz: {frequency_hz:.3f}")
            oscillators[freq_rad] = LearnableSineOscillator(freq_rad, sr)
            oscillators[freq_rad].freq_rad.requires_grad = False
    return oscillators

def render_audio(note_events, oscillators, sr, duration_samples, effect_chain):
    output = torch.zeros(duration_samples)
    
    for freq_rad, start_sample, end_sample in note_events:
        if start_sample >= duration_samples:  
            continue
        print("Start and end samples duration samples", start_sample, end_sample, duration_samples)
        # Get the oscillator (already initialized)
        oscillator = oscillators[freq_rad]
        

        output_length = min(end_sample, duration_samples) - start_sample
       # segment = torch.tanh(oscillator.forward(output_length))
        segment = oscillator.forward(output_length)
        if effect_chain is not None:
            segment = effect_chain(segment)
        output[start_sample:start_sample + output_length] += segment
    return output


midi_path = sys.argv[1]

midi_events = process_midi(midi_path, sample_rate)
oscillators = initialize_oscillators(midi_events, sample_rate)

target_audio = sys.argv[2]
target_audio, sr = torchaudio.load(target_audio)
duration_samples = target_audio.shape[1] 
audio = render_audio(midi_events, oscillators, sample_rate, duration_samples, None)
# Save audio using torchaudio
#audio = audio/torch.max(torch.abs(audio))
audio = audio.unsqueeze(0)
torchaudio.save("dry_output.wav", audio.detach(), sample_rate)
dry_audio = audio.detach()
dry_audio_sum = dry_audio.abs().sum()
parameters_to_optimize = []
for oscillator in oscillators.values():
    parameters_to_optimize.extend(oscillator.parameters())

lowpass = LearnableLowpass(sample_rate, 14000.0)
lowpass.filter_q.requires_grad = False

lowpass.filter_freq.requires_grad = False

effect_chain = nn.Sequential(
   # LearnableIRReverb(2048),

    LearnableASR(),
    SubtractiveSynthAsEffect(sample_rate),

    #LearnableNoiseReverb(128),
   # NotchFilter(sample_rate),
    PeriodicAllPassFilter(3, sample_rate),
    SoftClipping(),
    LearnableParametricIRReverb(sample_rate, sample_rate),
    #LearnableLowpass(sample_rate, sample_rate/2.0),
    Gain(),
)
parameters_to_optimize.extend(effect_chain.parameters())


optimizer = torch.optim.AdamW(parameters_to_optimize, lr=3e-2) 
num_steps = 10000
spec_loss = SpectrogramLoss(sample_rate)
for i in range(num_steps):
    optimizer.zero_grad()
    audio = render_audio(midi_events, oscillators, sample_rate, duration_samples, effect_chain)
    #audio = audio / audio.abs().max()
    # Compute msel oss between audio and target
    #audio = lowpass(audio)
    #loss = torch.mean((audio - target_audio)**2)
    
    loss =  spec_loss(audio, target_audio)
    print("Step", i, "Loss", loss.item())
    loss.backward()
    optimizer.step()
    
    # Decay lr
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.9999
    
    if i % 50 == 0:
        torchaudio.save(f"output_intermediate.wav", audio.detach().unsqueeze(0), sample_rate)
    
audio = render_audio(midi_events, oscillators, sample_rate, duration_samples, effect_chain)
# Save audio using torchaudio
#audio = audio/torch.max(torch.abs(audio))
audio = audio.unsqueeze(0)

torchaudio.save("output.wav", audio.detach(), sample_rate)

for oscillator in oscillators.values():
    print("Harmonic content")
    print(oscillator.amplitude_parameters)