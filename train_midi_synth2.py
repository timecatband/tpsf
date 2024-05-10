import torch
import torch.nn as nn
import torchaudio
from midi.parser import process_midi
from midi.learnable_midi_synth import LearnableMidiSynthAsEffect
from synth.oscillator import LearnableHarmonicSynth
from goals.spectral_losses import SpectrogramLoss, MultiScaleSpectrogramLoss
from trainers.single_source_optimizer import SingleSourceOptimizer
from effects.build_effect_chain import build_effect_chain
import sys
import json
import numpy as np

def stretch_signal(signal, stretched_size):
    stretched_samples_per_original_sample = stretched_size / signal.shape[0]
    stretched_components = []
    for i in range(signal.shape[0]):
        # Repeat signal i by stretched_samples_per_original_sample times
        stretched_sample = signal[i].repeat(int(stretched_samples_per_original_sample))
        # Mix in the next sample
        if i + 1 < signal.shape[0]:
            blend = torch.linspace(1, 1/stretched_samples_per_original_sample, int(stretched_samples_per_original_sample))
            stretched_sample = stretched_sample * blend + signal[i + 1] * (1 - blend)
        stretched_components.append(stretched_sample)
    stretched_signal = torch.cat(stretched_components)
    return stretched_signal

dev = "cpu"
#f torch.backends.mps.is_available():
 #   dev = "mps"
if torch.cuda.is_available():
    dev = "cuda"
print("Using device", dev)
dev = torch.device(dev)

experiments_dir = sys.argv[1]
experiments_json = experiments_dir + "/experiment.json"
config = None
with open(experiments_json) as f:
    config = json.load(f)
midi_file = config["midi_file"]
wav_file = config["wav_file"]
effects = config["effect_chain"]
effect_chain_string = ""
for effect in effects:
    effect_chain_string += effect + ","

loudness_npy = config["loudness"]
pitch_npy = config["pitch"]
loudness_npy = experiments_dir + "/" + loudness_npy
pitch_npy = experiments_dir + "/" + pitch_npy
loudness = np.load(loudness_npy)
pitch = np.load(pitch_npy)
loudness = torch.tensor(loudness)
pitch = torch.tensor(pitch)
print("loudness shape", loudness.shape)
print("pitch shape", pitch.shape)




wav_file = experiments_dir + "/" + wav_file
midi_file = experiments_dir + "/" + midi_file
target_audio, sr = torchaudio.load(wav_file)
print("target audio shape and sr", target_audio.shape, sr)
weights = None
if "weights" in config:
    weights = config["weights"]
    weights = experiments_dir + "/" + weights
else:
    print("No weights specified in config")

loudness = stretch_signal(loudness, target_audio.shape[1])
pitch = stretch_signal(pitch, target_audio.shape[1])
# Clip audio to loudness length
print ("Rescaled loudness shape", loudness.shape)
print ("Rescaled pitch shape", pitch.shape)

# Stretch loudness and pitch to match target audio length
length = target_audio.shape[1]

midi_events = process_midi(midi_file, sr)

effect_chain = build_effect_chain(effect_chain_string)
synth = LearnableHarmonicSynth(sr, 10)
target_audio = target_audio.to(dev)
#loss = SpectrogramLoss(sr)
loss = MultiScaleSpectrogramLoss(sr)
loss_wrapper = lambda x: loss(x, target_audio)

synthAsEffect = LearnableMidiSynthAsEffect(sr, synth, effect_chain, midi_events, pitch)
synthAsEffect = synthAsEffect.to(dev)
if weights is not None:
    print("Loading weights", weights)
    synthAsEffect.load_state_dict(torch.load(weights))

length = target_audio.shape[1]
input_audio = torch.zeros(length)
input_audio.requires_grad = False
input_audio = input_audio.to(dev)

#synthAsEffect.lms = torch.jit.script(synthAsEffect)

optimizer = SingleSourceOptimizer(input_audio, sr, synthAsEffect, loss_wrapper)
output = optimizer.optimize(2000, 0.01)
output = output.unsqueeze(0)
torchaudio.save("output.wav", output.detach().cpu(), sr)
torch.save(synthAsEffect.state_dict(), "synth_as_effect.pth")