import torch
import torch.nn as nn
import torchaudio
from midi.parser import process_midi
from midi.learnable_midi_synth import LearnableMidiSynthAsEffect
from synth.oscillator import LearnableHarmonicSynth
from synth.karplus import KarplusSynth
from synth.sequence import SynthSequence, build_synth_chain
from goals.spectral_losses import SpectrogramLoss, MultiScaleSpectrogramLoss
from trainers.single_source_optimizer import SingleSourceOptimizer
from effects.build_effect_chain import build_effect_chain
import sys
import json
import numpy as np
from common.config import parse_synth_experiment_config_from_file


dev = "cpu"
#f torch.backends.mps.is_available():
 #   dev = "mps"
if torch.cuda.is_available():
    dev = "cuda"
print("Using device", dev)
dev = torch.device(dev)

experiments_dir = sys.argv[1]
config = parse_synth_experiment_config_from_file(experiments_dir)

target_audio = config["target_audio"]
sr = config["sr"]
print("target audio shape and sr", target_audio.shape, sr)
weights = None
if "weights" in config:
    weights = config["weights"]
else:
    print("No weights specified in config")

loudness = config["loudness"]
pitch = config["pitch"]
if loudness is not None and pitch is not None:
    print ("Rescaled loudness shape", loudness.shape)
    print ("Rescaled pitch shape", pitch.shape)

# Stretch loudness and pitch to match target audio length
length = target_audio.shape[1]
midi_file = config["midi_file"]

midi_events = process_midi(midi_file, sr)

effect_chain = build_effect_chain(config["effect_chain"])
synth = build_synth_chain(config["synths"])
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