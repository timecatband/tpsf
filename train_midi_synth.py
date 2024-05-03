import torch
import torch.nn as nn
import torchaudio
from midi.parser import process_midi
from midi.learnable_midi_synth import LearnableMidiSynthAsEffect
from synth.oscillator import LearnableHarmonicSynth
from goals.spectral_losses import SpectrogramLoss
from trainers.single_source_optimizer import SingleSourceOptimizer
import sys

midi_file = sys.argv[1]
target_audio = sys.argv[2]
target_audio, sr = torchaudio.load(target_audio)
print("sr", sr)
midi_events = process_midi(midi_file, sr)
effect_chain = None
synth = LearnableHarmonicSynth(sr, 10)

loss = SpectrogramLoss(sr)
loss_wrapper = lambda x: loss(x, target_audio)

synthAsEffect = LearnableMidiSynthAsEffect(sr, synth, effect_chain, midi_events)
length = target_audio.shape[1]
input_audio = torch.zeros(length)
input_audio.requires_grad = False

optimizer = SingleSourceOptimizer(input_audio, sr, synthAsEffect, loss_wrapper)
output = optimizer.optimize(1000, 0.001)
output = output.unsqueeze(0)
torchaudio.save("output.wav", output.detach(), sr)