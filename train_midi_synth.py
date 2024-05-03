import torch
import torch.nn as nn
import torchaudio
from midi.parser import process_midi
from midi.learnable_midi_synth import LearnableMidiSynthAsEffect
from synth.oscillator import LearnableHarmonicSynth
from goals.spectral_losses import SpectrogramLoss
from trainers.single_source_optimizer import SingleSourceOptimizer
from effects.build_effect_chain import build_effect_chain
import sys

dev = "cpu"
#f torch.backends.mps.is_available():
 #   dev = "mps"
if torch.cuda.is_available():
    dev = "cuda"
print("Using device", dev)
dev = torch.device(dev)

midi_file = sys.argv[1]
target_audio = sys.argv[2]
target_audio, sr = torchaudio.load(target_audio)
print("sr", sr)
midi_events = process_midi(midi_file, sr)
effect_chain = build_effect_chain("ComplexOscillator[starting_freq=3.0,sr=44100],SubtractiveSynth[sr=44100],Envelope[stages=3],ParametricIRReverb[length=44100,sampling_rate=44100],PeriodicAllPassFilter[order=2],SoftClipping,Lowpass[sample_rate=44100]")
synth = LearnableHarmonicSynth(sr, 10)
target_audio = target_audio.to(dev)
loss = SpectrogramLoss(sr)
loss_wrapper = lambda x: loss(x, target_audio)

synthAsEffect = LearnableMidiSynthAsEffect(sr, synth, effect_chain, midi_events)
synthAsEffect = synthAsEffect.to(dev)

length = target_audio.shape[1]
input_audio = torch.zeros(length)
input_audio.requires_grad = False
input_audio = input_audio.to(dev)

optimizer = SingleSourceOptimizer(input_audio, sr, synthAsEffect, loss_wrapper)
output = optimizer.optimize(10000, 0.01)
output = output.unsqueeze(0)
torchaudio.save("output.wav", output.detach().cpu(), sr)