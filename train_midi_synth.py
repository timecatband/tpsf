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
print("target audio shape and sr", target_audio.shape, sr)
print("sr", sr)
weights = None
if len(sys.argv) > 3:
    weights = sys.argv[3]
midi_events = process_midi(midi_file, sr)
#effect_chain = build_effect_chain("SubtractiveSynth[sr=44100], Envelope[stages=4], ComplexOscillator[starting_freq=3.0,sr=44100],ComplexOscillator[starting_freq=1.5,sr=44100],ParametricIRReverb[length=44100,sampling_rate=44100],PeriodicAllPassFilter[order=3],SoftClipping,Lowpass[sample_rate=44100],NotchFilter[sample_rate=44100]")
#,PeriodicAllPassFilter[order=3],SoftClipping,Lowpass[sample_rate=44100],NotchFilter[sample_rate=44100]")
#effect_string = "Gain,"
effect_string = "Envelope[stages=4],"
#effect_string += "SubtractiveSynth[sr=44100],"

#effect_string += "SubtractiveSynth[sr=44100],"
#effect_string += "Envelope[stages=4],"

effect_string += "ComplexOscillator[starting_freq=3.0,sr=44100],"
effect_string += "ComplexOscillator[starting_freq=1.5,sr=44100],"
#effect_string += "ParametricIRReverb[length=44100,sampling_rate=44100],"
#effect_string += "PeriodicAllPassFilter[order=3],"
effect_string += "SoftClipping,"
#effect_string += "Lowpass[sample_rate=44100],"
#effect_string += "NotchFilter[sample_rate=44100]"

effect_chain = build_effect_chain(effect_string)
synth = LearnableHarmonicSynth(sr, 10)
target_audio = target_audio.to(dev)
#loss = SpectrogramLoss(sr)
loss = MultiScaleSpectrogramLoss(sr)
loss_wrapper = lambda x: loss(x, target_audio)

synthAsEffect = LearnableMidiSynthAsEffect(sr, synth, effect_chain, midi_events)
synthAsEffect = synthAsEffect.to(dev)
if weights is not None:
    synthAsEffect.load_state_dict(torch.load(weights))

length = target_audio.shape[1]
input_audio = torch.zeros(length)
input_audio.requires_grad = False
input_audio = input_audio.to(dev)

optimizer = SingleSourceOptimizer(input_audio, sr, synthAsEffect, loss_wrapper)
output = optimizer.optimize(2000, 0.001)
output = output.unsqueeze(0)
torchaudio.save("output.wav", output.detach().cpu(), sr)
torch.save(synthAsEffect.state_dict(), "synth_as_effect.pth")