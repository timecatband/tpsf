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
export = False
if len(sys.argv) > 3:
    weights = sys.argv[3]
if len(sys.argv) > 4:
    if sys.argv[4] == "export":
        export = True
    
midi_events = process_midi(midi_file, sr)
synth = LearnableHarmonicSynth(sr, 10)
target_audio = target_audio.to(dev)
#loss = SpectrogramLoss(sr)
#effect_string = "SubtractiveSynth[sr=44100],"
effect_string = "Envelope[stages=4],"
#effect_string += "SubtractiveSynth[sr=44100],"

effect_string += "ComplexOscillator[starting_freq=3.0,sr=44100],"
effect_string += "ComplexOscillator[starting_freq=1.5,sr=44100],"
#effect_string += "ParametricIRReverb[length=44100,sampling_rate=44100],"
effect_string += "SoftClipping,"
#effect_string += "Lowpass[sample_rate=44100],"


effect_chain = build_effect_chain(effect_string)

synthAsEffect = LearnableMidiSynthAsEffect(sr, synth, effect_chain, midi_events)
# Load state dict
if weights is not None:
    synthAsEffect.load_state_dict(torch.load(weights))
synthAsEffect = synthAsEffect.to(dev)

synth_script = torch.jit.script(synthAsEffect.lms.harmonic_scaler)
synth_script.save("harmonic_scaler.pt")
print("Exported script")
effect_chain_script = torch.jit.script(synthAsEffect.lms.effect_chain)
effect_chain_script.save("effect_chain.pt")

#completeJit = torch.jit.script(synthAsEffect)
synthAsEffect.lms.effect_chain = effect_chain_script
with torch.no_grad():
    audio = synthAsEffect(torch.zeros(target_audio.shape[1]))
torchaudio.save("output.wav", audio.unsqueeze(0), sr)
synthAsEffect.verb.save_impulse("learned_impulse.wav")