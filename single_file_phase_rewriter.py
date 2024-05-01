from goals.unsupervised_waveform_losses import *
from trainers.single_source_optimizer import SingleSourceOptimizer
from effects.spectral_phase import PhaseDistortionFieldEffect, PhaseConvolverEffect
import sys
import torch
import torchaudio

#dev = torch.device("mps")
dev = torch.device("cpu")

num_steps = 500
lr = 3e-5
audio_file = sys.argv[1]

audio_file, sr = torchaudio.load(audio_file)
audio_file = audio_file[:, 40*sr:60*sr]
audio_file = audio_file.to(dev)

#effect_pipeline = PhaseDistortionFieldEffect()
effect_pipeline = PhaseConvolverEffect()
effect_pipeline = effect_pipeline.to(dev)
objective = total_energy_loss
optimizer = SingleSourceOptimizer(audio_file, sr, effect_pipeline, objective)
optimized_waveform = optimizer.optimize(num_steps, lr)
torchaudio.save("output.wav", optimized_waveform.detach().cpu(), sr)

