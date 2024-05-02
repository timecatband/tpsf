import torch
import torchaudio

import sys
from synth.complex_oscillator import LearnableSineOscillator
import numpy as np

# TODO: Doesn't work

dev = torch.device("cpu")
audio_file, sr = torchaudio.load(sys.argv[1])
#audio_file = audio_file[:, 20*sr:30*sr]
audio_file = audio_file.to(dev)

oscillator = LearnableSineOscillator(np.pi*.1, sr)
oscillator = oscillator.to(dev)

lr = 3e-4
optim = torch.optim.Adam(oscillator.parameters(), lr=lr)
num_steps = 10000
for i in range(num_steps):
    optim.zero_grad()
    wave = oscillator(audio_file.shape[1])
    # Scale wave to have same max as audio file
    wave = wave * (audio_file.abs().max()/wave.abs().max())
    mix = wave + audio_file
    mix = mix.clamp(-1, 1)
   # loss = torch.abs(mix).sum()
    #loss = -loss
    area_under_new_wave = torch.abs(wave).sum()
    area_under_old_wave = torch.abs(audio_file).sum()
    area_under_mix = torch.abs(mix).sum()
    loss = area_under_mix - (area_under_new_wave + area_under_old_wave)
    loss = -loss
    print("Step", i, "Loss", loss.item())
    print("Oscillator params", oscillator.print())
    loss.backward()
    optim.step()

#Normalize mix
mix = mix/torch.max(torch.abs(mix))
torchaudio.save("output.wav", mix.detach(), sr)
# Save input
audio_file = audio_file/torch.max(torch.abs(audio_file))
torchaudio.save("input.wav", audio_file.detach(), sr)