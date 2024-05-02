import torch
import torchaudio
import torch.nn as nn
import sys
from synth.complex_oscillator import LearnableSineOscillator
import synth.oscillator
from effects.reverb import IRReverb
from goals.reverb import ReverbBoostingLoss

ir_path = sys.argv[1]
ir, sample_rate = torchaudio.load(ir_path)

seconds_of_audio_to_generate = 4
oscillator = synth.oscillator.LearnableSineOscillator(0.4, sample_rate)
#oscillator.initial_phase.requires_grad = False
reverb = IRReverb(ir)
optim = torch.optim.AdamW(oscillator.parameters(), lr=1e-4)
num_steps = 20000
loss_fn = ReverbBoostingLoss(reverb)
for i in range(num_steps):
    optim.zero_grad()
    audio = oscillator(seconds_of_audio_to_generate)
    audio = reverb(audio.unsqueeze(0))
    loss = torch.abs(audio).sum()
    print("Step", i, "Loss", loss.item())
    print("Oscillator params", oscillator.print())
    loss.backward()
    optim.step()
with torch.no_grad():
    oscillator = synth.oscillator.LearnableSineOscillator(oscillator.freq_rad / 16, sample_rate)
    audio = oscillator(seconds_of_audio_to_generate*sample_rate)
    oscillator = synth.oscillator.LearnableSineOscillator(oscillator.freq_rad + 0.001, sample_rate)
    other_audio = oscillator(seconds_of_audio_to_generate*sample_rate)
torchaudio.save("output.wav", audio.detach().unsqueeze(0), sample_rate)
torchaudio.save("output2.wav", other_audio.detach().unsqueeze(0), sample_rate)