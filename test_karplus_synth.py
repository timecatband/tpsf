from synth.oscillator import KarplusSynth
import torch
import torchaudio

kps = KarplusSynth(44100)
#kps = torch.jit.script(kps)
freq_hz = 110
freq_rad = 2 * 3.14159 * freq_hz
freq_rad = freq_rad / 44100
audio = kps(freq_rad, 44100, None, torch.tensor([1.0]), None)
print(audio.shape)
audio = audio.unsqueeze(0)
torchaudio.save("karplus_synth.wav", audio.detach(), 44100)