from effects.reverb import IRReverb
from effects.equalizers import BiquadEq, NotchFilter
import torch
import torch.nn as nn
import torchaudio
import sys
ir = sys.argv[1]
audio_file = sys.argv[2]
ir, sr = torchaudio.load(ir)
audio_file, sr = torchaudio.load(audio_file)
# Clip audio to only seconds 20-30
audio_file = audio_file[:, 60*sr:80*sr]

reverb = IRReverb(ir)
# Ensure reverb parameters are frozen (no training)
#for param in reverb.parameters():
 #   param.requires_grad = False
processed_waveform = reverb(audio_file)
torchaudio.save("output.wav", processed_waveform.detach(), sr)

eq = NotchFilter(sr)#BiquadEq(sr)
effect_chain = nn.Sequential(eq, reverb)

optim = torch.optim.AdamW(eq.parameters(), lr=0.01)
for i in range(40):
    optim.zero_grad()
    eqed_audio = eq(audio_file)
    # Rescale eqed audio to have same peak as original
    eqed_audio = eqed_audio * (audio_file.abs().max()/eqed_audio.abs().max())
    reverbed_audio = reverb(eqed_audio)
    reverb_no_eq = reverb(audio_file)
    # Clip
    reverbed_audio = torch.clamp(reverbed_audio, -1, 1)
    revebered_no_eq = torch.clamp(reverb_no_eq, -1, 1)
    loss = -(reverbed_audio.abs().sum()-reverb_no_eq.abs().sum())
    loss += torch.sqrt((audio_file.abs().sum()-eqed_audio.abs().sum())**2)
   # loss = -(reverbed_audio.abs().sum()-eqed_audio.abs().sum())
    print("Step", i, "Loss", loss.item())
    print("Eq params", eq.print())
    loss.backward()
    optim.step()
    
processed_waveform = eq(audio_file)
# Rescale processed waveform to have same peak as input
processed_waveform *= audio_file.abs().max()/processed_waveform.abs().max()
torchaudio.save("output.wav", processed_waveform.detach(), sr)
processed_waveform = effect_chain(audio_file)
processed_waveform = processed_waveform/torch.max(torch.abs(processed_waveform))
torchaudio.save("output_witih_verb.wav", processed_waveform.detach(), sr)
# Output with original with verb only
processed_waveform = reverb(audio_file)
processed_waveform = processed_waveform/torch.max(torch.abs(processed_waveform))
torchaudio.save("output_verb_only.wav", processed_waveform.detach(), sr)