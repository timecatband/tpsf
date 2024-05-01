import torch

# Maximize the area under a single waveform, only useful
# when optimizing things that can't directly impact amplitude...
def total_energy_loss(waveform):
    waveform = waveform.abs()
    return -torch.sum(waveform)

# Encourages the area under the mixed waveform to be greater than the sum of the areas
# under the individual waveforms. This can guide us towards constructive interference
# TODO: Should this have clipping?
def interference_loss(waveform1, waveform2):
    a1 = torch.sum(waveform1.abs())
    a2 = torch.sum(waveform2.abs())
    mixed = waveform1 + waveform2
    am = torch.sum(mixed.abs())
    return am - (a1 + a2)

class TotalEnergyLoss(torch.nn.Module):
    def forward(self, waveform):
        return total_energy_loss(waveform)
class InterferenceLoss(torch.nn.Module):
    def forward(self, waveform1, waveform2):
        return interference_loss(waveform1, waveform2)