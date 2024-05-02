import torch
import torch.nn as nn
class ReverbBoostingLoss(nn.Module):
    def __init__(self, reverb, original_audio=None):
        super().__init__()
        self.reverb = reverb
        self.original_audio = original_audio
    def forward(self, x):
        loss = -(self.reverb(x).abs().sum()-x.abs().sum())
        if self.original_audio is not None:
            loss += torch.sqrt((self.original_audio.abs().sum()-x.abs().sum())**2)
        return loss