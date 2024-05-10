import torch
import torch.nn as nn
class SynthSequence(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    def forward(self, feedback_line, freq_rad: float, output_length_samples: int, latents, t, pitches=None):
        for module in self.children():
            feedback_line = module(feedback_line, freq_rad, output_length_samples, latents, t, pitches)
        return feedback_line