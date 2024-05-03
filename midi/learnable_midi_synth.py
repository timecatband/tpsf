import torch
import torchaudio
import torch.nn as nn

class LearnableMidiSynth(nn.Module):
    def __init__(self, sr, synth, effect_chain):
        super().__init__()
        self.synth = synth
        self.effect_chain = effect_chain
        self.sr = sr
    def forward(self, note_events, duration_samples):
        output = torch.zeros(duration_samples)
        for freq_rad, start_sample, end_sample in note_events:
            if start_sample >= duration_samples:
                print("Skipping note" + str(start_sample) + " " + str(end_sample) + " " + str(duration_samples))
                continue
            output_length = min(end_sample, duration_samples) - start_sample
            segment = self.synth(freq_rad, output_length)
            if self.effect_chain is not None:
                segment = self.effect_chain(segment)
            output[start_sample:start_sample + output_length] += segment
        return output

class LearnableMidiSynthAsEffect(nn.Module):
    def __init__(self, sr, synth, effect_chain, note_events):
        super().__init__()
        self.lms = LearnableMidiSynth(sr, synth, effect_chain)
        self.note_events = note_events
    def forward(self, x):
        length_samples = x.shape[0]
        
        return self.lms(self.note_events, length_samples)+x