import torch
import torchaudio
import torch.nn as nn
from effects.reverb import LearnableParametricIRReverb

class LearnableMidiSynth(nn.Module):
    def __init__(self, sr, synth, effect_chain, enable_time_latent = True, time_latent_size = 2):
        super().__init__()
        self.synth = synth
        self.effect_chain = effect_chain
        self.sr = sr
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # TODO Questionable
        self.window_length = 1024
        self.window = torch.hann_window(self.window_length).to(self.device)
        self.enable_time_latent = enable_time_latent
        self.time_latent_size = time_latent_size
        self.time_embedder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, time_latent_size)
        )
    def forward(self, note_events, duration_samples):
        output = torch.zeros(duration_samples).to(self.device)
        for freq_rad, velocity, start_sample, end_sample in note_events:
            if start_sample >= duration_samples:
                print("Skipping note" + str(start_sample) + " " + str(end_sample) + " " + str(duration_samples))
                continue
            output_length = min(end_sample, duration_samples) - start_sample
            extended_length = output_length + self.window_length
            extended_length = min(extended_length, duration_samples - start_sample)
            normalized_t_start = start_sample / duration_samples
            time_latent = None
            if self.enable_time_latent:
                time_latent = self.time_embedder(torch.tensor([normalized_t_start]).to(self.device).unsqueeze(0)).squeeze(0)
            segment = self.synth(freq_rad, output_length, time_latent, velocity/127.0)
            
            # TODO: Move this in to synth
            # Normalize midi velocity
            #segment *= velocity / 127.0
            
            # Apply the tapering window
            segment[-self.window_length:] *= self.window
            if self.effect_chain is not None:
                segment = self.effect_chain(segment)
            output[start_sample:start_sample + output_length] += segment
        return output

class LearnableMidiSynthAsEffect(nn.Module):
    def __init__(self, sr, synth, effect_chain, note_events, enable_room_reverb = True):
        super().__init__()
        self.lms = LearnableMidiSynth(sr, synth, effect_chain)
        self.note_events = note_events
        self.enable_room_reverb = enable_room_reverb
        if self.enable_room_reverb:
            self.verb = LearnableParametricIRReverb(sr, sr)
    def forward(self, x):
        length_samples = x.shape[0]
        
        out = self.lms(self.note_events, length_samples)+x
        if (self.enable_room_reverb):
            out = self.verb(out)
        return out