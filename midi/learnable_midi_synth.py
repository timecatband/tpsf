import torch
import torchaudio
import torch.nn as nn
from effects.reverb import LearnableParametricIRReverb
from synth.oscillator import HarmonicScaler
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
        self.harmonic_scaler = HarmonicScaler(8, time_latent_size)
        
    def forward(self, note_events, output, t):
        duration_samples = output.shape[0]
        for freq_rad, velocity, start_sample, end_sample in note_events:
            if start_sample >= duration_samples:
                print("Skipping note" + str(start_sample) + " " + str(end_sample) + " " + str(duration_samples))
                continue
            output_length = min(end_sample, duration_samples) - start_sample
            extended_length = output_length + self.window_length
            extended_length = min(extended_length, duration_samples - start_sample)
            normalized_global_t_start = start_sample / duration_samples
            time_latent = None
            if self.enable_time_latent:
                time_latent = self.time_embedder(torch.tensor([normalized_global_t_start]).to(self.device).unsqueeze(0)).squeeze(0)
            velocity = velocity / 127.0
            hamps, velocity = self.harmonic_scaler(freq_rad, time_latent, velocity)
            segment = self.synth(freq_rad, output_length, hamps, velocity, t)
            
            # Apply the tapering window
            segment[-self.window_length:] *= self.window
            if self.effect_chain is not None:
                segment = self.effect_chain(segment, t)
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
        
        x = self.lms(self.note_events, x, 0.0)
        if (self.enable_room_reverb):
            # Questionable 0.0
            x = self.verb(x, 0.0)
        return x