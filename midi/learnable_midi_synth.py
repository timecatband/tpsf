import torch
import torchaudio
import torch.nn as nn
from effects.reverb import LearnableParametricIRReverb

class TimeDistortionField(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.distort_time = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, t):
        return self.distort_time(t)

class LearnableMidiSynth(nn.Module):
    def __init__(self, sr, synth, effect_chain, latent_embedder, enable_time_latent = True, time_latent_size = 2):
        super().__init__()
        self.synth = synth
        self.effect_chain = effect_chain
        self.sr = sr
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # TODO Questionable
        self.window_length = 512
        self.window = torch.hann_window(self.window_length).to(self.device)
        self.window = torch.linspace(1, 0, self.window_length).to(self.device)
        self.enable_time_latent = enable_time_latent
        self.time_latent_size = time_latent_size
        self.time_embedder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, time_latent_size)
        )
        self.harmonic_embedder = latent_embedder
      #  self.karplus_synth = torch.jit.script(self.karplus_synth)
        self.blend = nn.Parameter(torch.tensor([0.5]))
        self.time_distortion = TimeDistortionField(64)
        
    def forward(self, note_events, output, t, pitch=None):
        duration_samples = output.shape[0] #+ self.window_length
        t = torch.tensor([t]).to(self.device)
        for freq_rad, velocity, start_sample, end_sample in note_events:
            if start_sample >= duration_samples:
                print("Skipping note" + str(start_sample) + " " + str(end_sample) + " " + str(duration_samples))
                continue
            output_length = min(end_sample, duration_samples) - start_sample
            extended_length = output_length #+ self.window_length
            extended_length = min(extended_length, duration_samples - start_sample)
            if start_sample + extended_length >= duration_samples:
                extended_length = duration_samples - start_sample
            normalized_global_t_start = start_sample / duration_samples
            time_latent = None
            if self.enable_time_latent:
                time_latent = self.time_embedder(torch.tensor([normalized_global_t_start]).to(self.device).unsqueeze(0)).squeeze(0)
            velocity = velocity / 127.0
            hamps = self.harmonic_embedder(torch.tensor([freq_rad]), time_latent, torch.tensor([velocity]))
            
            pitches_in_segment = None
            if pitch is not None:
                pitches_in_segment = pitch[start_sample:start_sample + extended_length]
            segment = self.synth(None, freq_rad, output_length, hamps, t, pitches=pitches_in_segment)
           # segment1 = segment1
            #segment1 = segment1 / segment1.abs().max()
            #segment2 = segment2 / segment2.abs().max()
            # Apply the tapering window
            #print("Blend", self.blend)
         #   blend = self.blend.clamp(0.3, 0.7)
            #print("Segment 1 and 2 abs sum", torch.abs(segment1).sum(), torch.abs(segment2).sum())
            # TODO Restore mixing segment 1
            #segment = segment2
            
            if self.effect_chain is not None:
                segment = self.effect_chain(segment, t)
            segment[-self.window_length:] *= self.window                
            # Clip to output length
            segment = segment[:output_length]
            if start_sample + output_length >= duration_samples:
                output_length = duration_samples - start_sample
           # print("Segment shape", segment.shape)
           # print("Start sample + output length", output_length)
            # TODO Maybe just messes everything up
            start_sample_t = start_sample / float(duration_samples)
            d_t = self.time_distortion(torch.tensor([start_sample_t]).to(self.device))
            d_t = d_t/100.0
            d_t = d_t.clamp(-0.01, 0.01)
            d_t = d_t + start_sample_t
            start_sample = int(d_t * duration_samples)
            if start_sample + output_length >= duration_samples:
                output_length = duration_samples - start_sample
            
            output[start_sample:start_sample + output_length] += segment[:output_length]
        return output

class LearnableMidiSynthAsEffect(nn.Module):
    def __init__(self, sr, synth, effect_chain, note_events, latent_embedder,  pitch = None, enable_room_reverb = True):
        super().__init__()
        self.lms = LearnableMidiSynth(sr, synth, effect_chain, latent_embedder)
        self.note_events = note_events
        # TODO Turn back on!!!
        self.enable_room_reverb = True #enable_room_reverb
        if self.enable_room_reverb:
            self.verb = LearnableParametricIRReverb(int(sr/2), sr)
        self.pitch = pitch
    def forward(self, x):
        length_samples = x.shape[0]
        
        x = self.lms(self.note_events, x, 0.0, self.pitch)
        # TODO REENABLE!
        if (self.enable_room_reverb):
            # Questionable 0.0
            x = self.verb(x, 0.0)
        return x