import torch
import torch.nn as nn
from effects.dynamics import LearnableASR
from effects.distortion import SoftClipping
from effects.filters import LearnableLowpass, LearnableHighpass, LearnableBandreject
import torchaudio
from effects.equalizers import NotchFilter
from synth.decorator import synthd



def karplus_strong_torch(wavetable, n_samples: int, decay_factor):
    samples = []
    current_sample = 0
    previous_value = 0
    wavetable_size = wavetable.size(0)
    while len(samples) < n_samples:
        new_value = decay_factor * 0.5 * (wavetable[current_sample] + previous_value)
        wavetable[current_sample] = new_value
        samples.append(new_value)
        previous_value = new_value
        current_sample = (current_sample + 1) % wavetable_size
    return torch.tensor(samples)

def karplus_strong_vectorized(wavetable, n_samples, decay_factor):
    wavetable_size = wavetable.size(0)
    samples = torch.zeros(n_samples)

    # Use roll to simulate the shifting in the wavetable
    rolled_wavetable = torch.roll(wavetable, shifts=-1, dims=0)

    # Apply the Karplus-Strong update rule using vector operations
    # Compute the decayed average of the current and next sample in the wavetable
    decayed_values = decay_factor * 0.5 * (wavetable + rolled_wavetable)
    
    # Use index manipulation to simulate repeated usage of wavetable values
    indices = torch.arange(n_samples) % wavetable_size
    samples = decayed_values[indices]

    # Update the wavetable for feedback
    wavetable[indices] = samples

    return samples
def karplus_strong_roll_vectorized(wavetable, n_samples, decay_factor, feedback_line, feedbackamt, distortion=None):
    wavetable_size = wavetable.size(0)
    # Initialize the output tensor
    samples = torch.empty(n_samples).to(wavetable.device)

    # Create a temporary working copy of the wavetable
    current_wavetable = wavetable#.detach()
    #current_wavetable.requires_grad = False
    current_wavetable = current_wavetable.detach()
   # current_wavetable.requires_grad = True
    feedback_amt = feedbackamt.clamp(0.01,1)

    # Vectorized processing of the entire wavetable
    for i in range(n_samples // wavetable_size):
        start_index = i * wavetable_size        
        # Roll the wavetable to right by one position
        
        feedback = feedbackamt*feedback_line[start_index:start_index + wavetable_size]
        feedback = distortion(feedback, torch.tensor([0.0]))       
        current_wavetable += feedback
        rolled_wavetable = torch.roll(current_wavetable, shifts=1, dims=0)
        
        # Update wavetable using decayed average of current and rolled values
        current_wavetable = decay_factor * 0.5 * (current_wavetable + rolled_wavetable)
        
        # Save the processed chunk to the samples tensor
        

        samples[start_index:start_index + wavetable_size] = current_wavetable

    # Handle remaining samples for cases where n_samples isn't a perfect multiple of wavetable_size
    remaining_samples = n_samples % wavetable_size
    if remaining_samples > 0:
        rolled_wavetable = torch.roll(current_wavetable, shifts=1, dims=0)
        current_wavetable = decay_factor * 0.5 * (current_wavetable + rolled_wavetable)
        samples[-remaining_samples:] = current_wavetable[:remaining_samples]

    return samples


@synthd("KarplusSynth")
class KarplusSynth(nn.Module):
    def __init__(self, sr, latent_start_dim=0):
        super(KarplusSynth, self).__init__()
        self.sr = sr
        self.buffer = torch.zeros(sr)
        self.buffer_length = sr
        self.index = 0
        self.decay = nn.Parameter(torch.tensor(0.99))  # Decay parameter as a learnable parameter
        self.envelope = LearnableASR(3)
        self.hamps_to_decay = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.ReLU(),
        )
        self.lowpass = LearnableLowpass(sr, 8000.0)
        self.lowpass2 = LearnableLowpass(sr, 8000.0)
        self.highpass = LearnableHighpass(sr, 20.0)
        self.fade_over_256 = torch.linspace(1, 0, 256)
        
        self.distortion = SoftClipping()
        if torch.cuda is not None and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.fade_over_256 = self.fade_over_256.to(self.device)   
        self.latent_start_dim = latent_start_dim
        self.num_latents = 7      
        # TODO Disable
        self.distortion = SoftClipping()
        
    def forward(self, feedback_line, freq_rad: float, output_length_samples: int, h, t, pitches=None):
        latents = h[:,self.latent_start_dim:self.latent_start_dim+self.num_latents]
        decay = latents[:,0]#self.decay.clamp(0.99, 0.99999)
        #decay = self.decay
        #print("Decay: ", decay)
        decay = decay / 10.0 + 0.9
        decay = decay.clamp(0.9, 0.999)
        decay = decay.squeeze(0)
    #    decay = self.decay
        freq = freq_rad * self.sr / (2 * 3.14159)
        wavetable_size = int(self.sr // freq)
      #  wavetable = 2 * torch.randint(0, 2, (wavetable_size,), dtype=torch.float32) - 1        
        wavetable = 2*torch.rand(wavetable_size).to(self.device) - 1
        wavetable = wavetable
        lowpass_freq = latents[:,1]*self.sr/4
        lowpass_q = latents[:,2]
        lowpass_freq = lowpass_freq.clamp(100, self.sr / 2 - 1)
        lowpass_q = lowpass_q.clamp(0.1, 0.999)
        wavetable = torchaudio.functional.lowpass_biquad(
            wavetable,
            self.sr,    # Sample rate
            lowpass_freq,
            lowpass_q
        )
        wavetable = self.lowpass(wavetable, t)
      #  wavetable = self.highpass(wavetable, t)
       # wavetable = self.envelope(wavetable, t)
      
       # wavetable = self.envelope(wavetable, t)
      #  wavetable = wavetable.detach().requires_grad_(False)
        
        feedbackamt = latents[:,3]
        out = karplus_strong_roll_vectorized(wavetable, output_length_samples, decay, feedback_line, feedbackamt, self.distortion)
        #print("out abs sum: ", out.abs().sum())
        out[-256:] *= self.fade_over_256
       # out *= h[:,0]
        #return out
        bandpass_freq = latents[:,5]*self.sr/4
        bandpass_freq = bandpass_freq.clamp(100, self.sr / 2 - 1)
        bandpass_q = latents[:,6]
        bandpass_q = bandpass_q.clamp(0.1, 0.999)
        out = torchaudio.functional.bandreject_biquad(out, self.sr, bandpass_freq, bandpass_q)
        return self.envelope(out,t)*latents[:,4]
