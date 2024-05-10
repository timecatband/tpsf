
import torch
import torch.nn as nn
from effects.dynamics import LearnableASR
from effects.distortion import SoftClipping
from effects.filters import LearnableLowpass, LearnableHighpass
import torchaudio
from effects.equalizers import NotchFilter


class LearnableSineOscillator(nn.Module):
    def __init__(self, freq_rad, sr):
        super(LearnableSineOscillator, self).__init__()
        self.freq_rad = nn.Parameter(torch.tensor([freq_rad]))
        self.sr = sr
        self.phase = nn.Parameter(torch.tensor([0.0]))
        self.phase.requires_grad = True
        self.amplitude_parameters = nn.Parameter(torch.tensor([1.0,0.5,0.25,.125,0.0625,0.03125,0.015625]))
    def forward(self, num_samples):
        time = torch.linspace(0, num_samples / self.sr, num_samples)
        x = self.freq_rad * time * self.sr
        waveform = torch.sin(x+self.phase)
        # Add harmonics
        waveform += torch.sin(2 * x+self.phase) * self.amplitude_parameters[0].clamp(0, 1)
        waveform += torch.sin(3 * x+self.phase) * self.amplitude_parameters[1].clamp(0, 1)
        waveform += torch.sin(4 * x+self.phase) * self.amplitude_parameters[2].clamp(0, 1)
        waveform += torch.sin(5 * x+self.phase) * self.amplitude_parameters[3].clamp(0, 1)
        waveform += torch.sin(6 * x+self.phase) * self.amplitude_parameters[4].clamp(0, 1)
        waveform += torch.sin(7 * x+self.phase) * self.amplitude_parameters[5].clamp(0, 1)
        waveform += torch.sin(8 * x+self.phase) * self.amplitude_parameters[6].clamp(0, 1)
        
       # waveform = waveform / waveform.abs().max()
        return waveform
    def print(self):
        print("freq_rad: ", self.freq_rad)
        print("phase: ", self.phase)
        freq_hz = self.freq_rad * self.sr / (2 * 3.14159)
        print("freq_hz: ", freq_hz)
        
class HarmonicScaler(nn.Module):    
    def __init__(self, num_harmonics, time_latent_size):
        super(HarmonicScaler, self).__init__()
        self.scale_harmonic_amps_by_freq = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_harmonics)
        )
        self.scale_harmonic_amps_by_time_latent = nn.Sequential(
            nn.Linear(time_latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_harmonics)
        )
        self.scale_harmonic_amps_by_fundamental_amplitude = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8,num_harmonics)
        )
        self.scale_fundamental_amplitude = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
        self.num_harmonics = num_harmonics
        if torch.cuda is not None and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    def forward(self, freq, global_time_latent, amplitude):
        #print("Using time latent: ", global_time_latent)
        hamps = torch.ones(self.num_harmonics).to(self.device)
        hamps = hamps*self.scale_harmonic_amps_by_freq(freq.unsqueeze(0).to(self.device))
        hamps = hamps*self.scale_harmonic_amps_by_time_latent(global_time_latent.unsqueeze(0).to(self.device)).squeeze(0)
        amplitude = self.scale_fundamental_amplitude(amplitude.unsqueeze(0).to(self.device)).squeeze(0)
        hamps = hamps * self.scale_harmonic_amps_by_fundamental_amplitude(amplitude).squeeze(0)
        return torch.cat((amplitude.unsqueeze(0), hamps), dim=-1)
    
def generate_sawtooth_wave(frequency, num_samples, sampling_rate=44100):
    """
    Generates a sawtooth wave with PyTorch.

    Args:
        frequency (float): Frequency of the wave in radians per second.
        num_samples (int): Number of samples to generate.
        sampling_rate (int, optional): Sampling rate of the signal. Defaults to 44100.

    Returns:
        torch.Tensor: A tensor containing the sawtooth wave samples.
    """

    # Time points 
    time = torch.arange(0, num_samples) / sampling_rate

    # Calculate the angular displacement at each time point
    angular_displacement = 2 * torch.pi * frequency * time  

    # Sawtooth wave (subtract from 1 for traditional downward ramp)
    waveform = 2 * (angular_displacement / (2 * torch.pi)) - 1  

    return waveform

        
class LearnableHarmonicSynth(nn.Module):
    def __init__(self, sr, num_harmonics, enable_amplitude_scaling=True, time_latent_size=2):
        super(LearnableHarmonicSynth, self).__init__()
        self.sr = sr
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.phase = nn.Parameter(torch.tensor(0.0))
        
       # self.harmonic_amplitudes = nn.Parameter(torch.ones(num_harmonics))
    #    if torch.backends.mps.is_available():
     #       self.device = torch.device("mps")
        if torch.cuda is not None and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print("Num harmonics: ", num_harmonics)
        # TODO Weird hack
        num_harmonics = 8
        self.num_harmonics = num_harmonics
        self.static_detune = nn.Parameter(torch.tensor(0.0))
        # Rescale harmonic amplitudes to decay
       # self.harmonic_amplitudes = harmonic_amplitudes / torch.arange(1, num_harmonics+1).float()
        self.num_partials = 4
        self.map_harmonic_amplitudes_and_freq_to_inharmonic_partials = nn.Sequential(
              nn.Linear(num_harmonics+1, 32),
              nn.ReLU(),
              nn.Linear(32, 32),
              nn.ReLU(),
              nn.Linear(32, self.num_partials*2),
         )
        self.inharmonic_envelope = LearnableASR(3)
        self.sawtooth_envelope = LearnableASR(3)
        self.inharmonic_distortion = SoftClipping()
        
    # TODO Modify this to accept amplitude latent
    def forward(self, freq, output_length_samples, hamps, t, pitches=None):
        
        t_float = t.float().item()
        freq_in = freq
        time = torch.linspace(t_float, t_float+output_length_samples / self.sr, output_length_samples).to(self.device)
        # TODO: Re-enable fine pitch
      #  if pitches is not None:
            # Convert pitches from hz to radians
       #     freq = torch.tensor(pitches).to(self.device) * 2 * 3.14159
        #    freq = freq / self.sr
            # Pad pitches to length of time
          #  freq = freq.unsqueeze(0).expand(output_length_samples, -1)
            
        # Pad freq with 0s to match time shape using pad
        # TODO: questionable
        #freq = freq.unsqueeze(0).expand(output_length_samples, -1)
        x = freq * time * self.sr
       # print("Max hamps: ", hamps.max())        
        scale = hamps[:,0]
        hamps = hamps[:,1:]
        
                
        waveform = scale*torch.sin(x+self.phase)
        for i in range(1, self.num_harmonics):
            # Convert frequency to hz and check if it is above half the sampling rate
            # TODO: Need max when re-enabling fine pitch
            freq_hz = freq * self.sr / (2 * 3.14159)

            if freq_hz * (i+1) > self.sr / 2:
                scale = torch.tensor([1e-4])
            waveform += scale * torch.sin((i+1) * x+self.phase) * hamps[:,i]
        
        starting_freq_tensor = torch.tensor([freq_in]).to(self.device).unsqueeze(0)
        freq_and_hamps = torch.cat((starting_freq_tensor, hamps), dim=-1)
        partials = self.map_harmonic_amplitudes_and_freq_to_inharmonic_partials(freq_and_hamps)
        partials_wf = torch.zeros_like(time)
       # print("Partials: ", partials)
        for i in range(self.num_partials):
            partial_freq = partials[:,i]*10 * freq_in
            partial_freq_hz = partial_freq * self.sr / (2 * 3.14159)
            scale = partials[:,self.num_partials+i]
            if partial_freq_hz > self.sr / 2:
                scale = torch.tensor([1e-4])
            
            
            partials_wf += scale*torch.sin(partial_freq * time * self.sr)
        #partials_wf = self.inharmonic_distortion(partials_wf,t)
        partials = partials / 8
        # TODO Restore partials
        #waveform += self.inharmonic_envelope(partials_wf, t)
      #  print("Partials: ", partials)
        # TODO...maybe this is bad
       # waveform = waveform / waveform.abs().max()
        waveform = waveform / hamps.abs().sum()
        #sawtooth = generate_sawtooth_wave(freq_in, output_length_samples, self.sr)
        #sawtooth = sawtooth * self.sawtooth_envelope(sawtooth, t)
        #waveform += sawtooth
        return waveform
    
class NullSynth(nn.Module):
    def __init__(self):
        super(NullSynth, self).__init__()
    def forward(self, x):
        return x
    


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
def karplus_strong_roll_vectorized(wavetable, n_samples, decay_factor, feedback_line, feedbackamt):
    wavetable_size = wavetable.size(0)
    # Initialize the output tensor
    samples = torch.empty(n_samples)

    # Create a temporary working copy of the wavetable
    current_wavetable = wavetable#.detach()
    #current_wavetable.requires_grad = False
    current_wavetable = current_wavetable.detach()
   # current_wavetable.requires_grad = True

    # Vectorized processing of the entire wavetable
    for i in range(n_samples // wavetable_size):
        start_index = i * wavetable_size        
        # Roll the wavetable to right by one position
        current_wavetable += feedbackamt*feedback_line[start_index:start_index + wavetable_size]        
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


class KarplusSynth(nn.Module):
    def __init__(self, sr):
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
    def forward(self, feedback_line, freq_rad: float, output_length_samples: int, h, t, pitches=None):
        latents = self.hamps_to_decay(h)
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
        out = karplus_strong_roll_vectorized(wavetable, output_length_samples, decay, feedback_line, feedbackamt)
        #print("out abs sum: ", out.abs().sum())
        out[-256:] *= self.fade_over_256
       # out *= h[:,0]
        #return out
        return self.envelope(out,t)*latents[:,4]
