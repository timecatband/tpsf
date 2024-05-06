import torchaudio
import torch
import torch.nn as nn
from effects.dynamics import LearnableASR
from effects.decorator import effect
from effects.filters import LearnableLowpass

def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

@effect("ParametricIRReverb")
class LearnableParametricIRReverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0.3, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.filter = LearnableLowpass(sampling_rate)
        self.register_buffer("t", t)
        self.fr_params = nn.Parameter(torch.randn(length))
        self.exponential_fade_over_time = torch.linspace(1, 0.0001, length)
        self.exponential_fade_over_time = torch.exp(self.exponential_fade_over_time)
        self.exponential_fade_over_time = self.exponential_fade_over_time / self.exponential_fade_over_time.max()
        self.smooth_fade_over_1000 = torch.linspace(1, 0.0001, 1000)
        self.one_at_length = torch.ones(length-1000).float()
        self.envelope = nn.Parameter(torch.cat((self.one_at_length, self.smooth_fade_over_1000)))
    def save_impulse(self, path):
        impulse = self.build_impulse().detach().squeeze(0).cpu()
        print("Impulse shape", impulse.shape)
        impulse = impulse.permute(1, 0)
        impulse = impulse / impulse.abs().max()
        torchaudio.save(path, impulse, self.sampling_rate)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse
    def apply_frequency_response(self, impulse):
        # FFT to convert the impulse to the frequency domain
        impulse_fft = torch.fft.fft(impulse, dim=-1)  # Ensure FFT is along the correct dimension
        print("Impulse fft shape", impulse_fft.shape)
        # Create frequency response based on self.fr_params
        # Make sure that the frequency response is shaped correctly:
        frequency_response = torch.exp(-nn.functional.softplus(self.fr_params))
        # Adjust dimensions to match [batch_size, signal_length, 1]
        frequency_response = frequency_response.unsqueeze(0).unsqueeze(-1)  # Adding necessary dimensions

        # Apply the frequency response
        modified_impulse_fft = impulse_fft * frequency_response  # Should now correctly apply without unintended broadcasting

        # Inverse FFT to convert back to the time domain
        modified_impulse = torch.fft.ifft(modified_impulse_fft, dim=1).real  # Ensure iFFT is along the correct dimension

        print("Impulse and modified shapes")
        print(impulse.shape, modified_impulse.shape)
        return modified_impulse*self.exponential_fade_over_time.unsqueeze(-1).unsqueeze(0)


    def forward(self, x, t):
        x = x.unsqueeze(0)
        lenx = x.shape[1]
        impulse = self.build_impulse()
        #impulse = self.apply_frequency_response(impulse)
        print('impulse shape', impulse.shape)
        #impulse = self.filter(impulse.squeeze(-1), t)
        # TODO Put back maybe?
        impulse = impulse * self.envelope.unsqueeze(-1)
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))
        
        
        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)
        return x.squeeze(0).squeeze(-1)

class IRReverb(nn.Module):
    def __init__(self, ir):
        super().__init__()
        self.ir = ir
    def forward(self, x):
        return torchaudio.functional.fftconvolve(x, self.ir, mode='full')
    
class SinAsModule(nn.Module):
    def __init__(self, freq=0.1):
        super().__init__()
        self.freq = freq
    def forward(self, x):
        return torch.sin(x*self.freq)

class LearnableIRReverbSinusoidal(nn.Module):
    def __init__(self, ir_length, num_sinusoids=32):
        super().__init__()
        self.ir_length = ir_length
        self.time_grid = nn.Parameter(torch.linspace(0, 1, ir_length), requires_grad=False)
        self.sinusoidal_net = nn.Sequential(
            nn.Linear(1, num_sinusoids),  # Input: time coordinate
            SinAsModule(),  # Sinusoidal activation
            nn.Linear(num_sinusoids, 1)   # Output: single IR value
        )    
        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        ir = self.sinusoidal_net(self.time_grid.unsqueeze(1)).squeeze(1)
        out = torchaudio.functional.fftconvolve(x, ir, mode='full')
        return out[:x.shape[0]] * self.blend + x *(1 - self.blend) 

@effect("LearnableIRReverb")
class LearnableIRReverb(nn.Module):
    def __init__(self, ir_length):
        super().__init__()
        self.ir = nn.Parameter(torch.randn(ir_length))
        self.blend = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        out = torchaudio.functional.fftconvolve(x, self.ir, mode='full')
        # Clip to input length
        return out[:x.shape[0]]*self.blend + x*(1-self.blend)

class LearnableNoiseReverb(nn.Module):
    def __init__(self, ir_length):
        super().__init__()
        self.reverb = LearnableIRReverb(ir_length)
        self.envelope = LearnableASR()
        self.blend = nn.Parameter(torch.tensor(0.01))
    def forward(self, x):
        noise = torch.randn_like(x)
        reverb = self.reverb(noise)
        reverb = self.envelope(reverb)
        return reverb*self.blend + x*(1-self.blend)

class ParametricEarlyReflections(nn.Module):
    def __init__(self, num_reflections, base_delay, decay_factor, filter_cutoff=None):
        super().__init__()
        self.num_reflections = nn.Parameter(torch.tensor(num_reflections, dtype=torch.int))
        self.base_delay = nn.Parameter(torch.tensor(base_delay))
        self.decay_factor = nn.Parameter(torch.tensor(decay_factor))

        if filter_cutoff is not None:
            self.filter = nn.Linear(1, 1)  # Simple low-pass filter
            self.filter.weight.data.fill_(filter_cutoff) 
            self.filter.bias.data.zero_()

    def forward(self, x):
        delays = torch.arange(self.num_reflections.item()) * self.base_delay
        amplitudes = self.decay_factor ** delays
        output = torch.zeros_like(x)

        for i in range(self.num_reflections.item()):
            delayed_x = torch.roll(x, shifts=int(delays[i]), dims=-1)  # Circular delay
            if hasattr(self, 'filter'):
                delayed_x = self.filter(delayed_x.unsqueeze(1)).squeeze(1)
            output += delayed_x * amplitudes[i]

        return output
