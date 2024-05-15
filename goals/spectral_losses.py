import torch
import torchaudio
import torch.nn as nn
import random
from PIL import Image
import numpy as np
from math import exp

def gaussian_kernel(size, sigma):
    """Generates a 1D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma.pow(2)))
    return gauss / gauss.sum()

def apply_variable_gaussian_blur(spectrogram, max_sigma, decay_rate=5.0):
    """
    Applies a variable Gaussian blur across the frequency bins of a spectrogram.
    max_sigma: Maximum sigma for the highest frequency bin.
    """
    num_bins = spectrogram.size(0)
    blurred_spectrogram = torch.zeros_like(spectrogram)
    decay_rate = torch.tensor(decay_rate).to(spectrogram.device)

    for i in range(num_bins-1, 0, -1):
        sigma = max_sigma * exp(-decay_rate*(i / num_bins))
        size = int(sigma * 6)  # Kernel size, typically 6*sigma to cover 99% of the Gaussian
        if size % 2 == 0:
            size += 1  # Ensure kernel size is odd
        if sigma > 0:
            kernel = gaussian_kernel(size, sigma)
            kernel = kernel.view(1, 1, -1).to(spectrogram.device)
            row_blurred = torch.nn.functional.conv1d(spectrogram[i:i+1, None, :], kernel, padding=size//2)
            blurred_spectrogram[i] = row_blurred
        else:
            blurred_spectrogram[i] = spectrogram[i]  # No blur if sigma is 0

    return blurred_spectrogram

import torch
import torch.nn.functional as F


def gaussian_2d_kernel(kernel_size, sigma):
    """Generate a 2D Gaussian kernel."""
    center = kernel_size // 2
    x = torch.arange(kernel_size).float() - center
    y = torch.arange(kernel_size).float() - center
    x, y = torch.meshgrid(x, y, indexing='xy')
    kernel = torch.exp(- (x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)  # Reshape to [out_channels, in_channels, height, width]

def apply_half_gaussian_blur(spectrogram, kernel_size, sigma):
    num_bins, num_steps = spectrogram.shape
    half_bins = num_bins // 2  # Half the frequency bins

    # Gaussian kernel
    kernel = gaussian_2d_kernel(kernel_size, sigma).to(spectrogram.device)

    print("Spec shape", spectrogram.shape)
    # Prepare for convolution
    top_half = spectrogram[half_bins:].unsqueeze(1)  # Get the top half and add channel dimension
    bottom_half = spectrogram[:half_bins]  # Bottom half remains unchanged

    # Apply Gaussian blur to the top half
    padding = kernel_size // 2
    print("top half shape", top_half.shape)
    top_half = top_half.permute((1,0,2))
    blurred_top_half = F.conv2d(top_half, kernel, padding=padding)
    blurred_top_half = blurred_top_half.squeeze(0).squeeze(0)  # Remove channel dimension

    # Concatenate back the blurred top half with the unblurred bottom half
    print("Blurred top half shape", blurred_top_half.shape)
    print("Bottom half shape", bottom_half.shape)
    # TODO Double weighting the blurry section is probably a bad idea but interesting...
    blurred_spectrogram = torch.cat((bottom_half, blurred_top_half), dim=0)# blurred_top_half), dim=0)

    return blurred_spectrogram


def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
#     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1
#     print("begin_front",begin_front)

    size = list(shape)
    size[axis] -= 1
#     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

#     slice_front = tf.slice(x, begin_front, size)
#     slice_back = tf.slice(x, begin_back, size)
#     print("slice_front",slice_front)
#     print(slice_front.shape)
#     print("slice_back",slice_back)

    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
#     print("dd",dd)
    ddmod = np.mod(dd+np.pi,2.0*np.pi)-np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
#     print("ddmod",ddmod)

    idx = np.logical_and(np.equal(ddmod, -np.pi),np.greater(dd,0)) # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
#     print("idx",idx)
    ddmod = np.where(idx, np.ones_like(ddmod) *np.pi, ddmod) # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
#     print("ddmod",ddmod)
    ph_correct = ddmod - dd
#     print("ph_corrct",ph_correct)
    
    idx = np.less(np.abs(dd), discont) # idx = tf.less(tf.abs(dd), discont)
    
    ddmod = np.where(idx, np.zeros_like(ddmod), dd) # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis) # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
#     print("idx",idx)
#     print("ddmod",ddmod)
#     print("ph_cumsum",ph_cumsum)
    
    
    shape = np.array(p.shape) # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis) 
    #ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
#     print("unwrapped",unwrapped)
    return unwrapped


def instantaneous_frequency(phase_angle, time_axis):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Args:
    phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
    time_axis: Axis over which to unwrap and take finite difference.
    Returns:
    dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
#     print("phase_unwrapped",phase_unwrapped.shape)
    
    dphase = diff(phase_unwrapped, axis=time_axis)
#     print("dphase",dphase.shape)
    
    # Add an initial phase to dphase
    size = np.array(phase_unwrapped.shape)
#     size = phase_unwrapped.get_shape().as_list()

    size[time_axis] = 1
#     print("size",size)
    begin = [0 for unused_s in size]
#     phase_slice = tf.slice(phase_unwrapped, begin, size)
#     print("begin",begin)
    phase_slice = phase_unwrapped[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1]]
#     print("phase_slice",phase_slice.shape)
    dphase = np.concatenate([phase_slice, dphase], axis=time_axis) / np.pi

#     dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
#     mag = np.complex(mag)
    temp_mag = np.zeros(mag.shape,dtype=np.complex_)
    temp_phase = np.zeros(mag.shape,dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
#             print(mag[i,j])
            temp_mag[i,j] = np.complex(mag[i,j])
#             print(temp_mag[i,j])
    
    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i,j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))
#             print(temp_mag[i,j])
    
#     phase = np.complex(np.cos(phase_angle), np.sin(phase_angle))
   
    return temp_mag * temp_phase

class LowpassSpectrogramLoss(nn.Module):
    def __init__(self, sr, lowpass_freq=10000.0, lowpass_q=0.1):
        super().__init__()
        
        self.loss = nn.L1Loss()
        n_fft = 2048
        overlap = 0.5
        self.spec = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=n_fft,  # You can adjust this if needed
                hop_length=int(n_fft * (1 - overlap)),
                window_fn=torch.hann_window,
                power=2,  # Use power spectrogram
                normalized=False,
                return_complex=True
            ) 
        #self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, normalized=False, n_fft=2048, n_mels=n_fft//2, hop_length=n_fft//4)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.spec.to(self.device)
        self.lowpass_freq = lowpass_freq
        self.lowpass_q = lowpass_q
        self.sr = sr
    def forward(self, x, y):
      #  x = torchaudio.functional.lowpass_biquad(x, self.sr, self.lowpass_freq, self.lowpass_q)
       # y = torchaudio.functional.lowpass_biquad(y, self.sr, self.lowpass_freq, self.lowpass_q)
        x = self.spec(x)
        y = self.spec(y)
        y = y[0]        
      #  x = instantaneous_frequency(x, 1)
       # y = instantaneous_frequency(y, 1)
        loss = self.loss(x, y)
        
        x = torch.log(x + 1e-4)
        y = torch.log(y + 1e-4)
        
        loss += self.loss(x, y)
        if (random.randint(0, 3) == 1):
            print("x shape", x.shape)
            print("y shape", y.shape)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x/np.max(x)
            y = y/np.max(y)
            x = x * 255
            y = y * 255
            x = x.astype(np.uint8)
            y = y.astype(np.uint8)
            x_pil = Image.fromarray(x)
            x_pil.save("x.png")
            y_pil = Image.fromarray(y)
            y_pil.save("y.png")       
        return loss 
        
        
        
      

class NoisySpectrogramLoss(nn.Module):
    def __init__(self, sr, denoise_count=2, denoise_level=0.05):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')
       # self.loss = nn.L1Loss()
        n_fft = 2048
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, normalized=False, n_fft=2048, n_mels=n_fft//2, hop_length=n_fft//4)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.spec.to(self.device)
        self.denoise_count = denoise_count
        self.denoise_level = denoise_level
    def forward(self, x, y):
        
        
        x = self.spec(x+torch.randn_like(x) * self.denoise_level)
        x = torch.log(x + 1e-7)
        total_loss = torch.zeros_like(x)

        for i in range(self.denoise_count):
            y_with_noise = y + torch.randn_like(y) * self.denoise_level
            y_spec = self.spec(y_with_noise)
            y_spec = y_spec[0]
            # Normalize with db
            y_spec = torch.log(y_spec + 1e-7)
            total_loss += self.loss(x, y_spec)
        if (random.randint(0, 10) == 1):
            print("x shape", x.shape)
            print("y shape", y.shape)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x/np.max(x)
            y = y/np.max(y)
            x = x * 255
            y = y * 255
            x = x.astype(np.uint8)
            y = y.astype(np.uint8)
            x_pil = Image.fromarray(x)
            x_pil.save("x.png")
            y_pil = Image.fromarray(y)
            y_pil.save("y.png")
        return total_loss.mean()
    

class SpectrogramLoss(nn.Module):
    def __init__(self, sr):
        super().__init__()
        self.loss = nn.MSELoss()
       # self.loss = nn.L1Loss()
        n_fft = 2048
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, normalized=False, n_fft=2048, n_mels=n_fft//2, hop_length=n_fft//4)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.spec.to(self.device)
    def forward(self, x, y):
        x = self.spec(x)
        y = self.spec(y)
        y = y[0]
        # Normalize with db
        x = torch.log(x + 1e-4)
        y = torch.log(y + 1e-4)
        loss = self.loss(x, y)
        if (random.randint(0, 10) == 1):
            print("x shape", x.shape)
            print("y shape", y.shape)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x/np.max(x)
            y = y/np.max(y)
            x = x * 255
            y = y * 255
            x = x.astype(np.uint8)
            y = y.astype(np.uint8)
            x_pil = Image.fromarray(x)
            x_pil.save("x.png")
            y_pil = Image.fromarray(y)
            y_pil.save("y.png")
        return loss
    
def compute_entropy(spectrogram):
        power = spectrogram
        probability_distribution = torch.nn.functional.softmax(power, dim=-1)
        entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + 1e-8), dim=-1)  # Sum over frequency bins
        return entropy
class MultiScaleSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate, n_fft_sizes=(2048, 1024,512,256,128), loss_type='L1', overlap=0.75):
        super().__init__()
        # TODO Disable
        #n_fft_sizes = (2048,)
        self.sample_rate = sample_rate
        self.n_fft_sizes = n_fft_sizes
        self.loss_type = loss_type

        # Initialize loss function
        if self.loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == 'MSE':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        self.mse = nn.L1Loss(reduction='none')

        # Create multiple spectrogram transforms
        self.spectrograms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_fft // 2,
                hop_length=n_fft // 4,
           ) for n_fft in self.n_fft_sizes
        ])
        #self.spectrograms = nn.ModuleList([
         #   torchaudio.transforms.Spectrogram(
         #       n_fft=n_fft,
         #       win_length=n_fft,  # You can adjust this if needed
         #       hop_length=int(n_fft * (1 - overlap)),
         #       window_fn=torch.hann_window,
         #       power=2,  # Use power spectrogram
         #       normalized=True,
         #       return_complex=True
         #   ) for n_fft in self.n_fft_sizes
        #])
        
        # Manage device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)  # Move the entire module to the device

    def forward(self, x, y):
            total_loss = 0
            for spec in self.spectrograms:
                x_spec = spec(x).to(self.device)
                y_spec = spec(y).to(self.device)

                # Normalize with dB
               # x_spec = torchaudio.transforms.AmplitudeToDB()(x_spec)
                #y_spec = torchaudio.transforms.AmplitudeToDB()(y_spec)
                direct_loss = self.loss_fn(x_spec, y_spec)
                x_spec = torch.log(x_spec + 1e-7)
                y_spec = torch.log(y_spec + 1e-7)
                
                y_spec = y_spec[0]
                print("x_spec shape", x_spec.shape)
                print("y_spec shape", y_spec.shape)

                # Compute direct loss
                direct_loss += self.loss_fn(x_spec, y_spec)

                # Compute entropy
              #  x_entropy = compute_entropy(x_spec)
               # y_entropy = compute_entropy(y_spec)
                # TODO This is currently really L1...maybe go back doenst seem great
                #entropy_loss = self.mse(x_entropy, y_entropy)

                # Compute weights based on target entropy
                # Normalize target entropy weights to be between 0 and 1
                #max_entropy = torch.max(y_entropy)
                #entropy_weights = y_entropy / max_entropy
                #direct_weights = 1 - entropy_weights

                # Weight the losses for each time bin
                #weighted_entropy_loss = entropy_loss * entropy_weights
                #weighted_direct_loss = direct_loss * direct_weights

                # Combine and average the weighted losses

                # Sum losses from all spectrograms
                total_loss += direct_loss
                # Every 100 losses (random) dump both spectrograms as images
                if random.randint(0, 5) == 3:
                    x_spec = x_spec.detach().cpu().numpy()
                    y_spec = y_spec.detach().cpu().numpy()
                    x_spec = x_spec/np.max(x_spec)
                    y_spec = y_spec/np.max(y_spec)
                    x_spec = x_spec * 255
                    y_spec = y_spec * 255
                    x_spec = x_spec.astype(np.uint8)
                    y_spec = y_spec.astype(np.uint8)
                    print("x_spec shape", x_spec.shape)
                    print("y_spec shape", y_spec.shape)
                    x_spec_pil = Image.fromarray(x_spec)
                    x_spec_pil.save("x_spec.png")
                    y_spec_pil = Image.fromarray(y_spec)
                    y_spec_pil.save("y_spec.png")
                

            return total_loss
