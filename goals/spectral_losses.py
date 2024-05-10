import torch
import torchaudio
import torch.nn as nn
import random
from PIL import Image
import numpy as np

class SpectrogramLoss(nn.Module):
    def __init__(self, sr):
        super().__init__()
        #self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, normalized=False)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.spec.to(self.device)
    def forward(self, x, y):
        x = self.spec(x)
        y = self.spec(y)
        # Normalize with db
        x = torchaudio.transforms.AmplitudeToDB()(x)
        y = torchaudio.transforms.AmplitudeToDB()(y)
        return self.loss(x, y)
    
def compute_entropy(spectrogram):
        power = spectrogram
        probability_distribution = torch.nn.functional.softmax(power, dim=-1)
        entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + 1e-8), dim=-1)  # Sum over frequency bins
        return entropy
class MultiScaleSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate, n_fft_sizes=(2048, 1024, 512), loss_type='L1'):
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
        self.mse = nn.MSELoss(reduction='none')

        # Create multiple spectrogram transforms
        self.spectrograms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_fft // 2,
                hop_length=n_fft // 2,
           ) for n_fft in self.n_fft_sizes
        ])
      #  self.spectrograms = nn.ModuleList([
     #       torchaudio.transforms.Spectrogram(
       #         n_fft=n_fft,
        #        win_length=n_fft,  # You can adjust this if needed
         #       hop_length=n_fft // 2,
          #      window_fn=torch.hamming_window,
           #     power=2,  # Use power spectrogram
            #    normalized=False
            #) for n_fft in self.n_fft_sizes
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
                x_spec = torchaudio.transforms.AmplitudeToDB()(x_spec)
                y_spec = torchaudio.transforms.AmplitudeToDB()(y_spec)
                y_spec = y_spec[0]
                print("x_spec shape", x_spec.shape)
                print("y_spec shape", y_spec.shape)

                # Compute direct loss
                direct_loss = self.loss_fn(x_spec, y_spec)

                # Compute entropy
                x_entropy = compute_entropy(x_spec)
                y_entropy = compute_entropy(y_spec)
                entropy_loss = self.mse(x_entropy, y_entropy)

                # Compute weights based on target entropy
                # Normalize target entropy weights to be between 0 and 1
                max_entropy = torch.max(y_entropy)
                entropy_weights = y_entropy / max_entropy
                direct_weights = 1 - entropy_weights

                # Weight the losses for each time bin
                weighted_entropy_loss = entropy_loss * entropy_weights
                weighted_direct_loss = direct_loss * direct_weights

                # Combine and average the weighted losses
                combined_loss = torch.mean(weighted_entropy_loss + weighted_direct_loss)
                
                print("Weighted direct loss", torch.mean(weighted_direct_loss).item())
                print("Weighted entropy loss", torch.mean(weighted_entropy_loss).item())

                # Sum losses from all spectrograms
                total_loss += combined_loss
                # Every 100 losses (random) dump both spectrograms as images
                if random.randint(0, 10) == 5:
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
