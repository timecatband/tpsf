import torch
import torchaudio
import torch.nn as nn

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
        power = spectrogram.abs().pow(2)
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
        self.mse = nn.MSELoss()

        # Create multiple spectrogram transforms
        self.spectrograms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                normalized=True
            ) for n_fft in self.n_fft_sizes
        ])

        # Manage device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)  # Move the entire module to the device

    def forward(self, x, y):
        total_loss = 0
        i = 0
        for spec in self.spectrograms:
            i = i + 1
            x_spec = spec(x).to(self.device)
            y_spec = spec(y).to(self.device)

            # Normalize with db
            x_spec = torchaudio.transforms.AmplitudeToDB()(x_spec)
            y_spec = torchaudio.transforms.AmplitudeToDB()(y_spec)

            direct_loss = self.loss_fn(x_spec, y_spec)
            entropy_loss = self.mse(compute_entropy(x_spec),compute_entropy(y_spec))
            print("Direct loss", direct_loss)
            print("Entropy loss", entropy_loss)
            total_loss += direct_loss + entropy_loss / float(i)
            

        return total_loss / len(self.spectrograms)  # Average loss across scales