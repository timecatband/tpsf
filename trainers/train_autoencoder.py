import torch
import torch.nn as nn
import torchaudio

class SmallDenoisingConvolutionalAutoencoder(nn.Module):
    def __init__(self, num_channels_in):
        super().__init__()
        # Simple convolutional autoencoder with no bottleneck
        self.autoencoder = nn.Sequential(
            nn.Conv1d(num_channels_in, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, num_channels_in, 3, padding=1),
        )

    def forward(self, x):
        #print("x shape", x.shape)
        x = self.autoencoder(x)
        return x
    
def extract_eq_features(waveform, num_bands=10):
    # Extract log-mel spectrogram features
    n_fft = (num_bands - 1) * 2
    mel_specgram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(waveform)
    log_mel_specgram = torchaudio.transforms.AmplitudeToDB()(mel_specgram)
    # Downscale (interpolate) y dimension to a fixed number of frequency bands

    return log_mel_specgram
class EqFeatureAutoencoderTrainer():
    # Train an autoencoder to model the distribution of eq features
    # in a single audio file
    def __init__(self, num_bands=64):
        self.model = SmallDenoisingConvolutionalAutoencoder(num_bands)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.num_bands = num_bands
    def train(self, waveform, num_steps=2000, batch_size_samples=32768):
        features = extract_eq_features(waveform, self.num_bands)
        loss_ema = None
        for i in range(num_steps):
            self.optimizer.zero_grad()
            # Assemble a batch from a random index
            start_index = torch.randint(0, features.size(2) - batch_size_samples, (1,))
            batch = features[:, :, start_index:start_index + batch_size_samples]
            print("feature shape", features.shape)
            # Add noise to batch
            noise = torch.randn_like(batch)
            batch = batch
            batch = batch / batch.abs().max()
            output = self.model(batch)#+noise)
            loss = self.criterion(output, batch)
            loss.backward()
            loss_ema = loss if loss_ema is None else 0.99 * loss_ema + 0.01 * loss
            print("Loss at step", i, "is", loss_ema.item())
            print("Noise squared sum", torch.mean(noise**2).item())
            self.optimizer.step()
        return self.model
    