import torch
import torch.nn as nn
import torchaudio

class SmallDenoisingConvolutionalAutoencoder(nn.Module):
    def __init__(self, num_channels_in):
        super().__init__()
        # Simple convolutional autoencoder with no bottleneck
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels_in, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, num_channels_in, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def extract_eq_features(waveform, num_bands=10):
    # Extract log-mel spectrogram features
    n_fft = (num_bands - 1) * 2
    mel_specgram = torchaudio.transforms.Spectrogram()(waveform, n_fft=n_fft)
    log_mel_specgram = torchaudio.transforms.AmplitudeToDB()(mel_specgram)
    # Downscale (interpolate) y dimension to a fixed number of frequency bands
    
class EqFeatureAutoencoderTrainer():
    # Train an autoencoder to model the distribution of eq features
    # in a single audio file
    def __init__(self, num_bands=10):
        self.model = SmallDenoisingConvolutionalAutoencoder(num_bands)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.num_bands = num_bands
    def train(self, waveform, num_steps=1000, batch_size_samples=8192):
        features = extract_eq_features(waveform, self.num_bands)
        for i in range(num_steps):
            self.optimizer.zero_grad()
            # Assemble a batch from a random index
            start_index = torch.randint(0, features.size(1) - batch_size_samples, (1,))
            batch = features[:, start_index:start_index + batch_size_samples]
            output = self.model(batch)
            loss = self.criterion(output, batch)
            loss.backward()
            self.optimizer.step()
        return self.model
    