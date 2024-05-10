import torch
import torch.nn as nn
import torchaudio
import random

class SmallDenoisingConvolutionalAutoencoder(nn.Module):
    def __init__(self):
            # Encoder
        super(SmallDenoisingConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        #print("x shape", x.shape)
        x = self.decoder(self.encoder(x))
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
    def __init__(self, num_bands=512):
        self.model = SmallDenoisingConvolutionalAutoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.num_bands = num_bands
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")
    def train(self, waveform, num_steps=1000, batch_size_samples=32768):
        features = extract_eq_features(waveform, self.num_bands).to(self.dev)
        loss_ema = None
        for i in range(num_steps):
            self.optimizer.zero_grad()
            # Assemble a batch from a random index
            batch = features # TODO Restore batching logic...
            noise = torch.randn_like(batch)
            noise_scale = random.uniform(0.0, 10.0).to(self.dev)
            
            noise = noise * noise_scale
            print("Noise max, features max", noise.abs().max().item(), batch.abs().max().item())            
            batch = batch
            batch = batch / batch.abs().max()
            output = self.model(batch+noise)
            loss = self.criterion(output, batch)
            loss.backward()
            loss_ema = loss if loss_ema is None else 0.99 * loss_ema + 0.01 * loss
            print("Loss at step", i, "is", loss_ema.item())
            print("Noise squared sum", torch.mean(noise**2).item())
            self.optimizer.step()
        return self.model
    
class AutoencoderLoss(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False
            if torch.cuda.is_available():
                self.dev = torch.device("cuda")
            else:
                self.dev = torch.device("cpu")
            self.model = self.model.to(self.dev)
        def forward(self, x):
            x = extract_eq_features(x, 256)
            out = self.model(x)
            loss = torch.mean((out - x)**2)
          #  loss.backward()
            #primary_loss = torch.norm()
            return loss
            
            
    