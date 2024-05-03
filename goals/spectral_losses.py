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