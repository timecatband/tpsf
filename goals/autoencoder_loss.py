import torch
import torch.nn

class AutoencoderLoss(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
        # Ensure the weights on the autoencoder are frozen
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    def forward(self, x):
        # Return a loss based on how close x was to the distribution modelled by the autoencoder
        return torch.mean((self.autoencoder(x) - x).pow(2))