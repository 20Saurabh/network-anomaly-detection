"""
Vanilla Autoencoder with BatchNorm + Dropout regularization.
Anomaly detection via reconstruction error (MSE).
"""
import torch
import torch.nn as nn
from .base import BaseModel
from . import register_model


@register_model("vanilla_ae")
class VanillaAutoencoder(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dims=None, dropout: float = 0.2, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "unsupervised"

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.BatchNorm1d(reversed_dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, **kwargs):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        return self.encoder(x)

    def get_loss_fn(self):
        return nn.MSELoss()
