"""
β-Variational Autoencoder with KL annealing.
Loss = Reconstruction (MSE) + β * KL Divergence.
β-annealing prevents posterior collapse in early training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from . import register_model


@register_model("vae")
class VariationalAutoencoder(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dims=None, latent_dim: int = 16,
                 dropout: float = 0.2, beta: float = 1.0, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "unsupervised"
        self.latent_dim = latent_dim
        self.beta = beta
        self._current_beta = 0.0  # For warmup

        if hidden_dims is None:
            hidden_dims = [64, 32]

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
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        reversed_dims = [latent_dim] + list(reversed(hidden_dims))
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

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self._current_beta * kl_loss, recon_loss, kl_loss

    def set_beta(self, epoch: int, warmup_epochs: int = 10):
        """β-annealing: linearly increase β from 0 to target over warmup_epochs."""
        self._current_beta = min(self.beta, self.beta * epoch / max(warmup_epochs, 1))

    def get_loss_fn(self):
        return None  # Custom loss handled in loss_function

    def get_anomaly_scores(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            scores = ((x - recon) ** 2).mean(dim=-1).cpu().numpy()
        return scores
