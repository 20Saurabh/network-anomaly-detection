"""
Contrastive Self-Supervised Learning for Network Anomaly Detection.
Pre-trains a feature extractor using NT-Xent (Normalized Temperature-scaled
Cross-Entropy) loss on UNLABELED data, then evaluates with a linear probe.

This demonstrates the framework's value in label-scarce regimes —
a critical requirement for real-world deployment and top-tier publication.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseModel
from . import register_model


class ContrastiveBackbone(nn.Module):
    """MLP backbone for feature extraction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 projection_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, F.normalize(z, dim=-1)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss."""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2*batch, proj_dim)

    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: (i, i+batch) and (i+batch, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


class TabularAugmenter:
    """Data augmentation for tabular data in contrastive learning."""

    def __init__(self, mask_ratio=0.15, noise_std=0.1):
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def __call__(self, x):
        """Apply random augmentation to create two views."""
        view1 = self._augment(x)
        view2 = self._augment(x)
        return view1, view2

    def _augment(self, x):
        augmented = x.clone()
        # Feature masking
        mask = torch.rand_like(x) < self.mask_ratio
        augmented[mask] = 0.0
        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        augmented = augmented + noise
        return augmented


@register_model("contrastive_ssl")
class ContrastiveSSL(BaseModel):
    """
    Self-supervised contrastive pre-training + linear probe.
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = 256, projection_dim: int = 64,
                 temperature: float = 0.5, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"  # After probe is attached
        self.temperature = temperature

        self.backbone = ContrastiveBackbone(
            input_dim, hidden_dim, projection_dim, dropout
        )

        self.augmenter = TabularAugmenter()

        # Linear probe (trained after pre-training)
        self.linear_probe = nn.Linear(hidden_dim // 2, num_classes)

        self._pretrained = False

    def forward(self, x, **kwargs):
        h, z = self.backbone(x)
        if self._pretrained:
            return self.linear_probe(h.detach() if not self.training else h)
        return z  # During pre-training, return projections

    def pretrain_step(self, x):
        """One step of contrastive pre-training."""
        view1, view2 = self.augmenter(x)
        _, z1 = self.backbone(view1)
        _, z2 = self.backbone(view2)
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss

    def freeze_backbone(self):
        """Freeze backbone after pre-training, train only the linear probe."""
        self._pretrained = True
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.linear_probe.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze everything for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
