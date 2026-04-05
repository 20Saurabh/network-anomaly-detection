"""
1D-CNN with Residual Connections for tabular classification.
Features are treated as a 1D signal; residual skip connections
prevent gradient degradation in deeper networks.
"""
import torch
import torch.nn as nn
from .base import BaseModel
from . import register_model


class ResidualBlock1D(nn.Module):
    """1D Convolutional residual block."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


@register_model("cnn1d")
class CNN1DClassifier(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 channels=None, kernel_size: int = 3,
                 dropout: float = 0.3, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"

        if channels is None:
            channels = [32, 64, 128]

        # Input projection: (batch, features) → (batch, 1, features)
        layers = [
            nn.Conv1d(1, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        ]

        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size,
                                     padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(ResidualBlock1D(channels[i + 1], kernel_size))
            layers.append(nn.MaxPool1d(2))

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, **kwargs):
        # x: (batch, features) → (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.classifier(x)
