"""
CNN-LSTM Hybrid: CNN front-end for local feature extraction,
followed by LSTM for temporal sequence modeling.
"""
import torch
import torch.nn as nn
from .base import BaseModel
from . import register_model


@register_model("cnn_lstm")
class CNNLSTMHybrid(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 cnn_channels: int = 64, lstm_hidden: int = 128,
                 lstm_layers: int = 1, dropout: float = 0.3, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"

        # CNN front-end
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        cnn_out_dim = input_dim // 2  # after MaxPool1d(2)

        # LSTM backend
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, **kwargs):
        # x: (batch, features) → (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # CNN: (batch, 1, features) → (batch, channels, features//2)
        cnn_out = self.cnn(x)

        # Transpose for LSTM: (batch, seq_len, channels)
        lstm_in = cnn_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(lstm_in)

        # Use last timestep
        last_hidden = lstm_out[:, -1, :]

        return self.classifier(last_hidden)
