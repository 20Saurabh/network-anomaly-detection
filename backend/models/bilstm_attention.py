"""
Bidirectional LSTM with Multi-Head Attention.
Attention allows the model to focus on the most informative timesteps
rather than relying solely on the final hidden state.
"""
import torch
import torch.nn as nn
from .base import BaseModel
from . import register_model


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling over LSTM outputs."""

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(attn_out + x)
        # Pool: mean over sequence
        return x.mean(dim=1), attn_weights


@register_model("bilstm_attention")
class BiLSTMAttention(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, n_heads: int = 4, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = MultiHeadAttentionPooling(hidden_dim * 2, n_heads)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, **kwargs):
        # x: (batch, features) → (batch, 1, features) → project → (batch, 1, hidden)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.size(-1) == self.input_dim:
            x = self.input_projection(x)

        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        pooled, _ = self.attention(lstm_out)  # (batch, hidden*2)
        return self.classifier(pooled)
