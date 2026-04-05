"""
Feature Tokenizer Transformer (FT-Transformer).
Each numerical feature gets its own learned embedding (token),
then all tokens attend to each other via Transformer encoder layers.
This is SOTA for tabular data (Gorishniy et al., 2021; ICLR spotlight).
"""
import math
import torch
import torch.nn as nn
from .base import BaseModel
from . import register_model


class NumericalFeatureTokenizer(nn.Module):
    """Embeds each scalar feature into d_model dimensions."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, d_model))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, n_features)
        # output: (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


@register_model("ft_transformer")
class FTTransformer(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 ffn_factor: float = 4.0/3.0, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"

        # Feature tokenizer
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * ffn_factor * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, **kwargs):
        # x: (batch, n_features)
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)

        # Prepend [CLS]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, d_model)

        # Transformer
        out = self.transformer(tokens)

        # [CLS] output
        cls_out = self.layer_norm(out[:, 0, :])

        return self.head(cls_out)
