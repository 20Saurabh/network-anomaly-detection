"""
GNN-Transformer Hybrid: Novel architecture combining
graph-topology awareness (GNN) with temporal self-attention (Transformer).

Phase 1: Graph encoding via SAGEConv → per-node embeddings
Phase 2: Embeddings treated as sequence → Transformer encoder
Phase 3: [CLS] token → classification

This is the NOVEL contribution architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from . import register_model

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@register_model("gnn_transformer")
class GNNTransformerHybrid(BaseModel):

    def __init__(self, input_dim: int, num_classes: int = 2,
                 gnn_hidden: int = 128, gnn_out: int = 64,
                 n_heads: int = 4, n_transformer_layers: int = 2,
                 dropout: float = 0.2, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"
        self.use_gnn = HAS_PYG

        if self.use_gnn:
            # Phase 1: GNN
            self.gnn1 = SAGEConv(input_dim, gnn_hidden)
            self.gnn2 = SAGEConv(gnn_hidden, gnn_out)
            self.gnn_norm1 = nn.LayerNorm(gnn_hidden)
            self.gnn_norm2 = nn.LayerNorm(gnn_out)
        else:
            # Fallback: Linear projection
            self.proj = nn.Sequential(
                nn.Linear(input_dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gnn_hidden, gnn_out),
                nn.LayerNorm(gnn_out),
                nn.ReLU(inplace=True),
            )

        # Phase 2: Transformer encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, gnn_out))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_out,
            nhead=n_heads,
            dim_feedforward=gnn_out * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Phase 3: Classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(gnn_out),
            nn.Linear(gnn_out, gnn_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_out, num_classes),
        )

        self.dropout = dropout

    def forward(self, x, edge_index=None, **kwargs):
        if self.use_gnn and edge_index is not None:
            # GNN encoding
            h = self.gnn1(x, edge_index)
            h = self.gnn_norm1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            h = self.gnn2(h, edge_index)
            h = self.gnn_norm2(h)
            h = F.relu(h)
        else:
            h = self.proj(x)

        # Treat as sequence: (batch, 1, gnn_out)
        if h.dim() == 2:
            h = h.unsqueeze(1)

        # Prepend [CLS]
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)

        # Transformer
        h = self.transformer(h)

        # [CLS] output
        cls_out = h[:, 0, :]

        return self.classifier(cls_out)
