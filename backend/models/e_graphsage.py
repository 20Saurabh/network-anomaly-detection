"""
E-GraphSAGE: Edge-centric Graph Neural Network for NIDS.
Adapted from Lo et al. (2022) — models network flows as edges in a
graph where nodes are IP addresses. Uses SAGEConv for inductive learning.
Falls back gracefully if PyTorch Geometric is not installed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseModel
from . import register_model

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@register_model("e_graphsage")
class EGraphSAGE(BaseModel):
    """
    Edge-centric GraphSAGE for network intrusion detection.
    If PyG is not installed, falls back to a MLP classifier.
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = 128, out_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.3, **kwargs):
        super().__init__(input_dim, num_classes)
        self.model_type = "supervised"
        self.use_gnn = HAS_PYG

        if self.use_gnn:
            # GNN layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(n_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, out_dim))

            self.dropout = dropout

            # Edge classifier: concat source and target node embeddings
            self.edge_classifier = nn.Sequential(
                nn.Linear(out_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # Fallback MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(out_dim, num_classes),
            )

    def forward(self, x, edge_index=None, **kwargs):
        if self.use_gnn and edge_index is not None:
            # GNN forward
            h = x
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Edge classification: concat source & target
            src, dst = edge_index
            edge_features = torch.cat([h[src], h[dst]], dim=-1)
            return self.edge_classifier(edge_features)
        else:
            # Fallback to MLP
            return self.mlp(x)

    def forward_nodes(self, x, edge_index):
        """Get node embeddings only."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
