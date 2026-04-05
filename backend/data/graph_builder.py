"""
Graph construction for GNN models.
Converts tabular flow data into PyTorch Geometric Data objects.
IP addresses → nodes, flows → edges with features.
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_graph_from_tabular(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    max_nodes: int = 5000,
) -> "torch_geometric.data.Data":
    """
    Build a synthetic graph structure from tabular data when
    IP-level information is not available.

    Strategy: k-NN graph in feature space.
    Each sample becomes a node; edges connect nearest neighbors.
    This is a well-established approach in GNN-for-tabular literature.
    """
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import knn_graph
    except ImportError:
        raise ImportError(
            "PyTorch Geometric required. Install: pip install torch-geometric"
        )

    n_samples = min(len(X), max_nodes)
    if n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y

    x = torch.FloatTensor(X_sub)
    labels = torch.LongTensor(y_sub)

    # k-NN graph construction
    k = min(10, n_samples - 1)
    edge_index = knn_graph(x, k=k, loop=False)

    data = Data(x=x, edge_index=edge_index, y=labels)
    data.num_classes = len(np.unique(y_sub))

    return data


def build_ip_graph(
    df,
    src_col: str = "src_ip",
    dst_col: str = "dst_ip",
    feature_cols: list = None,
    label_col: str = "label",
) -> "torch_geometric.data.Data":
    """
    Build a proper network graph from flow data with IP addresses.
    Nodes = unique IPs, Edges = flows, Edge features = flow statistics.

    Use this when the dataset has IP address columns.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("PyTorch Geometric required.")

    # Map IPs to integer indices
    all_ips = set(df[src_col].unique()) | set(df[dst_col].unique())
    ip_to_idx = {ip: i for i, ip in enumerate(sorted(all_ips))}
    num_nodes = len(ip_to_idx)

    # Build edge index
    src_indices = df[src_col].map(ip_to_idx).values
    dst_indices = df[dst_col].map(ip_to_idx).values
    edge_index = torch.LongTensor(np.stack([src_indices, dst_indices]))

    # Edge features
    if feature_cols:
        edge_attr = torch.FloatTensor(df[feature_cols].values.astype(np.float32))
    else:
        edge_attr = None

    # Edge labels
    if label_col in df.columns:
        edge_labels = torch.LongTensor(df[label_col].values)
    else:
        edge_labels = None

    # Node features: degree-based (simple but effective)
    node_features = torch.zeros(num_nodes, 4)  # in-degree, out-degree, total-in-bytes, total-out-bytes
    for i in range(len(df)):
        src, dst = src_indices[i], dst_indices[i]
        node_features[src, 1] += 1  # out-degree
        node_features[dst, 0] += 1  # in-degree

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=edge_labels,
    )
    data.num_nodes = num_nodes

    return data
