"""
PyTorch Dataset and DataLoader wrappers for tabular and graph data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAINING_CONFIG, DEVICE


class TabularDataset(Dataset):
    """Standard tabular dataset for supervised models."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AutoencoderDataset(Dataset):
    """Dataset for autoencoders — input == target."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.X[idx], self.y[idx]
        return self.X[idx], self.X[idx]


class SequentialDataset(Dataset):
    """
    Dataset for sequential models (LSTM, CNN-LSTM).
    Reshapes flat features into (seq_len, feature_dim).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Reshape (n_features,) → (seq_len, n_features // seq_len)
        if self.seq_len > 1 and len(x) >= self.seq_len:
            feat_dim = len(x) // self.seq_len
            x = x[:self.seq_len * feat_dim].reshape(self.seq_len, feat_dim)
        else:
            x = x.unsqueeze(0)  # (1, n_features)
        return x, self.y[idx]


def create_dataloaders(
    data: Dict,
    model_type: str = "supervised",
    batch_size: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.

    model_type: 'supervised' | 'autoencoder' | 'sequential'
    """
    bs = batch_size or TRAINING_CONFIG.batch_size

    if model_type == "autoencoder":
        DatasetClass = AutoencoderDataset
        train_ds = DatasetClass(data["X_train"], data.get("y_train"))
        val_ds = DatasetClass(data["X_val"], data.get("y_val"))
        test_ds = DatasetClass(data["X_test"], data.get("y_test"))
    elif model_type == "sequential":
        DatasetClass = SequentialDataset
        train_ds = DatasetClass(data["X_train"], data["y_train"])
        val_ds = DatasetClass(data["X_val"], data["y_val"])
        test_ds = DatasetClass(data["X_test"], data["y_test"])
    else:
        train_ds = TabularDataset(data["X_train"], data["y_train"])
        val_ds = TabularDataset(data["X_val"], data["y_val"])
        test_ds = TabularDataset(data["X_test"], data["y_test"])

    # Weighted sampler for class imbalance
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler and model_type != "autoencoder":
        y_train = data["y_train"]
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[y_train]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(y_train),
            replacement=True,
        )
        shuffle_train = False  # sampler handles shuffling

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=TRAINING_CONFIG.num_workers,
            pin_memory=TRAINING_CONFIG.pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=bs * 2, shuffle=False,
            num_workers=TRAINING_CONFIG.num_workers,
            pin_memory=TRAINING_CONFIG.pin_memory,
        ),
        "test": DataLoader(
            test_ds, batch_size=bs * 2, shuffle=False,
            num_workers=TRAINING_CONFIG.num_workers,
            pin_memory=TRAINING_CONFIG.pin_memory,
        ),
    }

    return loaders


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    class_counts = np.bincount(y)
    total = len(y)
    weights = total / (len(class_counts) * class_counts + 1e-8)
    return torch.FloatTensor(weights).to(DEVICE)
