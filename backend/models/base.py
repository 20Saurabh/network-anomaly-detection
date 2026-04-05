"""
Abstract base class for all models in the benchmark.
Provides a unified interface for training, evaluation, and anomaly scoring.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseModel(ABC, nn.Module):
    """Abstract base for all PyTorch-based models."""

    def __init__(self, input_dim: int, num_classes: int = 2, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = "supervised"  # or "unsupervised"

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    def get_loss_fn(self):
        """Return the appropriate loss function."""
        if self.model_type == "unsupervised":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_anomaly_scores(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute per-sample anomaly scores.
        For supervised models: prediction probability of anomaly class.
        For unsupervised models: reconstruction error.
        Override in subclasses for custom behavior.
        """
        self.eval()
        with torch.no_grad():
            if self.model_type == "unsupervised":
                output = self.forward(x)
                if isinstance(output, tuple):
                    recon = output[0]
                else:
                    recon = output
                scores = ((x - recon) ** 2).mean(dim=-1).cpu().numpy()
            else:
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                # Score = probability of being anomalous (class 1)
                if probs.shape[-1] >= 2:
                    scores = probs[:, 1].cpu().numpy()
                else:
                    scores = probs[:, 0].cpu().numpy()
        return scores


class BaseSklearnModel(ABC):
    """Abstract base for sklearn-based models (e.g., Isolation Forest)."""

    def __init__(self, **kwargs):
        self.model = None
        self.model_type = "unsupervised"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        pass
