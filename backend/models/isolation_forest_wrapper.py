"""
Isolation Forest wrapper with the same interface as deep learning models.
Traditional ML baseline — tree-based unsupervised anomaly detection.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from .base import BaseSklearnModel
from . import register_model


@register_model("isolation_forest")
class IsolationForestWrapper(BaseSklearnModel):

    def __init__(self, n_estimators: int = 200, contamination="auto",
                 random_state: int = 42, **kwargs):
        super().__init__()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model_type = "unsupervised"
        self._is_fitted = False

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        self._is_fitted = True

    def predict(self, X):
        """Returns 0 (normal) or 1 (anomaly)."""
        preds = self.model.predict(X)
        # sklearn: 1 = normal, -1 = anomaly → convert to 0/1
        return (preds == -1).astype(int)

    def get_anomaly_scores(self, X):
        """Higher score = more anomalous."""
        # sklearn: more negative = more anomalous
        return -self.model.score_samples(X)

    def count_parameters(self):
        return self.model.n_estimators * 100  # Approximate
