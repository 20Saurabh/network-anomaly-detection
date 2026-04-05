"""
Stacking Ensemble Meta-Learner.
Combines predictions from all base models via a learned meta-classifier.
Uses cross-validated base model predictions to avoid data leakage.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from .base import BaseSklearnModel
from . import register_model


@register_model("ensemble_stacking")
class EnsembleStacking(BaseSklearnModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = "supervised"
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
        )
        self.base_model_names = []
        self._is_fitted = False

    def fit(self, X_meta: np.ndarray, y: np.ndarray,
            base_model_names: list = None):
        """
        Fit meta-learner on stacked base model predictions.

        X_meta: (n_samples, n_base_models * n_classes) — concatenated
                probability predictions from all base models.
        y: true labels
        """
        self.meta_learner.fit(X_meta, y)
        self.base_model_names = base_model_names or []
        self._is_fitted = True

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict(X_meta)

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict_proba(X_meta)

    def get_anomaly_scores(self, X_meta: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_meta)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs[:, 0]

    def count_parameters(self):
        if hasattr(self.meta_learner, 'coef_'):
            return self.meta_learner.coef_.size + self.meta_learner.intercept_.size
        return 0
