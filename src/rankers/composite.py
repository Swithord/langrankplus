from typing import Optional, Tuple
import numpy as np
from .base import BaseRanker


class CompositeDistanceRanker(BaseRanker):
    """
    Training-free ranker that averages distance features across (target, source) pairs.
    Predicts negative average distance as the relevance score - lower distance means
    higher relevance.
    """

    def __init__(self, weights: Optional[list[float]] = None):
        """
        :param weights: optional weights for each feature. If None, all features are
            weighted equally. Weights are normalized to sum to 1 internally.
        """
        self.weights = weights

    def fit(self, X: np.ndarray,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'CompositeDistanceRanker':
        if self.weights is not None and len(self.weights) != X.shape[1]:
            raise ValueError(f"Number of weights ({len(self.weights)}) does not match "
                             f"number of features ({X.shape[1]})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            avg = X.mean(axis=1)
        else:
            w = np.asarray(self.weights, dtype=float)
            w = w / w.sum()
            avg = X @ w
        return -avg
