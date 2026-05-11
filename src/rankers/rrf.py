from typing import Optional, Tuple
import numpy as np
from scipy.stats import rankdata
from .base import BaseRanker


class RRFRanker(BaseRanker):
    """
    Training-free ranker that combines feature-wise rankings using Reciprocal Rank
    Fusion (RRF). For each feature, (target, source) pairs are ranked by that feature;
    the final relevance score is the weighted sum of reciprocal ranks.

    By default, features are treated as distance-like: lower values mean better ranks.

    Refer to: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf 
    """

    def __init__(self,
                 weights: Optional[list[float]] = None,
                 rrf_k: float = 60.0,
                 ascending: bool = True):
        """
        :param weights: optional weights for each feature. If None, all features are
            weighted equally. Weights are normalized to sum to 1 internally.
        :param rrf_k: RRF smoothing constant. Larger values make rank differences
            less sharp.
        :param ascending: if True, lower feature values get better ranks. If False,
            higher feature values get better ranks.
        """
        self.weights = weights
        self.rrf_k = rrf_k
        self.ascending = ascending

    def fit(self, X: np.ndarray,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'RRFRanker':
        if self.weights is not None and len(self.weights) != X.shape[1]:
            raise ValueError(f"Number of weights ({len(self.weights)}) does not match "
                             f"number of features ({X.shape[1]})")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            w = np.ones(X.shape[1], dtype=float) / X.shape[1]
        else:
            w = np.asarray(self.weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("RRF weights must be non-negative")
            if w.sum() <= 0:
                raise ValueError("RRF weights must have positive sum")
            w = w / w.sum()

        scores = np.zeros(X.shape[0], dtype=float)

        for j in range(X.shape[1]):
            col = X[:, j]
            values = col if self.ascending else -col
            ranks = rankdata(values, method='min')
            scores += w[j] / (self.rrf_k + ranks)

        return scores