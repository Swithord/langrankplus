from typing import Optional, Tuple
import numpy as np
from .base import BaseRanker


class SingleFeatureRanker(BaseRanker):
    """
    Training-free baseline that ranks by a single feature.
    This is really just intended as a baseline.
    Formally, Blaschke et al. and Philippy et al. implictly uses this "ranker".
    """

    def __init__(self, feature_idx: int = 0, ascending: bool = True):
        """
        :param feature_idx: index of the feature to use
        :param ascending: if True, lower values = higher relevance (distance-like).
            If False, higher values = higher relevance (similarity-like).
        """
        self.feature_idx = feature_idx
        self.ascending = ascending

    def fit(self, X: np.ndarray,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'SingleFeatureRanker':
        if self.feature_idx >= X.shape[1]:
            raise ValueError(f"feature_idx {self.feature_idx} out of range for X "
                             f"with {X.shape[1]} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_idx]
        return -col if self.ascending else col
