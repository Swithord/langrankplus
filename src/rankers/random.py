from typing import Optional, Tuple
import numpy as np
from .base import BaseRanker


class RandomRanker(BaseRanker):
    """
    Training-free random baseline. Predicts independent random relevance scores,
    where higher scores mean more relevant.
    """

    def __init__(self, random_state: int = 42):
        """
        :param random_state: seed for the random baseline
        """
        self.random_state = random_state

    def fit(self, X: np.ndarray,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'RandomRanker':
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.random(X.shape[0])