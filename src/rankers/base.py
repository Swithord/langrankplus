from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator


class BaseRanker(BaseEstimator, ABC):
    """
    Abstract base class for transfer language rankers.
    Concrete rankers learn (or directly compute) relevance scores for (target, source)
    language pairs given some feature representation. Inherits from sklearn's
    BaseEstimator so that sklearn.base.clone() and get/set_params work out of the box,
    which the evaluator relies on for cross-validation.
    """

    @abstractmethod
    def fit(self, X: np.ndarray,
            y: np.ndarray,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'BaseRanker':
        """
        Fit the ranker.
        :param X: feature matrix of shape (n_samples, n_features)
        :param y: relevance scores of shape (n_samples,)
        :param groups: group ids (target language) of shape (n_samples,) - samples
            with the same group are ranked together as a single query
        :param eval_set: optional (X_val, y_val) tuple for early stopping
        :param eval_groups: optional group ids for the validation set
        :return: self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores.
        :param X: feature matrix of shape (n_samples, n_features)
        :return: predicted scores - higher means more relevant
        """
        pass
    