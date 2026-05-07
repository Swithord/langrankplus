from typing import Optional, Tuple
import numpy as np
from lightgbm import LGBMRanker
import pandas as pd
from .base import BaseRanker


class LightGBMRanker(BaseRanker):
    """
    Wrapper around lightgbm.LGBMRanker conforming to the BaseRanker interface.
    Trains via lambdarank on graded relevance with NDCG@k as the eval metric.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 num_leaves: int = 16,
                 learning_rate: float = 0.1,
                 min_child_samples: int = 5,
                 eval_at: int = 3,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.eval_at = eval_at
        self.random_state = random_state
        self._ranker = None

    @staticmethod
    def _to_df(X: np.ndarray):
        return pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    @staticmethod
    def _sort_and_get_group_sizes(X: np.ndarray, y: np.ndarray,
                                  groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list[int]]:
        """
        LGBM requires contiguous groups in X (sklearn splitters don't guarantee this).
        Sort by group, then return contiguous group sizes.
        """
        groups = np.asarray(groups)
        sort_idx = np.argsort(groups, kind='stable')
        X_sorted = X[sort_idx]
        y_sorted = np.asarray(y)[sort_idx]
        groups_sorted = groups[sort_idx]
        _, counts = np.unique(groups_sorted, return_counts=True)
        # np.unique returns sorted unique values, and since data is now sorted by
        # group, count order matches data order
        return X_sorted, y_sorted, counts.tolist()

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'LightGBMRanker':
        if groups is None:
            raise ValueError("LightGBMRanker requires groups for fit")

        X_train, y_train, group_sizes = self._sort_and_get_group_sizes(X, y, groups)

        self._ranker = LGBMRanker(
            boosting_type='gbdt',
            objective='lambdarank',
            metric='ndcg',
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        fit_kwargs = {'group': group_sizes}

        if eval_set is not None and eval_groups is not None:
            X_val, y_val = eval_set
            X_val_s, y_val_s, val_group_sizes = self._sort_and_get_group_sizes(
                X_val, y_val, eval_groups)
            fit_kwargs['eval_set'] = [(self._to_df(X_val_s), y_val_s)]
            fit_kwargs['eval_group'] = [val_group_sizes]
            fit_kwargs['eval_at'] = [self.eval_at]
            fit_kwargs['eval_metric'] = 'ndcg'

        self._ranker.fit(self._to_df(X_train), y_train, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._ranker is None:
            raise RuntimeError("Ranker has not been fit yet")
        return self._ranker.predict(self._to_df(X))
