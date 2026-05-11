from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseRanker
from .composite import (
    _as_groups,
    _build_pairwise_differences,
    _fit_simplex_pairwise,
)


class RRFRanker(BaseRanker):
    """
    Reciprocal-rank-fusion source-language ranker.

    If trainable=False, the ranker uses fixed weights and rrf_k. If weights=None,
    it uses equal weights.

    If trainable=True, the ranker selects rrf_k from a grid and fits nonnegative
    simplex weights using a pairwise logistic ranking surrogate.
    """

    def __init__(self,
                 weights: Optional[list[float]] = None,
                 rrf_k: float = 60.0,
                 ascending: bool = True,
                 trainable: bool = False,
                 rrf_k_grid: Sequence[float] = (1, 5, 10, 20, 40, 60, 100),
                 n_steps: int = 1000,
                 learning_rate: float = 0.05,
                 max_pairs_per_query: int = 5000,
                 score_scale: float = 10.0,
                 random_state: int = 42,
                 verbose: bool = False):
        self.weights = weights
        self.rrf_k = rrf_k
        self.ascending = ascending
        self.trainable = trainable
        self.rrf_k_grid = tuple(rrf_k_grid)
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.max_pairs_per_query = max_pairs_per_query
        self.score_scale = score_scale
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {
            "weights": self.weights,
            "rrf_k": self.rrf_k,
            "ascending": self.ascending,
            "trainable": self.trainable,
            "rrf_k_grid": self.rrf_k_grid,
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "max_pairs_per_query": self.max_pairs_per_query,
            "score_scale": self.score_scale,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _fixed_weights(self, n_features: int) -> np.ndarray:
        if self.weights is None:
            return np.repeat(1.0 / n_features, n_features)

        weights = np.asarray(self.weights, dtype=float)
        if weights.shape[0] != n_features:
            raise ValueError(
                f"Expected {n_features} weights, got {weights.shape[0]}"
            )
        if np.any(weights < 0):
            raise ValueError("RRF weights must be nonnegative")
        if weights.sum() <= 0:
            raise ValueError("RRF weights must have positive sum")

        return weights / weights.sum()

    def _rrf_features(self, X: np.ndarray, groups=None, rrf_k: Optional[float] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        groups = _as_groups(groups, X.shape[0])
        k = float(self.rrf_k if rrf_k is None else rrf_k)

        Z = np.zeros_like(X, dtype=float)

        for group in np.unique(groups):
            idx = np.where(groups == group)[0]
            Xg = X[idx]

            for col_idx in range(X.shape[1]):
                ranks = pd.Series(Xg[:, col_idx]).rank(
                    method="average",
                    ascending=self.ascending,
                ).to_numpy(dtype=float)
                Z[idx, col_idx] = 1.0 / (k + ranks)

        return Z

    def fit(self,
            X,
            y=None,
            groups=None,
            eval_set=None,
            eval_groups=None):
        X = np.asarray(X, dtype=float)

        if not self.trainable:
            self.weights_ = self._fixed_weights(X.shape[1])
            self.rrf_k_ = float(self.rrf_k)
            self.surrogate_loss_ = None
            return self

        if y is None:
            raise ValueError("y is required when trainable=True")

        y = np.asarray(y, dtype=float)

        best_loss = float("inf")
        best_weights = None
        best_k = None

        iterator = tqdm(
            list(self.rrf_k_grid),
            desc="RRF pairwise fitting",
            disable=not self.verbose,
            leave=False,
        )

        for k in iterator:
            Z = self._rrf_features(X, groups=groups, rrf_k=float(k))

            diffs = _build_pairwise_differences(
                Z,
                y,
                groups,
                mode="score",
                max_pairs_per_query=self.max_pairs_per_query,
                random_state=self.random_state,
            )

            weights, surrogate_loss = _fit_simplex_pairwise(
                diffs,
                n_steps=self.n_steps,
                learning_rate=self.learning_rate,
                score_scale=self.score_scale,
                random_state=self.random_state,
                verbose=False,
                desc=f"RRF k={k}",
            )

            if surrogate_loss < best_loss:
                best_loss = surrogate_loss
                best_weights = weights
                best_k = float(k)

            if self.verbose:
                iterator.set_postfix(
                    k=float(k),
                    loss=f"{surrogate_loss:.4f}",
                    best_loss=f"{best_loss:.4f}",
                )

        if best_weights is None or best_k is None:
            raise RuntimeError("RRF pairwise fitting failed")

        self.weights_ = best_weights / best_weights.sum()
        self.rrf_k_ = best_k
        self.surrogate_loss_ = float(best_loss)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "weights_"):
            self.weights_ = self._fixed_weights(X.shape[1])

        if not hasattr(self, "rrf_k_"):
            self.rrf_k_ = float(self.rrf_k)

        Z = self._rrf_features(X, groups=None, rrf_k=self.rrf_k_)
        return Z @ self.weights_

    def save(self, path: str | Path) -> None:
        if not hasattr(self, "weights_"):
            if self.weights is None:
                raise ValueError("Cannot save an unfitted ranker without weights")
            weights = np.asarray(self.weights, dtype=float)
            weights = weights / weights.sum()
        else:
            weights = self.weights_

        rrf_k = float(getattr(self, "rrf_k_", self.rrf_k))

        payload = {
            "method": "rrf_pairwise",
            "weights": weights.tolist(),
            "rrf_k": rrf_k,
            "ascending": self.ascending,
            "trainable": False,
            "rrf_k_grid": list(self.rrf_k_grid),
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "max_pairs_per_query": self.max_pairs_per_query,
            "score_scale": self.score_scale,
            "random_state": self.random_state,
            "surrogate_loss": getattr(self, "surrogate_loss_", None),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "RRFRanker":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        ranker = cls(
            weights=payload["weights"],
            rrf_k=float(payload["rrf_k"]),
            ascending=payload.get("ascending", True),
            trainable=False,
            rrf_k_grid=tuple(payload.get("rrf_k_grid", [1, 5, 10, 20, 40, 60, 100])),
            n_steps=payload.get("n_steps", 1000),
            learning_rate=payload.get("learning_rate", 0.05),
            max_pairs_per_query=payload.get("max_pairs_per_query", 5000),
            score_scale=payload.get("score_scale", 10.0),
            random_state=payload.get("random_state", 42),
            verbose=False,
        )
        ranker.weights_ = np.asarray(payload["weights"], dtype=float)
        ranker.weights_ = ranker.weights_ / ranker.weights_.sum()
        ranker.rrf_k_ = float(payload["rrf_k"])
        ranker.surrogate_loss_ = payload.get("surrogate_loss")
        return ranker