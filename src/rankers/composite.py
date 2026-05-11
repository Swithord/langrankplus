from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
from tqdm import tqdm

from .base import BaseRanker


def _softmax(theta: np.ndarray) -> np.ndarray:
    shifted = theta - np.max(theta)
    exp_theta = np.exp(shifted)
    return exp_theta / exp_theta.sum()


def _as_groups(groups, n_rows: int) -> np.ndarray:
    if groups is None:
        return np.zeros(n_rows, dtype=int)
    return np.asarray(groups)


def _build_pairwise_differences(X: np.ndarray,
                                y: np.ndarray,
                                groups,
                                *,
                                mode: str,
                                max_pairs_per_query: int,
                                random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    groups = _as_groups(groups, X.shape[0])
    all_diffs = []

    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        Xg = X[idx]
        yg = y[idx]

        better_idx, worse_idx = np.where(yg[:, None] > yg[None, :])

        if better_idx.size == 0:
            continue

        if max_pairs_per_query is not None and better_idx.size > max_pairs_per_query:
            keep = rng.choice(
                better_idx.size,
                size=max_pairs_per_query,
                replace=False,
            )
            better_idx = better_idx[keep]
            worse_idx = worse_idx[keep]

        if mode == "distance":
            diffs = Xg[worse_idx] - Xg[better_idx]
        elif mode == "score":
            diffs = Xg[better_idx] - Xg[worse_idx]
        else:
            raise ValueError("mode must be 'distance' or 'score'")

        all_diffs.append(diffs)

    if not all_diffs:
        raise ValueError("No informative within-query source pairs were found")

    return np.vstack(all_diffs)


def _pairwise_loss_and_gradient(theta: np.ndarray,
                                diffs: np.ndarray,
                                score_scale: float) -> tuple[float, np.ndarray, np.ndarray]:
    weights = _softmax(theta)
    margins = score_scale * (diffs @ weights)

    loss = float(np.logaddexp(0.0, -margins).mean())

    sigmoid_neg_margin = np.exp(-np.logaddexp(0.0, margins))
    grad_w = -score_scale * (sigmoid_neg_margin[:, None] * diffs).mean(axis=0)

    grad_theta = weights * (grad_w - float(np.dot(grad_w, weights)))

    return loss, grad_theta, weights


def _fit_simplex_pairwise(diffs: np.ndarray,
                          *,
                          n_steps: int,
                          learning_rate: float,
                          score_scale: float,
                          random_state: int,
                          verbose: bool,
                          desc: str) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    n_features = diffs.shape[1]

    if n_steps <= 0:
        weights = np.repeat(1.0 / n_features, n_features)
        loss, _, _ = _pairwise_loss_and_gradient(
            np.zeros(n_features),
            diffs,
            score_scale=score_scale,
        )
        return weights, loss

    theta = rng.normal(loc=0.0, scale=0.01, size=n_features)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    best_loss = float("inf")
    best_weights = _softmax(theta)

    iterator = tqdm(
        range(1, n_steps + 1),
        desc=desc,
        disable=not verbose,
        leave=False,
    )

    for step in iterator:
        loss, grad, weights = _pairwise_loss_and_gradient(
            theta,
            diffs,
            score_scale=score_scale,
        )

        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)

        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)

        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if verbose:
            iterator.set_postfix(loss=f"{loss:.4f}", best_loss=f"{best_loss:.4f}")

    return best_weights / best_weights.sum(), float(best_loss)


class CompositeDistanceRanker(BaseRanker):
    """
    Composite source-language ranker.

    If trainable=False, the ranker uses fixed weights. If weights=None, it uses
    equal weights.

    If trainable=True, the ranker fits nonnegative simplex weights using a
    pairwise logistic ranking surrogate.
    """

    def __init__(self,
                 weights: Optional[list[float]] = None,
                 trainable: bool = False,
                 n_steps: int = 1000,
                 learning_rate: float = 0.05,
                 max_pairs_per_query: int = 5000,
                 score_scale: float = 10.0,
                 random_state: int = 42,
                 verbose: bool = False):
        self.weights = weights
        self.trainable = trainable
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.max_pairs_per_query = max_pairs_per_query
        self.score_scale = score_scale
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {
            "weights": self.weights,
            "trainable": self.trainable,
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
            raise ValueError("Composite weights must be nonnegative")
        if weights.sum() <= 0:
            raise ValueError("Composite weights must have positive sum")

        return weights / weights.sum()

    def fit(self,
            X,
            y=None,
            groups=None,
            eval_set=None,
            eval_groups=None):
        X = np.asarray(X, dtype=float)

        if not self.trainable:
            self.weights_ = self._fixed_weights(X.shape[1])
            self.surrogate_loss_ = None
            return self

        if y is None:
            raise ValueError("y is required when trainable=True")

        y = np.asarray(y, dtype=float)

        diffs = _build_pairwise_differences(
            X,
            y,
            groups,
            mode="distance",
            max_pairs_per_query=self.max_pairs_per_query,
            random_state=self.random_state,
        )

        weights, surrogate_loss = _fit_simplex_pairwise(
            diffs,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            score_scale=self.score_scale,
            random_state=self.random_state,
            verbose=self.verbose,
            desc="Composite pairwise fitting",
        )

        self.weights_ = weights
        self.surrogate_loss_ = surrogate_loss
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "weights_"):
            self.weights_ = self._fixed_weights(X.shape[1])

        return -(X @ self.weights_)

    def save(self, path: str | Path) -> None:
        if not hasattr(self, "weights_"):
            if self.weights is None:
                raise ValueError("Cannot save an unfitted ranker without weights")
            weights = np.asarray(self.weights, dtype=float)
            weights = weights / weights.sum()
        else:
            weights = self.weights_

        payload = {
            "method": "composite_pairwise",
            "weights": weights.tolist(),
            "trainable": False,
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
    def load(cls, path: str | Path) -> "CompositeDistanceRanker":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        ranker = cls(
            weights=payload["weights"],
            trainable=False,
            n_steps=payload.get("n_steps", 1000),
            learning_rate=payload.get("learning_rate", 0.05),
            max_pairs_per_query=payload.get("max_pairs_per_query", 5000),
            score_scale=payload.get("score_scale", 10.0),
            random_state=payload.get("random_state", 42),
            verbose=False,
        )
        ranker.weights_ = np.asarray(payload["weights"], dtype=float)
        ranker.weights_ = ranker.weights_ / ranker.weights_.sum()
        ranker.surrogate_loss_ = payload.get("surrogate_loss")
        return ranker