from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import add_query_id, get_query_cols, normalize_query_features
from ..metrics import performance_loss
from ..rankers.composite import CompositeDistanceRanker
from ..rankers.rrf import RRFRanker


@dataclass
class CalibrationResult:
    method: str
    feature_cols: list[str]
    performance_col: str
    weights: list[float]
    objective_value: float
    normalizer: str
    rrf_k: Optional[float] = None
    ascending: bool = True
    n_steps: int = 1000
    learning_rate: float = 0.05
    max_pairs_per_query: int = 5000
    score_scale: float = 10.0
    random_state: int = 42
    surrogate_loss: Optional[float] = None


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True)


def load_calibration(path: str | Path) -> CalibrationResult:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationResult(**data)


def ranker_from_calibration(result: CalibrationResult):
    if result.method.startswith("composite"):
        return CompositeDistanceRanker(weights=result.weights)

    if result.method.startswith("rrf"):
        if result.rrf_k is None:
            raise ValueError("RRF calibration requires rrf_k")
        return RRFRanker(
            weights=result.weights,
            rrf_k=float(result.rrf_k),
            ascending=result.ascending,
        )

    raise ValueError(f"Unknown calibration method: {result.method}")


def _resolve_n_steps(n_steps: Optional[int], n_samples: Optional[int]) -> int:
    if n_steps is not None:
        return int(n_steps)
    if n_samples is not None:
        return int(n_samples)
    return 1000


def _prepare_calibration_frame(df: pd.DataFrame,
                               feature_cols: list[str],
                               target_col: str,
                               dataset_col: Optional[str],
                               normalizer: str) -> pd.DataFrame:
    work_df = normalize_query_features(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        dataset_col=dataset_col,
        method=normalizer,
    )

    work_df = add_query_id(
        work_df,
        target_col=target_col,
        dataset_col=dataset_col,
        query_id_col="_query_id",
    )

    return work_df


def _build_pairwise_differences(df: pd.DataFrame,
                                feature_cols: list[str],
                                performance_col: str,
                                *,
                                mode: str,
                                max_pairs_per_query: int,
                                random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    all_diffs = []

    for _, qdf in df.groupby("_query_id", sort=False):
        X = qdf[feature_cols].to_numpy(dtype=float)
        y = qdf[performance_col].to_numpy(dtype=float)

        better_idx, worse_idx = np.where(y[:, None] > y[None, :])

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
            diffs = X[worse_idx] - X[better_idx]
        elif mode == "score":
            diffs = X[better_idx] - X[worse_idx]
        else:
            raise ValueError("mode must be 'distance' or 'score'")

        all_diffs.append(diffs)

    if not all_diffs:
        raise ValueError("No informative within-query pairs were found")

    return np.vstack(all_diffs)


def _softmax(theta: np.ndarray) -> np.ndarray:
    shifted = theta - np.max(theta)
    exp_theta = np.exp(shifted)
    return exp_theta / exp_theta.sum()


def _pairwise_loss_and_gradient(theta: np.ndarray,
                                diffs: np.ndarray,
                                score_scale: float) -> tuple[float, np.ndarray, np.ndarray]:
    weights = _softmax(theta)
    margins = score_scale * (diffs @ weights)

    loss = float(np.logaddexp(0.0, -margins).mean())

    sigmoid_neg_margin = 1.0 / (1.0 + np.exp(margins))
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


def _mean_source_selection_loss(df: pd.DataFrame,
                                ranker,
                                feature_cols: list[str],
                                performance_col: str) -> float:
    losses = []

    for _, qdf in df.groupby("_query_id", sort=False):
        X = qdf[feature_cols].to_numpy(dtype=float)
        scores = ranker.predict(X)
        performances = qdf[performance_col].to_numpy(dtype=float)

        pred_best_idx = int(np.argmax(scores))
        actual_best_idx = int(np.argmax(performances))

        pred_best_perf = float(performances[pred_best_idx])
        actual_best_perf = float(performances[actual_best_idx])

        loss = performance_loss(pred_best_perf, actual_best_perf)
        if not np.isnan(loss):
            losses.append(loss)

    if not losses:
        return float("nan")

    return float(np.mean(losses))


def fit_composite_weights(df: pd.DataFrame,
                          feature_cols: list[str],
                          performance_col: str,
                          target_col: str = "task_lang",
                          source_col: str = "transfer_lang",
                          dataset_col: Optional[str] = "dataset",
                          n_steps: Optional[int] = None,
                          n_samples: Optional[int] = None,
                          learning_rate: float = 0.05,
                          max_pairs_per_query: int = 5000,
                          score_scale: float = 10.0,
                          random_state: int = 42,
                          normalizer: str = "minmax",
                          verbose: bool = False,
                          desc: str = "Composite pairwise fitting") -> CalibrationResult:
    steps = _resolve_n_steps(n_steps, n_samples)

    work_df = _prepare_calibration_frame(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        dataset_col=dataset_col,
        normalizer=normalizer,
    )

    diffs = _build_pairwise_differences(
        work_df,
        feature_cols=feature_cols,
        performance_col=performance_col,
        mode="distance",
        max_pairs_per_query=max_pairs_per_query,
        random_state=random_state,
    )

    weights, surrogate_loss = _fit_simplex_pairwise(
        diffs,
        n_steps=steps,
        learning_rate=learning_rate,
        score_scale=score_scale,
        random_state=random_state,
        verbose=verbose,
        desc=desc,
    )

    ranker = CompositeDistanceRanker(weights=weights.tolist())
    exact_loss = _mean_source_selection_loss(
        work_df,
        ranker=ranker,
        feature_cols=feature_cols,
        performance_col=performance_col,
    )

    return CalibrationResult(
        method="composite_pairwise",
        feature_cols=list(feature_cols),
        performance_col=performance_col,
        weights=weights.tolist(),
        objective_value=exact_loss,
        normalizer=normalizer,
        rrf_k=None,
        ascending=True,
        n_steps=steps,
        learning_rate=learning_rate,
        max_pairs_per_query=max_pairs_per_query,
        score_scale=score_scale,
        random_state=random_state,
        surrogate_loss=surrogate_loss,
    )


def _add_rrf_features(df: pd.DataFrame,
                      feature_cols: list[str],
                      target_col: str,
                      dataset_col: Optional[str],
                      rrf_k: float,
                      ascending: bool = True) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    query_cols = get_query_cols(
        out,
        target_col=target_col,
        dataset_col=dataset_col,
    )

    rrf_cols = []

    for col in feature_cols:
        new_col = f"__rrf_{col}"
        ranks = out.groupby(query_cols)[col].rank(
            method="average",
            ascending=ascending,
        )
        out[new_col] = 1.0 / (float(rrf_k) + ranks)
        rrf_cols.append(new_col)

    return out, rrf_cols


def fit_rrf_weights(df: pd.DataFrame,
                    feature_cols: list[str],
                    performance_col: str,
                    target_col: str = "task_lang",
                    source_col: str = "transfer_lang",
                    dataset_col: Optional[str] = "dataset",
                    rrf_k_grid: Sequence[float] = (1, 5, 10, 20, 40, 60, 100),
                    n_steps: Optional[int] = None,
                    n_samples: Optional[int] = None,
                    learning_rate: float = 0.05,
                    max_pairs_per_query: int = 5000,
                    score_scale: float = 10.0,
                    random_state: int = 42,
                    normalizer: str = "none",
                    verbose: bool = False,
                    desc: str = "RRF pairwise fitting") -> CalibrationResult:
    steps = _resolve_n_steps(n_steps, n_samples)

    work_df = _prepare_calibration_frame(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        dataset_col=dataset_col,
        normalizer=normalizer,
    )

    best_result = None
    best_surrogate_loss = float("inf")

    iterator = tqdm(
        list(rrf_k_grid),
        desc=desc,
        disable=not verbose,
        leave=False,
    )

    for rrf_k in iterator:
        rrf_df, rrf_cols = _add_rrf_features(
            work_df,
            feature_cols=feature_cols,
            target_col=target_col,
            dataset_col=dataset_col,
            rrf_k=float(rrf_k),
            ascending=True,
        )

        diffs = _build_pairwise_differences(
            rrf_df,
            feature_cols=rrf_cols,
            performance_col=performance_col,
            mode="score",
            max_pairs_per_query=max_pairs_per_query,
            random_state=random_state,
        )

        weights, surrogate_loss = _fit_simplex_pairwise(
            diffs,
            n_steps=steps,
            learning_rate=learning_rate,
            score_scale=score_scale,
            random_state=random_state,
            verbose=False,
            desc=f"RRF k={rrf_k}",
        )

        ranker = RRFRanker(
            weights=weights.tolist(),
            rrf_k=float(rrf_k),
            ascending=True,
        )
        exact_loss = _mean_source_selection_loss(
            work_df,
            ranker=ranker,
            feature_cols=feature_cols,
            performance_col=performance_col,
        )

        if verbose:
            iterator.set_postfix(
                best_loss=f"{best_surrogate_loss:.4f}",
                k=float(rrf_k),
                loss=f"{surrogate_loss:.4f}",
            )

        if surrogate_loss < best_surrogate_loss:
            best_surrogate_loss = surrogate_loss
            best_result = CalibrationResult(
                method="rrf_pairwise",
                feature_cols=list(feature_cols),
                performance_col=performance_col,
                weights=weights.tolist(),
                objective_value=exact_loss,
                normalizer=normalizer,
                rrf_k=float(rrf_k),
                ascending=True,
                n_steps=steps,
                learning_rate=learning_rate,
                max_pairs_per_query=max_pairs_per_query,
                score_scale=score_scale,
                random_state=random_state,
                surrogate_loss=float(surrogate_loss),
            )

    if best_result is None:
        raise RuntimeError("RRF pairwise fitting failed")

    return best_result