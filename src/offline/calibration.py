from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm

from ..data import get_query_cols, normalize_query_features
from ..metrics import (
    relevance_from_performance,
    ndcg_at_k,
    performance_loss,
    top_k_accuracy,
)
from ..rankers.composite import CompositeDistanceRanker
from ..rankers.rrf import RRFRanker
from ..rankers.base import BaseRanker


@dataclass
class CalibrationResult:
    """
    Frozen offline calibration result.

    The resulting weights and hyperparameters can be used online without fitting
    on the new target task.
    """
    method: str
    feature_cols: list[str]
    performance_col: str
    weights: list[float]
    objective_value: float
    objective_name: str = 'mean_top1_performance_loss'
    normalizer: str = 'minmax'
    rrf_k: Optional[float] = None
    ascending: bool = True
    n_weight_samples: int = 0
    random_state: int = 42


def _evaluate_fixed_ranker(df: pd.DataFrame,
                           ranker: BaseRanker,
                           feature_cols: list[str],
                           performance_col: str,
                           target_col: str = 'task_lang',
                           source_col: str = 'transfer_lang',
                           dataset_col: Optional[str] = 'dataset',
                           ndcg_k: int = 3,
                           top_k_relevance: int = 10) -> dict[str, float]:
    """
    Evaluate a fixed-rule ranker over all queries in df.

    This is used internally for offline calibration. Metrics are returned as raw
    ratios in [0, 1].
    """
    query_cols = get_query_cols(df, target_col=target_col, dataset_col=dataset_col)

    ndcg_scores = []
    perf_losses = []
    top_1_hits = []
    top_3_hits = []

    for _, qdf in df.groupby(query_cols, sort=False):
        X = qdf[feature_cols].to_numpy(dtype=float)
        perf = qdf[performance_col].to_numpy(dtype=float)
        relevance = relevance_from_performance(perf, top_k=top_k_relevance)

        fold_ranker = clone(ranker)
        fold_ranker.fit(X)
        pred = fold_ranker.predict(X)

        ndcg_scores.append(ndcg_at_k(relevance, pred, k=ndcg_k))

        pred_best_idx = int(np.argmax(pred))
        actual_best_idx = int(np.argmax(perf))
        pred_best_perf = float(perf[pred_best_idx])
        actual_best_perf = float(perf[actual_best_idx])

        loss = performance_loss(pred_best_perf, actual_best_perf)
        if not np.isnan(loss):
            perf_losses.append(loss)

        top_1_hits.append(top_k_accuracy(relevance, pred, k=1))
        top_3_hits.append(top_k_accuracy(relevance, pred, k=3))

    return {
        'mean_ndcg': float(np.mean(ndcg_scores)) if ndcg_scores else float('nan'),
        'mean_performance_loss': float(np.mean(perf_losses)) if perf_losses else float('nan'),
        'mean_top_1_accuracy': float(np.mean(top_1_hits)) if top_1_hits else float('nan'),
        'mean_top_3_accuracy': float(np.mean(top_3_hits)) if top_3_hits else float('nan'),
    }


def _candidate_weights(n_features: int,
                       n_samples: int,
                       random_state: int) -> np.ndarray:
    """
    Generate non-negative weights that sum to one. Equal weights are always included.
    """
    rng = np.random.default_rng(random_state)
    equal = np.ones((1, n_features), dtype=float) / n_features
    if n_samples <= 0:
        return equal
    sampled = rng.dirichlet(np.ones(n_features), size=n_samples)
    return np.vstack([equal, sampled])


def fit_composite_weights(df: pd.DataFrame,
                          feature_cols: list[str],
                          performance_col: str,
                          target_col: str = 'task_lang',
                          source_col: str = 'transfer_lang',
                          dataset_col: Optional[str] = 'dataset',
                          n_samples: int = 50000,
                          random_state: int = 42,
                          normalizer: str = 'minmax',
                          verbose: bool = False,
                          desc: str = 'Composite calibration') -> CalibrationResult:
    """
    Fit task-agnostic offline weights for CompositeDistanceRanker by minimizing
    average top-1 performance loss over historical transfer matrices.

    The fitted weights are frozen and normalized by construction.
    """
    work_df = normalize_query_features(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        dataset_col=dataset_col,
        method=normalizer,
    )

    best_weights = None
    best_loss = float('inf')

    candidates = _candidate_weights(len(feature_cols), n_samples, random_state)
    iterator = tqdm(candidates, desc=desc, disable=not verbose, leave=False)

    for weights in iterator:
        ranker = CompositeDistanceRanker(weights=weights.tolist())
        metrics = _evaluate_fixed_ranker(
            work_df,
            ranker=ranker,
            feature_cols=feature_cols,
            performance_col=performance_col,
            target_col=target_col,
            source_col=source_col,
            dataset_col=dataset_col,
        )
        loss = metrics['mean_performance_loss']

        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()

        if verbose:
            iterator.set_postfix(best_loss=f'{best_loss:.4f}')

    if best_weights is None:
        raise RuntimeError("Failed to fit composite weights")

    best_weights = best_weights / best_weights.sum()

    return CalibrationResult(
        method='composite',
        feature_cols=list(feature_cols),
        performance_col=performance_col,
        weights=best_weights.tolist(),
        objective_value=float(best_loss),
        normalizer=normalizer,
        rrf_k=None,
        ascending=True,
        n_weight_samples=n_samples,
        random_state=random_state,
    )


def fit_rrf_weights(df: pd.DataFrame,
                    feature_cols: list[str],
                    performance_col: str,
                    target_col: str = 'task_lang',
                    source_col: str = 'transfer_lang',
                    dataset_col: Optional[str] = 'dataset',
                    rrf_k_grid: Sequence[float] = (1, 5, 10, 20, 40, 60, 100),
                    n_samples: int = 50000,
                    random_state: int = 42,
                    normalizer: str = 'none',
                    verbose: bool = False,
                    desc: str = 'RRF calibration') -> CalibrationResult:
    """
    Fit task-agnostic offline weights and RRF smoothing constant by minimizing
    average top-1 performance loss over historical transfer matrices.

    RRF only uses ranks, so normalization is not required. The normalizer argument
    is kept for symmetry with composite distance.
    """
    work_df = normalize_query_features(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        dataset_col=dataset_col,
        method=normalizer,
    )

    best_weights = None
    best_rrf_k = None
    best_loss = float('inf')

    weights_array = _candidate_weights(len(feature_cols), n_samples, random_state)
    total = len(rrf_k_grid) * len(weights_array)

    iterator = tqdm(total=total, desc=desc, disable=not verbose, leave=False)

    for rrf_k in rrf_k_grid:
        for weights in weights_array:
            ranker = RRFRanker(weights=weights.tolist(), rrf_k=float(rrf_k), ascending=True)
            metrics = _evaluate_fixed_ranker(
                work_df,
                ranker=ranker,
                feature_cols=feature_cols,
                performance_col=performance_col,
                target_col=target_col,
                source_col=source_col,
                dataset_col=dataset_col,
            )
            loss = metrics['mean_performance_loss']

            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_rrf_k = float(rrf_k)

            if verbose:
                iterator.set_postfix(best_loss=f'{best_loss:.4f}', best_k=best_rrf_k)

            iterator.update(1)

    iterator.close()

    if best_weights is None or best_rrf_k is None:
        raise RuntimeError("Failed to fit RRF weights")

    best_weights = best_weights / best_weights.sum()

    return CalibrationResult(
        method='rrf',
        feature_cols=list(feature_cols),
        performance_col=performance_col,
        weights=best_weights.tolist(),
        objective_value=float(best_loss),
        normalizer=normalizer,
        rrf_k=best_rrf_k,
        ascending=True,
        n_weight_samples=n_samples,
        random_state=random_state,
    )


def save_calibration(result: CalibrationResult, path: str) -> None:
    """
    Save a calibration result as JSON.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open('w') as f:
        json.dump(asdict(result), f, indent=2)


def load_calibration(path: str) -> CalibrationResult:
    """
    Load a calibration result from JSON.
    """
    with Path(path).open('r') as f:
        data = json.load(f)
    return CalibrationResult(**data)


def ranker_from_calibration(result: CalibrationResult) -> BaseRanker:
    """
    Construct the corresponding frozen ranker from a calibration result.
    """
    if result.method == 'composite':
        return CompositeDistanceRanker(weights=result.weights)
    if result.method == 'rrf':
        if result.rrf_k is None:
            raise ValueError("RRF calibration result must contain rrf_k")
        return RRFRanker(weights=result.weights,
                         rrf_k=result.rrf_k,
                         ascending=result.ascending)
    raise ValueError(f"Unknown calibration method: {result.method}")