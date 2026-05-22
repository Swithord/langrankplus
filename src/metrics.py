from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata, ttest_rel
from sklearn.metrics import ndcg_score


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("At least one item is required")

    return y_true, y_score


def _top_k_indices(y_score: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, min(k, y_score.shape[0])))
    if k == 0:
        return np.array([], dtype=int)

    order = np.argsort(-y_score, kind="mergesort")
    return order[:k]


def ndcg_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)

    if np.all(y_true <= 0):
        return 0.0

    return float(
        ndcg_score(
            y_true.reshape(1, -1),
            y_score.reshape(1, -1),
            k=int(k),
        )
    )


def precision_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    idx = _top_k_indices(y_score, k)

    if idx.size == 0:
        return 0.0

    relevant = y_true > 0
    return float(np.mean(relevant[idx]))


def recall_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    relevant = y_true > 0
    n_relevant = int(np.sum(relevant))

    if n_relevant == 0:
        return 0.0

    idx = _top_k_indices(y_score, k)
    return float(np.sum(relevant[idx]) / n_relevant)


def hit_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    idx = _top_k_indices(y_score, k)

    if idx.size == 0:
        return 0.0

    relevant = y_true > 0
    return float(np.any(relevant[idx]))


def average_precision_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)

    relevant = y_true > 0
    n_relevant = int(np.sum(relevant))

    if n_relevant == 0:
        return 0.0

    idx = _top_k_indices(y_score, k)
    if idx.size == 0:
        return 0.0

    hits = relevant[idx].astype(float)
    precision_values = np.cumsum(hits) / np.arange(1, idx.size + 1)

    denominator = min(n_relevant, idx.size)
    if denominator == 0:
        return 0.0

    return float(np.sum(precision_values * hits) / denominator)


def exact_best_rank(y_true, y_score) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)

    best_relevance = float(np.max(y_true))
    best_mask = y_true == best_relevance

    pred_ranks = rankdata(-y_score, method="ordinal")

    return float(np.min(pred_ranks[best_mask]))


def reciprocal_rank(y_true, y_score) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)

    best_relevance = float(np.max(y_true))
    if best_relevance <= 0:
        return 0.0

    rank = exact_best_rank(y_true, y_score)
    if np.isnan(rank) or rank <= 0:
        return 0.0

    return float(1.0 / rank)


def exact_best_hit_at_k(y_true, y_score, k: int) -> float:
    rank = exact_best_rank(y_true, y_score)
    if np.isnan(rank):
        return 0.0
    return float(rank <= k)


def r_precision(y_true, y_score) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    n_relevant = int(np.sum(y_true > 0))

    if n_relevant == 0:
        return 0.0

    return precision_at_k(y_true, y_score, n_relevant)


def err_at_k(y_true, y_score, k: int) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    idx = _top_k_indices(y_score, k)

    if idx.size == 0:
        return 0.0

    max_relevance = float(np.max(y_true))
    if max_relevance <= 0:
        return 0.0

    relevance = y_true[idx]
    satisfaction = (np.power(2.0, relevance) - 1.0) / np.power(2.0, max_relevance)

    err = 0.0
    continuation = 1.0

    for rank, prob_stop in enumerate(satisfaction, start=1):
        err += continuation * prob_stop / rank
        continuation *= 1.0 - prob_stop

    return float(err)


def top_k_accuracy(y_true, y_score, k: int) -> float:
    return exact_best_hit_at_k(y_true, y_score, k)


def performance_loss(selected_performance: float,
                     best_performance: float) -> float:
    selected_performance = float(selected_performance)
    best_performance = float(best_performance)

    if best_performance <= 0:
        return float("nan")

    return float((best_performance - selected_performance) / best_performance)


def compute_ir_metrics(y_true,
                       y_score,
                       cutoffs: Sequence[int]) -> dict[str, float]:
    y_true, y_score = _as_arrays(y_true, y_score)

    metrics = {
        "mrr": reciprocal_rank(y_true, y_score),
        "exact_best_rank": exact_best_rank(y_true, y_score),
        "r_precision": r_precision(y_true, y_score),
        "relevant_count": float(np.sum(y_true > 0)),
    }

    for k in cutoffs:
        k = int(k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_score, k)
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_score, k)
        metrics[f"hit@{k}"] = hit_at_k(y_true, y_score, k)
        metrics[f"map@{k}"] = average_precision_at_k(y_true, y_score, k)
        metrics[f"err@{k}"] = err_at_k(y_true, y_score, k)
        metrics[f"exact_best_hit@{k}"] = exact_best_hit_at_k(y_true, y_score, k)

    return metrics


def paired_ttest(values_a, values_b) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return float("nan")

    if np.allclose(a, b):
        return 1.0

    return float(ttest_rel(a, b).pvalue)