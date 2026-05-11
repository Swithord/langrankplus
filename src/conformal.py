from dataclasses import dataclass
from typing import Any
import math
import numpy as np
import pandas as pd


@dataclass
class ConformalPrediction:
    """
    Conformal source-set prediction for a single held-out query.
    """
    threshold: float
    set_mask: np.ndarray
    set_sources: list[Any]
    set_size: int
    contains_best_source: float
    singleton: float
    empty: float
    best_in_set_performance: float
    best_in_set_performance_loss: float


def conformal_quantile(nonconformity_scores: list[float],
                       alpha: float) -> float:
    """
    Finite-sample split-conformal quantile.

    For n calibration queries, use the ceil((n + 1)(1 - alpha))-th smallest
    nonconformity score. If this index exceeds n, return infinity, which gives
    the trivial full set and preserves the conservative finite-sample rule.

    :param nonconformity_scores: calibration nonconformity scores
    :param alpha: target miscoverage level
    :return: conformal threshold
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    scores = np.asarray(nonconformity_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one calibration score is required")

    scores = np.sort(scores)
    n = scores.size
    k = int(math.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return float('inf')

    return float(scores[k - 1])


def best_source_gap_nonconformity(scores: np.ndarray,
                                  performances: np.ndarray) -> float:
    """
    Nonconformity score for a query.

    The score is the gap between the ranker's top score and the score assigned
    to the oracle-best source language.

    A small value means the oracle-best source was near the top of the ranker's
    scoring rule.

    :param scores: predicted source scores; higher is better
    :param performances: observed transfer performances; higher is better
    :return: nonconformity score
    """
    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    best_idx = int(np.argmax(performances))
    return float(np.max(scores) - scores[best_idx])


def conformal_source_set(scores: np.ndarray,
                         threshold: float) -> np.ndarray:
    """
    Construct a conformal source set from predicted scores and a threshold.

    The set contains all sources whose score is within threshold of the top score.

    :param scores: predicted source scores; higher is better
    :param threshold: conformal threshold
    :return: boolean mask for selected sources
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    if np.isinf(threshold):
        return np.ones(scores.shape[0], dtype=bool)

    return scores >= np.max(scores) - threshold


def evaluate_conformal_source_set(scores: np.ndarray,
                                  performances: np.ndarray,
                                  sources: np.ndarray,
                                  threshold: float) -> ConformalPrediction:
    """
    Evaluate a conformal source set for one held-out query.

    :param scores: predicted source scores; higher is better
    :param performances: observed transfer performances; higher is better
    :param sources: source-language identifiers
    :param threshold: conformal threshold
    :return: ConformalPrediction
    """
    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)
    sources = np.asarray(sources)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if sources.shape[0] != scores.shape[0]:
        raise ValueError("sources and scores must have the same length")

    set_mask = conformal_source_set(scores, threshold)
    set_size = int(set_mask.sum())

    best_idx = int(np.argmax(performances))
    best_performance = float(performances[best_idx])
    contains_best = float(bool(set_mask[best_idx]))

    if set_size == 0:
        best_in_set_performance = float('nan')
        best_in_set_loss = float('nan')
    else:
        best_in_set_performance = float(np.max(performances[set_mask]))
        if best_performance == 0:
            best_in_set_loss = float('nan')
        else:
            best_in_set_loss = float(
                (best_performance - best_in_set_performance) / best_performance
            )

    set_sources = sources[set_mask].tolist()

    return ConformalPrediction(
        threshold=float(threshold),
        set_mask=set_mask,
        set_sources=set_sources,
        set_size=set_size,
        contains_best_source=contains_best,
        singleton=float(set_size == 1),
        empty=float(set_size == 0),
        best_in_set_performance=best_in_set_performance,
        best_in_set_performance_loss=best_in_set_loss,
    )


def serialize_sources(sources: list[Any]) -> str:
    """
    Serialize source identifiers for CSV output.
    """
    return ",".join(str(source) for source in sources)


def calibration_scores_from_queries(df: pd.DataFrame,
                                    ranker,
                                    feature_cols: list[str],
                                    performance_col: str,
                                    query_col: str = '_query_id') -> list[float]:
    """
    Compute one conformal nonconformity score per calibration query.

    :param df: calibration dataframe containing query_col
    :param ranker: fitted ranker
    :param feature_cols: feature columns
    :param performance_col: performance column
    :param query_col: query id column
    :return: list of nonconformity scores
    """
    scores = []

    for _, qdf in df.groupby(query_col, sort=False):
        X = qdf[feature_cols].to_numpy(dtype=float)
        pred = ranker.predict(X)
        perf = qdf[performance_col].to_numpy(dtype=float)
        scores.append(best_source_gap_nonconformity(pred, perf))

    return scores