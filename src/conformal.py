from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import math

import numpy as np
import pandas as pd


VALID_CONFORMAL_MODES = {
    "gap_best",
    "rank_best",
    "rank_near_best",
}

VALID_NEAR_BEST_RULES = {
    "relative",
    "std",
}


@dataclass
class ConformalPrediction:
    threshold: float
    mode: str
    effective_k: Optional[int]
    set_mask: np.ndarray
    set_sources: list[Any]
    set_size: int
    set_size_fraction: float
    capped: float
    contains_best_source: float
    contains_near_best_source: float
    singleton: float
    empty: float
    best_in_set_performance: float
    best_in_set_performance_loss: float


@dataclass
class TopKOperationalResult:
    k: int
    rule: str
    epsilon: float
    std_multiplier: float
    set_size: int
    contains_near_best_source: float
    best_in_set_performance: float
    best_in_set_performance_loss: float


def validate_conformal_mode(mode: str) -> None:
    if mode not in VALID_CONFORMAL_MODES:
        raise ValueError(
            f"Unknown conformal mode {mode!r}. "
            f"Expected one of {sorted(VALID_CONFORMAL_MODES)}."
        )


def validate_near_best_rule(rule: str) -> None:
    if rule not in VALID_NEAR_BEST_RULES:
        raise ValueError(
            f"Unknown near-best rule {rule!r}. "
            f"Expected one of {sorted(VALID_NEAR_BEST_RULES)}."
        )


def metric_float_tag(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p").replace("-", "m")


def conformal_quantile(nonconformity_scores: list[float],
                       alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    scores = np.asarray(nonconformity_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one calibration score is required")

    scores = np.sort(scores)
    n = scores.size
    k = int(math.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return float("inf")

    return float(scores[k - 1])


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=int)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    return ranks


def near_best_mask(performances: np.ndarray,
                   *,
                   rule: str = "relative",
                   epsilon: float = 0.05,
                   std_multiplier: float = 1.0) -> np.ndarray:
    validate_near_best_rule(rule)

    performances = np.asarray(performances, dtype=float)
    if performances.size == 0:
        raise ValueError("performances must be nonempty")

    best = float(np.max(performances))

    if rule == "relative":
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        if best <= 0:
            return performances >= best

        losses = (best - performances) / best
        return losses <= epsilon

    if rule == "std":
        if std_multiplier < 0:
            raise ValueError("std_multiplier must be nonnegative")

        sd = float(np.std(performances, ddof=0))
        threshold = best - std_multiplier * sd
        return performances >= threshold

    raise ValueError(f"Unknown near-best rule: {rule}")


def best_source_gap_nonconformity(scores: np.ndarray,
                                  performances: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    best_idx = int(np.argmax(performances))
    return float(np.max(scores) - scores[best_idx])


def best_source_rank_nonconformity(scores: np.ndarray,
                                   performances: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    ranks = descending_ranks(scores)
    best_idx = int(np.argmax(performances))
    return float(ranks[best_idx])


def near_best_rank_nonconformity(scores: np.ndarray,
                                 performances: np.ndarray,
                                 *,
                                 near_best_rule: str = "relative",
                                 near_best_epsilon: float = 0.05,
                                 near_best_std_multiplier: float = 1.0) -> float:
    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    ranks = descending_ranks(scores)
    good = near_best_mask(
        performances,
        rule=near_best_rule,
        epsilon=near_best_epsilon,
        std_multiplier=near_best_std_multiplier,
    )

    if not np.any(good):
        raise ValueError("No near-best source found")

    return float(np.min(ranks[good]))


def nonconformity_score(scores: np.ndarray,
                        performances: np.ndarray,
                        *,
                        mode: str,
                        near_best_rule: str = "relative",
                        near_best_epsilon: float = 0.05,
                        near_best_std_multiplier: float = 1.0) -> float:
    validate_conformal_mode(mode)

    if mode == "gap_best":
        return best_source_gap_nonconformity(scores, performances)

    if mode == "rank_best":
        return best_source_rank_nonconformity(scores, performances)

    if mode == "rank_near_best":
        return near_best_rank_nonconformity(
            scores,
            performances,
            near_best_rule=near_best_rule,
            near_best_epsilon=near_best_epsilon,
            near_best_std_multiplier=near_best_std_multiplier,
        )

    raise ValueError(f"Unknown conformal mode: {mode}")


def calibration_scores_from_queries(df: pd.DataFrame,
                                    ranker,
                                    feature_cols: list[str],
                                    performance_col: str,
                                    query_col: str = "_query_id",
                                    mode: str = "rank_near_best",
                                    near_best_rule: str = "relative",
                                    near_best_epsilon: float = 0.05,
                                    near_best_std_multiplier: float = 1.0) -> list[float]:
    validate_conformal_mode(mode)
    validate_near_best_rule(near_best_rule)

    scores = []

    for _, qdf in df.groupby(query_col, sort=False):
        X = qdf[feature_cols].to_numpy(dtype=float)
        pred = ranker.predict(X)
        perf = qdf[performance_col].to_numpy(dtype=float)

        scores.append(
            nonconformity_score(
                pred,
                perf,
                mode=mode,
                near_best_rule=near_best_rule,
                near_best_epsilon=near_best_epsilon,
                near_best_std_multiplier=near_best_std_multiplier,
            )
        )

    return scores


def _top_k_mask(scores: np.ndarray,
                k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    k = int(max(0, min(k, scores.shape[0])))

    mask = np.zeros(scores.shape[0], dtype=bool)
    if k == 0:
        return mask

    order = np.argsort(-scores, kind="mergesort")
    mask[order[:k]] = True
    return mask


def _gap_mask(scores: np.ndarray,
              threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if np.isinf(threshold):
        return np.ones(scores.shape[0], dtype=bool)

    return scores >= np.max(scores) - threshold


def conformal_source_set(scores: np.ndarray,
                         threshold: float,
                         *,
                         mode: str,
                         max_set_size: Optional[int] = None) -> tuple[np.ndarray, Optional[int], float]:
    validate_conformal_mode(mode)

    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    capped = 0.0

    if mode == "gap_best":
        mask = _gap_mask(scores, threshold)
        effective_k = int(mask.sum())

        if max_set_size is not None and mask.sum() > max_set_size:
            mask = _top_k_mask(scores, max_set_size)
            effective_k = int(max_set_size)
            capped = 1.0

        return mask, effective_k, capped

    if np.isinf(threshold):
        k = scores.shape[0]
    else:
        k = int(math.ceil(threshold))

    k = max(1, min(k, scores.shape[0]))

    if max_set_size is not None and k > max_set_size:
        k = int(max_set_size)
        capped = 1.0

    mask = _top_k_mask(scores, k)
    return mask, k, capped


def evaluate_conformal_source_set(scores: np.ndarray,
                                  performances: np.ndarray,
                                  sources: np.ndarray,
                                  threshold: float,
                                  *,
                                  mode: str,
                                  near_best_rule: str = "relative",
                                  near_best_epsilon: float = 0.05,
                                  near_best_std_multiplier: float = 1.0,
                                  max_set_size: Optional[int] = None) -> ConformalPrediction:
    validate_conformal_mode(mode)
    validate_near_best_rule(near_best_rule)

    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)
    sources = np.asarray(sources)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if sources.shape[0] != scores.shape[0]:
        raise ValueError("sources and scores must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    set_mask, effective_k, capped = conformal_source_set(
        scores,
        threshold,
        mode=mode,
        max_set_size=max_set_size,
    )

    set_size = int(set_mask.sum())
    set_size_fraction = float(set_size / scores.shape[0])

    best_idx = int(np.argmax(performances))
    best_performance = float(performances[best_idx])
    contains_best = float(bool(set_mask[best_idx]))

    good_mask = near_best_mask(
        performances,
        rule=near_best_rule,
        epsilon=near_best_epsilon,
        std_multiplier=near_best_std_multiplier,
    )
    contains_near_best = float(bool(np.any(set_mask & good_mask)))

    if set_size == 0:
        best_in_set_performance = float("nan")
        best_in_set_loss = float("nan")
    else:
        best_in_set_performance = float(np.max(performances[set_mask]))
        if best_performance <= 0:
            best_in_set_loss = float("nan")
        else:
            best_in_set_loss = float(
                (best_performance - best_in_set_performance) / best_performance
            )

    return ConformalPrediction(
        threshold=float(threshold),
        mode=mode,
        effective_k=effective_k,
        set_mask=set_mask,
        set_sources=sources[set_mask].tolist(),
        set_size=set_size,
        set_size_fraction=set_size_fraction,
        capped=capped,
        contains_best_source=contains_best,
        contains_near_best_source=contains_near_best,
        singleton=float(set_size == 1),
        empty=float(set_size == 0),
        best_in_set_performance=best_in_set_performance,
        best_in_set_performance_loss=best_in_set_loss,
    )


def evaluate_top_k_near_best_budget(scores: np.ndarray,
                                    performances: np.ndarray,
                                    *,
                                    k: int,
                                    rule: str,
                                    epsilon: float = 0.05,
                                    std_multiplier: float = 1.0) -> TopKOperationalResult:
    validate_near_best_rule(rule)

    scores = np.asarray(scores, dtype=float)
    performances = np.asarray(performances, dtype=float)

    if scores.shape[0] != performances.shape[0]:
        raise ValueError("scores and performances must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("At least one source candidate is required")

    set_mask = _top_k_mask(scores, k)
    set_size = int(set_mask.sum())

    good_mask = near_best_mask(
        performances,
        rule=rule,
        epsilon=epsilon,
        std_multiplier=std_multiplier,
    )
    contains_near_best = float(bool(np.any(set_mask & good_mask)))

    best_performance = float(np.max(performances))

    if set_size == 0:
        best_in_set_performance = float("nan")
        best_in_set_loss = float("nan")
    else:
        best_in_set_performance = float(np.max(performances[set_mask]))
        if best_performance <= 0:
            best_in_set_loss = float("nan")
        else:
            best_in_set_loss = float(
                (best_performance - best_in_set_performance) / best_performance
            )

    return TopKOperationalResult(
        k=int(k),
        rule=rule,
        epsilon=float(epsilon),
        std_multiplier=float(std_multiplier),
        set_size=set_size,
        contains_near_best_source=contains_near_best,
        best_in_set_performance=best_in_set_performance,
        best_in_set_performance_loss=best_in_set_loss,
    )


def evaluate_top_k_std_budget(scores: np.ndarray,
                              performances: np.ndarray,
                              *,
                              k: int,
                              std_multiplier: float) -> TopKOperationalResult:
    return evaluate_top_k_near_best_budget(
        scores,
        performances,
        k=k,
        rule="std",
        std_multiplier=std_multiplier,
    )


def serialize_sources(sources: list[Any]) -> str:
    return ",".join(str(source) for source in sources)