import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import ttest_rel, rankdata


def relevance_from_performance(perf: np.ndarray, top_k: int = 10) -> np.ndarray:
    """
    Convert performance scores to graded relevance (top_k items get top_k..1, rest 0).
    Uses 'min' rank method for ties (matching pandas .rank(method='min')).
    :param perf: array of performance scores
    :param top_k: number of top items to assign relevance to
    :return: array of relevance scores
    """
    relevance = np.zeros(len(perf))
    ranks = rankdata(-perf, method='min')
    mask = ranks <= top_k
    relevance[mask] = top_k + 1 - ranks[mask]
    return relevance


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """
    Compute NDCG@k for a single query.
    :param y_true: true relevance scores
    :param y_pred: predicted scores (higher is better)
    :param k: cutoff
    :return: NDCG@k
    """
    return ndcg_score([y_true], [y_pred], k=k)


def performance_loss(predicted_perf: float, optimal_perf: float) -> float:
    """
    Compute relative performance loss: |predicted - optimal| / optimal.
    :param predicted_perf: performance of the predicted-best source
    :param optimal_perf: performance of the actually-best source
    :return: relative performance loss, or NaN if optimal is 0
    """
    if optimal_perf == 0:
        return np.nan
    return abs(predicted_perf - optimal_perf) / optimal_perf


def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1) -> float:
    """
    Whether the actual best source is in the top-k predicted.
    :param y_true: true relevance scores
    :param y_pred: predicted scores
    :param k: cutoff
    :return: 1.0 if hit, 0.0 otherwise
    """
    top_k_pred = set(np.argsort(y_pred)[-k:].tolist())
    actual_best = int(np.argmax(y_true))
    return float(actual_best in top_k_pred)


def paired_ttest(scores_a: list[float], scores_b: list[float]) -> float:
    """
    Paired t-test between two lists of per-fold scores.
    :param scores_a: first list (e.g. NDCG scores from method A)
    :param scores_b: second list
    :return: p-value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Score lists must have the same length: "
                         f"{len(scores_a)} vs {len(scores_b)}")
    _, p_value = ttest_rel(scores_a, scores_b)
    return float(p_value)
