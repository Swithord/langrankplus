from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from tqdm import tqdm

from .rankers.base import BaseRanker
from .validation import validate_dataset
from .metrics import (
    relevance_from_performance,
    ndcg_at_k,
    performance_loss,
    top_k_accuracy,
    paired_ttest,
)


@dataclass
class EvaluationResult:
    """
    Result of a leave-one-group-out cross-validated evaluation.

    Means are reported as percentages (out of 100). Per-fold lists hold raw values
    (NDCG in [0, 1], performance_loss as a relative ratio) for downstream paired tests.
    """
    mean_ndcg: float                          # mean NDCG@k as percentage
    ndcg_scores: list[float]                  # per-fold NDCG@k in [0, 1]
    mean_performance_loss: float              # mean relative loss as percentage
    performance_losses: list[float]           # per-fold relative loss
    mean_top_1_accuracy: float                # mean top-1 hit rate as percentage
    mean_top_3_accuracy: float                # mean top-3 hit rate as percentage
    per_fold: pd.DataFrame                    # one row per held-out target
    k: int = 3

    def __repr__(self) -> str:
        return (f"EvaluationResult(ndcg@{self.k}={self.mean_ndcg:.2f}, "
                f"perf_loss={self.mean_performance_loss:.2f}, "
                f"top1={self.mean_top_1_accuracy:.2f}, "
                f"top3={self.mean_top_3_accuracy:.2f})")


class TransferEvaluator:
    """
    Evaluate a ranker via leave-one-group-out cross-validation.
    For each target language, the model is fit on the remaining targets (with a held-out
    group-based validation split for early stopping), then asked to rank
    the sources for the held-out target.
    NDCG@k, relative performance loss, and top-k accuracy are aggregated across folds.
    """

    def __init__(self,
                 target_col: str = 'task_lang',
                 source_col: str = 'transfer_lang',
                 performance_col: str = 'performance',
                 k: int = 3,
                 top_k_relevance: int = 10,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        :param target_col: column containing the target/task language code
        :param source_col: column containing the source/transfer language code
        :param performance_col: column containing the downstream performance score
        :param k: cutoff for NDCG@k
        :param top_k_relevance: top-k items per target get graded relevance (k..1, rest 0)
        :param val_size: validation fraction (group-aware) for early stopping
        :param random_state: seed for the validation split
        :param verbose: whether to show a tqdm progress bar over folds
        """
        self.target_col = target_col
        self.source_col = source_col
        self.performance_col = performance_col
        self.k = k
        self.top_k_relevance = top_k_relevance
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose

    def _add_relevance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a `_relevance` column based on within-target performance ranking.
        """
        df = df.copy()
        df['_relevance'] = 0.0
        for target in df[self.target_col].unique():
            mask = df[self.target_col] == target
            perf = df.loc[mask, self.performance_col].to_numpy()
            df.loc[mask, '_relevance'] = relevance_from_performance(
                perf, top_k=self.top_k_relevance)
        return df

    def evaluate(self,
                 ranker: BaseRanker,
                 df: pd.DataFrame,
                 feature_cols: list[str]) -> EvaluationResult:
        """
        Run leave-one-group-out CV and compute metrics.
        :param ranker: the ranker to evaluate. Cloned per fold via sklearn.base.clone
            so the passed-in instance is not mutated.
        :param df: dataset with feature, target, source, and performance columns
        :param feature_cols: list of feature column names
        :return: EvaluationResult
        """
        validate_dataset(df, feature_cols, self.target_col, self.source_col,
                         self.performance_col)

        ranks = (df.groupby(self.target_col)[self.performance_col]
                 .rank(method='min', ascending=False))
        df['_relevance'] = np.where(ranks <= self.top_k_relevance,
                                    self.top_k_relevance + 1 - ranks,
                                    0.0)

        groups = df[self.target_col].to_numpy()
        logo = LeaveOneGroupOut()
        splits = list(logo.split(df, groups=groups))

        ndcg_scores = []
        perf_losses = []
        top_1_hits = []
        top_3_hits = []
        per_fold_records = []

        iterator = tqdm(splits, desc='LOO-CV', disable=not self.verbose)

        for train_val_idx, test_idx in tqdm(iterator):
            train_val_data = df.iloc[train_val_idx]
            test_data = df.iloc[test_idx]

            # Group-aware train/val split for early stopping
            tv_groups = train_val_data[self.target_col].to_numpy()
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.val_size,
                                         random_state=self.random_state)
            train_idx, val_idx = next(splitter.split(train_val_data, groups=tv_groups))
            train_data = train_val_data.iloc[train_idx]
            val_data = train_val_data.iloc[val_idx]

            X_train = train_data[feature_cols].to_numpy()
            y_train = train_data['_relevance'].to_numpy()
            groups_train = train_data[self.target_col].to_numpy()

            X_val = val_data[feature_cols].to_numpy()
            y_val = val_data['_relevance'].to_numpy()
            groups_val = val_data[self.target_col].to_numpy()

            X_test = test_data[feature_cols].to_numpy()
            y_test = test_data['_relevance'].to_numpy()

            fold_ranker = clone(ranker)

            fold_ranker.fit(X_train, y_train, groups=groups_train,
                            eval_set=(X_val, y_val), eval_groups=groups_val)
            y_pred = fold_ranker.predict(X_test)

            fold_ndcg = ndcg_at_k(y_test, y_pred, k=self.k)
            ndcg_scores.append(fold_ndcg)

            test_perf = test_data[self.performance_col].to_numpy()
            pred_best_idx = int(np.argmax(y_pred))
            actual_best_idx = int(np.argmax(test_perf))
            pred_best_perf = float(test_perf[pred_best_idx])
            actual_best_perf = float(test_perf[actual_best_idx])

            ploss = performance_loss(pred_best_perf, actual_best_perf)
            if not np.isnan(ploss):
                perf_losses.append(ploss)

            top_1_hits.append(top_k_accuracy(y_test, y_pred, k=1))
            top_3_hits.append(top_k_accuracy(y_test, y_pred, k=3))

            target_lang = test_data[self.target_col].iloc[0]
            pred_best_source = test_data[self.source_col].iloc[pred_best_idx]
            actual_best_source = test_data[self.source_col].iloc[actual_best_idx]
            per_fold_records.append({
                'target_lang': target_lang,
                'predicted_best_source': pred_best_source,
                'actual_best_source': actual_best_source,
                'predicted_performance': pred_best_perf,
                'actual_best_performance': actual_best_perf,
                'performance_loss': ploss,
                'ndcg': fold_ndcg,
            })

        per_fold_df = pd.DataFrame(per_fold_records)

        mean_perf_loss = float(np.mean(perf_losses) * 100) if perf_losses else float('nan')

        return EvaluationResult(
            mean_ndcg=float(np.mean(ndcg_scores) * 100),
            ndcg_scores=ndcg_scores,
            mean_performance_loss=mean_perf_loss,
            performance_losses=perf_losses,
            mean_top_1_accuracy=float(np.mean(top_1_hits) * 100),
            mean_top_3_accuracy=float(np.mean(top_3_hits) * 100),
            per_fold=per_fold_df,
            k=self.k,
        )

    @staticmethod
    def compare(result_a: EvaluationResult,
                result_b: EvaluationResult) -> dict[str, float]:
        """
        Paired t-test on per-fold NDCG between two results. Both results must come from
        the same dataset (so per-fold lists align by target).
        :return: dict with `ndcg_p_value`
        """
        return {
            'ndcg_p_value': paired_ttest(result_a.ndcg_scores, result_b.ndcg_scores),
        }
