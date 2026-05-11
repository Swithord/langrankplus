from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from tqdm import tqdm

from .rankers.base import BaseRanker
from .validation import validate_dataset
from .data import add_query_id, get_query_cols
from .metrics import (
    ndcg_at_k,
    performance_loss,
    top_k_accuracy,
    paired_ttest,
)
from .conformal import (
    calibration_scores_from_queries,
    conformal_quantile,
    evaluate_conformal_source_set,
    serialize_sources,
)


@dataclass
class EvaluationResult:
    """
    Result of a leave-one-query-out cross-validated evaluation.

    Means are reported as percentages. Per-fold lists hold raw values
    (NDCG in [0, 1], performance_loss as a relative ratio) for downstream tests.
    """
    method_name: str
    mean_ndcg: float
    ndcg_scores: list[float]
    mean_performance_loss: float
    performance_losses: list[float]
    mean_top_1_accuracy: float
    mean_top_3_accuracy: float
    per_fold: pd.DataFrame
    k: int = 3

    mean_conformal_best_source_coverage: Optional[float] = None
    mean_conformal_set_size: Optional[float] = None
    mean_conformal_singleton_rate: Optional[float] = None
    mean_conformal_empty_rate: Optional[float] = None
    mean_conformal_best_in_set_loss: Optional[float] = None
    conformal_alpha: Optional[float] = None

    def __repr__(self) -> str:
        out = (f"EvaluationResult(method={self.method_name}, "
               f"ndcg@{self.k}={self.mean_ndcg:.2f}, "
               f"perf_loss={self.mean_performance_loss:.2f}, "
               f"top1={self.mean_top_1_accuracy:.2f}, "
               f"top3={self.mean_top_3_accuracy:.2f}")

        if self.mean_conformal_best_source_coverage is not None:
            out += (f", conf_cov={self.mean_conformal_best_source_coverage:.2f}, "
                    f"conf_size={self.mean_conformal_set_size:.2f}")

        return out + ")"


FoldRankerFactory = Callable[[pd.DataFrame, list[str]], BaseRanker]


class TransferEvaluator:
    """
    Evaluate source-language rankers via leave-one-query-out cross-validation.

    A query is:
    - task_lang for a single dataset;
    - (dataset, task_lang) for multi-dataset evaluation.

    In each outer fold, all fitted methods receive the same fitting rows. If
    conformal evaluation is enabled, the non-held-out queries are first split into
    fitting queries and conformal-calibration queries. The conformal-calibration
    queries are used only to estimate the conformal threshold.
    """

    def __init__(self,
                 target_col: str = 'task_lang',
                 source_col: str = 'transfer_lang',
                 performance_col: str = 'performance',
                 dataset_col: Optional[str] = 'dataset',
                 k: int = 3,
                 top_k_relevance: int = 10,
                 val_size: float = 0.0,
                 random_state: int = 42,
                 verbose: bool = False,
                 include_conformal: bool = False,
                 conformal_alpha: float = 0.1,
                 conformal_cal_size: float = 0.2):
        self.target_col = target_col
        self.source_col = source_col
        self.performance_col = performance_col
        self.dataset_col = dataset_col
        self.k = k
        self.top_k_relevance = top_k_relevance
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.include_conformal = include_conformal
        self.conformal_alpha = conformal_alpha
        self.conformal_cal_size = conformal_cal_size

    def _prepare(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        validate_dataset(
            df,
            feature_cols=feature_cols,
            target_col=self.target_col,
            source_col=self.source_col,
            performance_col=self.performance_col,
            dataset_col=self.dataset_col,
            require_performance=True,
            require_complete_matrix=False,
        )

        out = df.copy()
        query_cols = get_query_cols(
            out,
            target_col=self.target_col,
            dataset_col=self.dataset_col,
        )

        ranks = (out.groupby(query_cols)[self.performance_col]
                 .rank(method='min', ascending=False))
        out['_relevance'] = np.where(
            ranks <= self.top_k_relevance,
            self.top_k_relevance + 1 - ranks,
            0.0,
        )

        out = add_query_id(
            out,
            target_col=self.target_col,
            dataset_col=self.dataset_col,
            query_id_col='_query_id',
        )

        return out

    def _split_train_val(self,
                         fitting_data: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Optionally split the fitting queries into training and validation queries.

        With val_size=0, all fitted methods receive all fitting rows.
        """
        if self.val_size <= 0:
            return fitting_data, None

        groups = fitting_data['_query_id'].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return fitting_data, None

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.val_size,
            random_state=self.random_state,
        )
        train_idx, val_idx = next(splitter.split(fitting_data, groups=groups))

        train_data = fitting_data.iloc[train_idx]
        val_data = fitting_data.iloc[val_idx]

        if train_data['_query_id'].nunique() == 0 or val_data['_query_id'].nunique() == 0:
            return fitting_data, None

        return train_data, val_data

    def _split_fitting_conformal(self,
                                 train_val_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split non-held-out queries into fitting data and conformal-calibration data.
        """
        groups = train_val_data['_query_id'].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 3:
            raise ValueError("Conformal evaluation requires at least three non-held-out "
                             "queries so that fitting and calibration splits are nonempty")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.conformal_cal_size,
            random_state=self.random_state,
        )
        fitting_idx, conformal_idx = next(splitter.split(train_val_data, groups=groups))

        fitting_data = train_val_data.iloc[fitting_idx]
        conformal_data = train_val_data.iloc[conformal_idx]

        if fitting_data['_query_id'].nunique() == 0 or conformal_data['_query_id'].nunique() == 0:
            raise ValueError("Conformal split produced an empty fitting or calibration set")

        return fitting_data, conformal_data

    def _fit_fold_ranker(self,
                         ranker: BaseRanker,
                         train_data: pd.DataFrame,
                         val_data: Optional[pd.DataFrame],
                         feature_cols: list[str]) -> BaseRanker:
        """
        Fit a fold ranker using the BaseRanker interface.
        """
        X_train = train_data[feature_cols].to_numpy(dtype=float)
        y_train = train_data['_relevance'].to_numpy(dtype=float)
        groups_train = train_data['_query_id'].to_numpy()

        if val_data is None:
            ranker.fit(X_train, y_train, groups=groups_train)
            return ranker

        X_val = val_data[feature_cols].to_numpy(dtype=float)
        y_val = val_data['_relevance'].to_numpy(dtype=float)
        groups_val = val_data['_query_id'].to_numpy()

        ranker.fit(
            X_train,
            y_train,
            groups=groups_train,
            eval_set=(X_val, y_val),
            eval_groups=groups_val,
        )
        return ranker

    def evaluate(self,
                 ranker: BaseRanker,
                 df: pd.DataFrame,
                 feature_cols: list[str],
                 method_name: Optional[str] = None,
                 fold_ranker_factory: Optional[FoldRankerFactory] = None) -> EvaluationResult:
        """
        Run leave-one-query-out CV and compute point-selection metrics. If
        include_conformal=True, also compute conformal source-set metrics.
        """
        work_df = self._prepare(df, feature_cols)
        groups = work_df['_query_id'].to_numpy()

        logo = LeaveOneGroupOut()
        splits = list(logo.split(work_df, groups=groups))

        ndcg_scores = []
        perf_losses = []
        top_1_hits = []
        top_3_hits = []
        per_fold_records = []

        conformal_coverages = []
        conformal_set_sizes = []
        conformal_singletons = []
        conformal_empty = []
        conformal_best_in_set_losses = []

        iterator = tqdm(splits, desc='LOO-CV', disable=not self.verbose)

        for train_val_idx, test_idx in iterator:
            train_val_data = work_df.iloc[train_val_idx].copy()
            test_data = work_df.iloc[test_idx].copy()

            if self.include_conformal:
                fitting_pool_data, conformal_cal_data = self._split_fitting_conformal(
                    train_val_data
                )
            else:
                fitting_pool_data = train_val_data
                conformal_cal_data = None

            train_data, val_data = self._split_train_val(fitting_pool_data)

            if fold_ranker_factory is None:
                fold_ranker = clone(ranker)
            else:
                fold_ranker = fold_ranker_factory(train_data, feature_cols)

            fold_ranker = self._fit_fold_ranker(
                ranker=fold_ranker,
                train_data=train_data,
                val_data=val_data,
                feature_cols=feature_cols,
            )

            if self.include_conformal:
                calibration_nonconformity = calibration_scores_from_queries(
                    conformal_cal_data,
                    ranker=fold_ranker,
                    feature_cols=feature_cols,
                    performance_col=self.performance_col,
                    query_col='_query_id',
                )
                conformal_threshold = conformal_quantile(
                    calibration_nonconformity,
                    alpha=self.conformal_alpha,
                )
            else:
                conformal_threshold = None

            X_test = test_data[feature_cols].to_numpy(dtype=float)
            y_test = test_data['_relevance'].to_numpy(dtype=float)
            y_pred = fold_ranker.predict(X_test)

            fold_ndcg = ndcg_at_k(y_test, y_pred, k=self.k)
            ndcg_scores.append(fold_ndcg)

            test_perf = test_data[self.performance_col].to_numpy(dtype=float)
            test_sources = test_data[self.source_col].to_numpy()

            pred_best_idx = int(np.argmax(y_pred))
            actual_best_idx = int(np.argmax(test_perf))
            pred_best_perf = float(test_perf[pred_best_idx])
            actual_best_perf = float(test_perf[actual_best_idx])

            ploss = performance_loss(pred_best_perf, actual_best_perf)
            if not np.isnan(ploss):
                perf_losses.append(ploss)

            top_1_hits.append(top_k_accuracy(y_test, y_pred, k=1))
            top_3_hits.append(top_k_accuracy(y_test, y_pred, k=3))

            record = {
                'method': method_name or ranker.__class__.__name__,
                'query_id': test_data['_query_id'].iloc[0],
                'target_lang': test_data[self.target_col].iloc[0],
                'predicted_best_source': test_data[self.source_col].iloc[pred_best_idx],
                'actual_best_source': test_data[self.source_col].iloc[actual_best_idx],
                'predicted_performance': pred_best_perf,
                'actual_best_performance': actual_best_perf,
                'performance_loss': ploss,
                'ndcg': fold_ndcg,
                'top_1_accuracy': top_1_hits[-1],
                'top_3_accuracy': top_3_hits[-1],
            }

            if self.dataset_col is not None and self.dataset_col in test_data.columns:
                record['dataset'] = test_data[self.dataset_col].iloc[0]

            if self.include_conformal:
                conformal_prediction = evaluate_conformal_source_set(
                    scores=y_pred,
                    performances=test_perf,
                    sources=test_sources,
                    threshold=conformal_threshold,
                )

                conformal_coverages.append(conformal_prediction.contains_best_source)
                conformal_set_sizes.append(conformal_prediction.set_size)
                conformal_singletons.append(conformal_prediction.singleton)
                conformal_empty.append(conformal_prediction.empty)

                if not np.isnan(conformal_prediction.best_in_set_performance_loss):
                    conformal_best_in_set_losses.append(
                        conformal_prediction.best_in_set_performance_loss
                    )

                record.update({
                    'conformal_alpha': self.conformal_alpha,
                    'conformal_threshold': conformal_prediction.threshold,
                    'conformal_set_size': conformal_prediction.set_size,
                    'conformal_contains_best_source': conformal_prediction.contains_best_source,
                    'conformal_singleton': conformal_prediction.singleton,
                    'conformal_empty': conformal_prediction.empty,
                    'conformal_best_in_set_performance': (
                        conformal_prediction.best_in_set_performance
                    ),
                    'conformal_best_in_set_performance_loss': (
                        conformal_prediction.best_in_set_performance_loss
                    ),
                    'conformal_sources': serialize_sources(conformal_prediction.set_sources),
                })

            per_fold_records.append(record)

        per_fold_df = pd.DataFrame(per_fold_records)
        name = method_name or ranker.__class__.__name__

        result = EvaluationResult(
            method_name=name,
            mean_ndcg=float(np.mean(ndcg_scores) * 100),
            ndcg_scores=ndcg_scores,
            mean_performance_loss=(
                float(np.mean(perf_losses) * 100) if perf_losses else float('nan')
            ),
            performance_losses=perf_losses,
            mean_top_1_accuracy=float(np.mean(top_1_hits) * 100),
            mean_top_3_accuracy=float(np.mean(top_3_hits) * 100),
            per_fold=per_fold_df,
            k=self.k,
        )

        if self.include_conformal:
            result.mean_conformal_best_source_coverage = float(
                np.mean(conformal_coverages) * 100
            )
            result.mean_conformal_set_size = float(np.mean(conformal_set_sizes))
            result.mean_conformal_singleton_rate = float(np.mean(conformal_singletons) * 100)
            result.mean_conformal_empty_rate = float(np.mean(conformal_empty) * 100)
            result.mean_conformal_best_in_set_loss = (
                float(np.mean(conformal_best_in_set_losses) * 100)
                if conformal_best_in_set_losses else float('nan')
            )
            result.conformal_alpha = self.conformal_alpha

        return result

    @staticmethod
    def compare(result_a: EvaluationResult,
                result_b: EvaluationResult) -> dict[str, float]:
        """
        Paired tests between two results. Lists must align by held-out query.
        """
        return {
            'ndcg_p_value': paired_ttest(result_a.ndcg_scores, result_b.ndcg_scores),
            'performance_loss_p_value': paired_ttest(
                result_a.performance_losses,
                result_b.performance_losses,
            ),
        }


def results_to_summary(results: list[EvaluationResult]) -> pd.DataFrame:
    """
    Convert evaluation results to a method-level summary table.
    """
    rows = []
    for result in results:
        row = {
            'method': result.method_name,
            f'ndcg@{result.k}': result.mean_ndcg,
            'performance_loss': result.mean_performance_loss,
            'top_1_accuracy': result.mean_top_1_accuracy,
            'top_3_accuracy': result.mean_top_3_accuracy,
            'n_folds': len(result.ndcg_scores),
        }

        if result.mean_conformal_best_source_coverage is not None:
            row.update({
                'conformal_alpha': result.conformal_alpha,
                'conformal_best_source_coverage': (
                    result.mean_conformal_best_source_coverage
                ),
                'conformal_average_set_size': result.mean_conformal_set_size,
                'conformal_singleton_rate': result.mean_conformal_singleton_rate,
                'conformal_empty_rate': result.mean_conformal_empty_rate,
                'conformal_best_in_set_performance_loss': (
                    result.mean_conformal_best_in_set_loss
                ),
            })

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    sort_cols = ['performance_loss', f'ndcg@{results[0].k}']
    return pd.DataFrame(rows).sort_values(by=sort_cols, ascending=[True, False])


def results_to_per_fold(results: list[EvaluationResult]) -> pd.DataFrame:
    """
    Stack per-fold results from all methods.
    """
    frames = [result.per_fold.copy() for result in results]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pairwise_comparisons(results: list[EvaluationResult]) -> pd.DataFrame:
    """
    Pairwise paired t-tests for NDCG and performance loss.
    """
    rows = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            tests = TransferEvaluator.compare(a, b)
            rows.append({
                'method_a': a.method_name,
                'method_b': b.method_name,
                **tests,
            })
    return pd.DataFrame(rows)