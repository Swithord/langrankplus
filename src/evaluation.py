from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from tqdm import tqdm

from .conformal import (
    calibration_scores_from_queries,
    conformal_quantile,
    evaluate_conformal_source_set,
    evaluate_top_k_near_best_budget,
    metric_float_tag,
    serialize_sources,
)
from .data import add_query_id, get_query_cols
from .metrics import (
    compute_ir_metrics,
    ndcg_at_k,
    paired_ttest,
    performance_loss,
    top_k_accuracy,
)
from .rankers.base import BaseRanker
from .resource_level_langs import RESOURCE_LEVELS, resource_level
from .validation import validate_dataset


@dataclass
class EvaluationResult:
    method_name: str
    mean_ndcg: float
    ndcg_scores: list[float]
    mean_performance_loss: float
    performance_losses: list[float]
    mean_top_1_accuracy: float
    mean_top_3_accuracy: float
    per_fold: pd.DataFrame
    k: int = 3

    ir_metrics: dict[str, float] = field(default_factory=dict)
    operational_metrics: dict[str, float] = field(default_factory=dict)
    conformal_metrics: dict[str, float] = field(default_factory=dict)

    conformal_mode: Optional[str] = None
    conformal_alpha: Optional[float] = None
    conformal_near_best_rules: Optional[list[str]] = None
    near_best_epsilon: Optional[float] = None
    near_best_std_multiplier: Optional[float] = None
    conformal_max_set_sizes: Optional[list[int]] = None

    def __repr__(self) -> str:
        out = (
            f"EvaluationResult(method={self.method_name}, "
            f"ndcg@{self.k}={self.mean_ndcg:.2f}, "
            f"perf_loss={self.mean_performance_loss:.2f}, "
            f"top1={self.mean_top_1_accuracy:.2f}, "
            f"top3={self.mean_top_3_accuracy:.2f}"
        )

        if "mrr" in self.ir_metrics:
            out += f", mrr={self.ir_metrics['mrr']:.2f}"

        if self.conformal_metrics:
            rel_key = "conformal_relative_uncapped_near_best_coverage"
            std_key = "conformal_std_uncapped_near_best_coverage"
            size_key = "conformal_relative_uncapped_average_set_size"

            if rel_key in self.conformal_metrics:
                out += f", conf_rel_cov={self.conformal_metrics[rel_key]:.2f}"
            if std_key in self.conformal_metrics:
                out += f", conf_std_cov={self.conformal_metrics[std_key]:.2f}"
            if size_key in self.conformal_metrics:
                out += f", conf_size={self.conformal_metrics[size_key]:.2f}"

        return out + ")"


FoldRankerFactory = Callable[[pd.DataFrame, list[str]], BaseRanker]


class TransferEvaluator:
    def __init__(self,
                 target_col: str = "task_lang",
                 source_col: str = "transfer_lang",
                 performance_col: str = "performance",
                 dataset_col: Optional[str] = "dataset",
                 k: int = 3,
                 top_k_relevance: int = 10,
                 val_size: float = 0.0,
                 random_state: int = 42,
                 verbose: bool = False,
                 include_conformal: bool = False,
                 conformal_alpha: float = 0.1,
                 conformal_cal_size: float = 0.2,
                 conformal_mode: str = "rank_near_best",
                 near_best_rule: str = "relative",
                 conformal_near_best_rules: Optional[Sequence[str]] = None,
                 near_best_epsilon: float = 0.05,
                 near_best_std_multiplier: float = 1.0,
                 conformal_max_set_size: Optional[int] = None,
                 conformal_max_set_sizes: Optional[Sequence[int]] = None,
                 operational_relative_epsilons: Sequence[float] = (0.05,),
                 operational_std_multipliers: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
                 operational_top_k: Sequence[int] = (3, 5, 10),
                 ir_cutoffs: Sequence[int] = (1, 3, 5, 10)):
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
        self.conformal_mode = conformal_mode
        self.near_best_rule = near_best_rule
        self.near_best_epsilon = near_best_epsilon
        self.near_best_std_multiplier = near_best_std_multiplier

        if conformal_near_best_rules is None:
            self.conformal_near_best_rules = ("relative", "std")
        else:
            self.conformal_near_best_rules = tuple(conformal_near_best_rules)

        caps: list[Optional[int]] = [None]
        if conformal_max_set_sizes is not None:
            caps.extend(int(x) for x in conformal_max_set_sizes)
        elif conformal_max_set_size is not None:
            caps.append(int(conformal_max_set_size))

        seen = set()
        self.conformal_caps: list[Optional[int]] = []
        for cap in caps:
            key = "none" if cap is None else int(cap)
            if key not in seen:
                seen.add(key)
                self.conformal_caps.append(cap)

        self.operational_relative_epsilons = tuple(float(x) for x in operational_relative_epsilons)
        self.operational_std_multipliers = tuple(float(x) for x in operational_std_multipliers)
        self.operational_top_k = tuple(int(x) for x in operational_top_k)
        self.ir_cutoffs = tuple(sorted(set(int(x) for x in ir_cutoffs)))

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

        ranks = out.groupby(query_cols)[self.performance_col].rank(
            method="min",
            ascending=False,
        )
        out["_relevance"] = np.where(
            ranks <= self.top_k_relevance,
            self.top_k_relevance + 1 - ranks,
            0.0,
        )

        out = add_query_id(
            out,
            target_col=self.target_col,
            dataset_col=self.dataset_col,
            query_id_col="_query_id",
        )

        return out

    def _split_train_val(self,
                         fitting_data: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.val_size <= 0:
            return fitting_data, None

        groups = fitting_data["_query_id"].to_numpy()
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

        if train_data["_query_id"].nunique() == 0 or val_data["_query_id"].nunique() == 0:
            return fitting_data, None

        return train_data, val_data

    def _split_fitting_conformal(self,
                                 train_val_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        groups = train_val_data["_query_id"].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 3:
            raise ValueError(
                "Conformal evaluation requires at least three non-held-out queries."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.conformal_cal_size,
            random_state=self.random_state,
        )
        fitting_idx, conformal_idx = next(splitter.split(train_val_data, groups=groups))

        fitting_data = train_val_data.iloc[fitting_idx]
        conformal_data = train_val_data.iloc[conformal_idx]

        if fitting_data["_query_id"].nunique() == 0 or conformal_data["_query_id"].nunique() == 0:
            raise ValueError("Conformal split produced an empty fitting or calibration set.")

        return fitting_data, conformal_data

    def _fit_fold_ranker(self,
                         ranker: BaseRanker,
                         train_data: pd.DataFrame,
                         val_data: Optional[pd.DataFrame],
                         feature_cols: list[str]) -> BaseRanker:
        X_train = train_data[feature_cols].to_numpy(dtype=float)
        y_train = train_data["_relevance"].to_numpy(dtype=float)
        groups_train = train_data["_query_id"].to_numpy()

        if val_data is None:
            ranker.fit(X_train, y_train, groups=groups_train)
            return ranker

        X_val = val_data[feature_cols].to_numpy(dtype=float)
        y_val = val_data["_relevance"].to_numpy(dtype=float)
        groups_val = val_data["_query_id"].to_numpy()

        ranker.fit(
            X_train,
            y_train,
            groups=groups_train,
            eval_set=(X_val, y_val),
            eval_groups=groups_val,
        )
        return ranker

    @staticmethod
    def _conformal_prefix(rule: str, cap: Optional[int]) -> str:
        if cap is None:
            return f"conformal_{rule}_uncapped"
        return f"conformal_{rule}_cap_{int(cap)}"

    def _evaluate_operational_topk(self,
                                   scores: np.ndarray,
                                   performances: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        per_fold = {}
        accum = {}

        for k_value in self.operational_top_k:
            budget_result = evaluate_top_k_near_best_budget(
                scores,
                performances,
                k=k_value,
                rule="relative",
                epsilon=0.0,
            )

            loss_key = f"top_{k_value}_best_in_set_performance_loss"
            set_size_key = f"top_{k_value}_set_size"

            per_fold[loss_key] = budget_result.best_in_set_performance_loss
            per_fold[set_size_key] = budget_result.set_size
            accum[loss_key] = budget_result.best_in_set_performance_loss
            accum[set_size_key] = budget_result.set_size

            for epsilon in self.operational_relative_epsilons:
                tag = metric_float_tag(epsilon)
                result = evaluate_top_k_near_best_budget(
                    scores,
                    performances,
                    k=k_value,
                    rule="relative",
                    epsilon=epsilon,
                )

                coverage_key = f"top_{k_value}_relative_{tag}_coverage"
                per_fold_key = f"top_{k_value}_relative_{tag}_contains_near_best"

                per_fold[per_fold_key] = result.contains_near_best_source
                accum[coverage_key] = result.contains_near_best_source

            for std_multiplier in self.operational_std_multipliers:
                tag = metric_float_tag(std_multiplier)
                result = evaluate_top_k_near_best_budget(
                    scores,
                    performances,
                    k=k_value,
                    rule="std",
                    std_multiplier=std_multiplier,
                )

                coverage_key = f"top_{k_value}_std_{tag}_coverage"
                per_fold_key = f"top_{k_value}_std_{tag}_contains_near_best"

                per_fold[per_fold_key] = result.contains_near_best_source
                accum[coverage_key] = result.contains_near_best_source

        return per_fold, accum

    def evaluate(self,
                 ranker: BaseRanker,
                 df: pd.DataFrame,
                 feature_cols: list[str],
                 method_name: Optional[str] = None,
                 fold_ranker_factory: Optional[FoldRankerFactory] = None) -> EvaluationResult:
        work_df = self._prepare(df, feature_cols)
        groups = work_df["_query_id"].to_numpy()

        logo = LeaveOneGroupOut()
        splits = list(logo.split(work_df, groups=groups))

        ndcg_scores = []
        perf_losses = []
        top_1_hits = []
        top_3_hits = []
        per_fold_records = []

        ir_metric_lists: dict[str, list[float]] = {}
        operational_metric_lists: dict[str, list[float]] = {}
        conformal_metric_lists: dict[str, list[float]] = {}

        iterator = tqdm(splits, desc="LOO-CV", disable=not self.verbose)

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

            conformal_thresholds: dict[str, float] = {}
            if self.include_conformal:
                for rule in self.conformal_near_best_rules:
                    calibration_nonconformity = calibration_scores_from_queries(
                        conformal_cal_data,
                        ranker=fold_ranker,
                        feature_cols=feature_cols,
                        performance_col=self.performance_col,
                        query_col="_query_id",
                        mode=self.conformal_mode,
                        near_best_rule=rule,
                        near_best_epsilon=self.near_best_epsilon,
                        near_best_std_multiplier=self.near_best_std_multiplier,
                    )
                    conformal_thresholds[rule] = conformal_quantile(
                        calibration_nonconformity,
                        alpha=self.conformal_alpha,
                    )

            X_test = test_data[feature_cols].to_numpy(dtype=float)
            y_test = test_data["_relevance"].to_numpy(dtype=float)
            y_pred = fold_ranker.predict(X_test)

            fold_ndcg = ndcg_at_k(y_test, y_pred, k=self.k)
            ndcg_scores.append(fold_ndcg)

            fold_ir = compute_ir_metrics(
                y_test,
                y_pred,
                cutoffs=self.ir_cutoffs,
            )
            for key, value in fold_ir.items():
                if key not in ir_metric_lists:
                    ir_metric_lists[key] = []
                if not np.isnan(value):
                    ir_metric_lists[key].append(float(value))

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

            target_lang_val = test_data[self.target_col].iloc[0]
            predicted_source_val = test_data[self.source_col].iloc[pred_best_idx]
            actual_best_source_val = test_data[self.source_col].iloc[actual_best_idx]

            record = {
                "method": method_name or ranker.__class__.__name__,
                "query_id": test_data["_query_id"].iloc[0],
                "target_lang": target_lang_val,
                "target_resource_level": resource_level(target_lang_val),
                "predicted_best_source": predicted_source_val,
                "predicted_source_resource_level": resource_level(predicted_source_val),
                "actual_best_source": actual_best_source_val,
                "actual_best_source_resource_level": resource_level(actual_best_source_val),
                "predicted_performance": pred_best_perf,
                "actual_best_performance": actual_best_perf,
                "performance_loss": ploss,
                "ndcg": fold_ndcg,
                "top_1_accuracy": top_1_hits[-1],
                "top_3_accuracy": top_3_hits[-1],
            }

            record.update(fold_ir)

            if self.dataset_col is not None and self.dataset_col in test_data.columns:
                record["dataset"] = test_data[self.dataset_col].iloc[0]

            operational_per_fold, operational_accum = self._evaluate_operational_topk(
                scores=y_pred,
                performances=test_perf,
            )
            record.update(operational_per_fold)

            for key, value in operational_accum.items():
                if key not in operational_metric_lists:
                    operational_metric_lists[key] = []
                if not np.isnan(value):
                    operational_metric_lists[key].append(float(value))

            if self.include_conformal:
                for rule in self.conformal_near_best_rules:
                    for cap in self.conformal_caps:
                        prefix = self._conformal_prefix(rule, cap)

                        conformal_prediction = evaluate_conformal_source_set(
                            scores=y_pred,
                            performances=test_perf,
                            sources=test_sources,
                            threshold=conformal_thresholds[rule],
                            mode=self.conformal_mode,
                            near_best_rule=rule,
                            near_best_epsilon=self.near_best_epsilon,
                            near_best_std_multiplier=self.near_best_std_multiplier,
                            max_set_size=cap,
                        )

                        fold_values = {
                            f"{prefix}_threshold": conformal_prediction.threshold,
                            f"{prefix}_effective_k": conformal_prediction.effective_k,
                            f"{prefix}_set_size": conformal_prediction.set_size,
                            f"{prefix}_set_size_fraction": conformal_prediction.set_size_fraction,
                            f"{prefix}_capped": conformal_prediction.capped,
                            f"{prefix}_contains_best_source": (
                                conformal_prediction.contains_best_source
                            ),
                            f"{prefix}_contains_near_best_source": (
                                conformal_prediction.contains_near_best_source
                            ),
                            f"{prefix}_singleton": conformal_prediction.singleton,
                            f"{prefix}_empty": conformal_prediction.empty,
                            f"{prefix}_best_in_set_performance": (
                                conformal_prediction.best_in_set_performance
                            ),
                            f"{prefix}_best_in_set_performance_loss": (
                                conformal_prediction.best_in_set_performance_loss
                            ),
                            f"{prefix}_sources": serialize_sources(
                                conformal_prediction.set_sources
                            ),
                        }
                        record.update(fold_values)

                        summary_values = {
                            f"{prefix}_best_source_coverage": (
                                conformal_prediction.contains_best_source
                            ),
                            f"{prefix}_near_best_coverage": (
                                conformal_prediction.contains_near_best_source
                            ),
                            f"{prefix}_average_set_size": conformal_prediction.set_size,
                            f"{prefix}_median_set_size": conformal_prediction.set_size,
                            f"{prefix}_average_set_size_fraction": (
                                conformal_prediction.set_size_fraction
                            ),
                            f"{prefix}_singleton_rate": conformal_prediction.singleton,
                            f"{prefix}_empty_rate": conformal_prediction.empty,
                            f"{prefix}_capped_rate": conformal_prediction.capped,
                            f"{prefix}_best_in_set_performance_loss": (
                                conformal_prediction.best_in_set_performance_loss
                            ),
                        }

                        for key, value in summary_values.items():
                            if key not in conformal_metric_lists:
                                conformal_metric_lists[key] = []
                            if value is not None and not np.isnan(value):
                                conformal_metric_lists[key].append(float(value))

                record.update({
                    "conformal_mode": self.conformal_mode,
                    "conformal_alpha": self.conformal_alpha,
                    "conformal_near_best_rules": ",".join(self.conformal_near_best_rules),
                    "near_best_epsilon": self.near_best_epsilon,
                    "near_best_std_multiplier": self.near_best_std_multiplier,
                    "conformal_max_set_sizes": ",".join(
                        "uncapped" if cap is None else str(cap)
                        for cap in self.conformal_caps
                    ),
                })

            per_fold_records.append(record)

        per_fold_df = pd.DataFrame(per_fold_records)
        name = method_name or ranker.__class__.__name__

        ir_metrics = {}
        for key, values in ir_metric_lists.items():
            if not values:
                ir_metrics[key] = float("nan")
            elif key in {"exact_best_rank", "relevant_count"}:
                ir_metrics[key] = float(np.mean(values))
            else:
                ir_metrics[key] = float(np.mean(values) * 100)

        operational_metrics = {}
        for key, values in operational_metric_lists.items():
            if not values:
                operational_metrics[key] = float("nan")
            elif key.endswith("_coverage"):
                operational_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_performance_loss"):
                operational_metrics[key] = float(np.mean(values) * 100)
            else:
                operational_metrics[key] = float(np.mean(values))

        conformal_metrics = {}
        for key, values in conformal_metric_lists.items():
            if not values:
                conformal_metrics[key] = float("nan")
            elif key.endswith("_coverage"):
                conformal_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_fraction"):
                conformal_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_rate"):
                conformal_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_performance_loss"):
                conformal_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_median_set_size"):
                conformal_metrics[key] = float(np.median(values))
            else:
                conformal_metrics[key] = float(np.mean(values))

        result = EvaluationResult(
            method_name=name,
            mean_ndcg=float(np.mean(ndcg_scores) * 100),
            ndcg_scores=ndcg_scores,
            mean_performance_loss=(
                float(np.mean(perf_losses) * 100) if perf_losses else float("nan")
            ),
            performance_losses=perf_losses,
            mean_top_1_accuracy=float(np.mean(top_1_hits) * 100),
            mean_top_3_accuracy=float(np.mean(top_3_hits) * 100),
            per_fold=per_fold_df,
            k=self.k,
            ir_metrics=ir_metrics,
            operational_metrics=operational_metrics,
            conformal_metrics=conformal_metrics,
        )

        if self.include_conformal:
            result.conformal_mode = self.conformal_mode
            result.conformal_alpha = self.conformal_alpha
            result.conformal_near_best_rules = list(self.conformal_near_best_rules)
            result.near_best_epsilon = self.near_best_epsilon
            result.near_best_std_multiplier = self.near_best_std_multiplier
            result.conformal_max_set_sizes = [
                int(cap) for cap in self.conformal_caps if cap is not None
            ]

        return result

    @staticmethod
    def compare(result_a: EvaluationResult,
                result_b: EvaluationResult) -> dict[str, float]:
        return {
            "ndcg_p_value": paired_ttest(result_a.ndcg_scores, result_b.ndcg_scores),
            "performance_loss_p_value": paired_ttest(
                result_a.performance_losses,
                result_b.performance_losses,
            ),
        }


def results_to_summary(results: list[EvaluationResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        row = {
            "method": result.method_name,
            f"ndcg@{result.k}": result.mean_ndcg,
            "performance_loss": result.mean_performance_loss,
            "top_1_accuracy": result.mean_top_1_accuracy,
            "top_3_accuracy": result.mean_top_3_accuracy,
            "n_folds": len(result.ndcg_scores),
        }

        row.update(result.ir_metrics)
        row.update(result.operational_metrics)

        if result.conformal_metrics:
            row.update({
                "conformal_mode": result.conformal_mode,
                "conformal_alpha": result.conformal_alpha,
                "conformal_near_best_rules": (
                    ",".join(result.conformal_near_best_rules)
                    if result.conformal_near_best_rules else ""
                ),
                "near_best_epsilon": result.near_best_epsilon,
                "near_best_std_multiplier": result.near_best_std_multiplier,
                "conformal_max_set_sizes": (
                    ",".join(str(x) for x in result.conformal_max_set_sizes)
                    if result.conformal_max_set_sizes else ""
                ),
            })
            row.update(result.conformal_metrics)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    sort_cols = ["performance_loss", f"ndcg@{results[0].k}"]
    return pd.DataFrame(rows).sort_values(by=sort_cols, ascending=[True, False])


def results_to_per_fold(results: list[EvaluationResult]) -> pd.DataFrame:
    frames = [result.per_fold.copy() for result in results]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def results_to_transfer_type_loss(results: list[EvaluationResult]) -> pd.DataFrame:
    """Per-method mean performance loss broken down by transfer type.

    Each fold is bucketed by (predicted_source_resource_level, target_resource_level).
    Cells are mean performance loss (in percentage units, matching summary CSVs).
    Folds whose source or target glottocode is unclassified are excluded.
    """
    grid_columns = [f"{src}->{tgt}" for src in RESOURCE_LEVELS for tgt in RESOURCE_LEVELS]
    target_columns = [f"*->{tgt}" for tgt in RESOURCE_LEVELS]
    rows = []

    for result in results:
        df = result.per_fold
        row: dict[str, object] = {"method": result.method_name}

        for src in RESOURCE_LEVELS:
            for tgt in RESOURCE_LEVELS:
                key = f"{src}->{tgt}"
                mask = (
                    (df["predicted_source_resource_level"] == src)
                    & (df["target_resource_level"] == tgt)
                )
                vals = df.loc[mask, "performance_loss"].dropna()
                row[key] = float(vals.mean() * 100) if len(vals) else float("nan")

        for tgt in RESOURCE_LEVELS:
            key = f"*->{tgt}"
            mask = df["target_resource_level"] == tgt
            vals = df.loc[mask, "performance_loss"].dropna()
            row[key] = float(vals.mean() * 100) if len(vals) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows, columns=["method", *grid_columns, *target_columns])


def pairwise_comparisons(results: list[EvaluationResult]) -> pd.DataFrame:
    rows = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            tests = TransferEvaluator.compare(a, b)
            rows.append({
                "method_a": a.method_name,
                "method_b": b.method_name,
                **tests,
            })

    return pd.DataFrame(rows)