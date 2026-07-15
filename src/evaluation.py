from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from tqdm import tqdm

from .conformal import (
    calibration_scores_from_queries,
    conformal_quantile,
    evaluate_conformal_source_set,
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


VALID_CNOTC_GROUPINGS = {"none", "resource_level"}
VALID_CNOTC_UNKNOWN_POLICIES = {"error", "separate"}


def _percent_tag(value: float) -> str:
    percent = 100.0 * float(value)
    if percent.is_integer():
        return str(int(percent))
    return str(percent).replace(".", "p").replace("-", "m")


def _minimum_finite_calibration_size(alpha: float) -> int:
    """Smallest n for which the standard split-conformal quantile is finite.

    The quantile index is ceil((n + 1) * (1 - alpha)). It is no larger
    than n exactly when n >= (1 - alpha) / alpha.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return int(math.ceil((1.0 - alpha) / alpha))


def _standard_error(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size <= 1:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


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
    shortlist_metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        out = (
            f"EvaluationResult(method={self.method_name}, "
            f"ndcg@{self.k}={self.mean_ndcg:.2f}, "
            f"perf_loss={self.mean_performance_loss:.2f}, "
            f"top1={self.mean_top_1_accuracy:.2f}, "
            f"top3={self.mean_top_3_accuracy:.2f}"
        )

        if self.shortlist_metrics:
            if "cnotc_trial_complexity" in self.shortlist_metrics:
                out += (
                    f", cnotc={self.shortlist_metrics['cnotc_trial_complexity']:.2f}"
                )
            if "cnotc_near_oracle_coverage" in self.shortlist_metrics:
                out += (
                    f", cnotc_cov={self.shortlist_metrics['cnotc_near_oracle_coverage']:.2f}"
                )

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
                 include_cnotc: bool = False,
                 cnotc_alpha: float = 0.1,
                 cnotc_epsilon: float = 0.05,
                 cnotc_cal_size: float = 0.2,
                 cnotc_grouping: str = "none",
                 cnotc_unknown_policy: str = "error",
                 budget_ks: Sequence[int] = (10,),
                 include_ir_metrics: bool = False,
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

        if cnotc_grouping not in VALID_CNOTC_GROUPINGS:
            raise ValueError(
                f"Unknown CNOTC grouping {cnotc_grouping!r}; "
                f"expected one of {sorted(VALID_CNOTC_GROUPINGS)}."
            )
        if cnotc_unknown_policy not in VALID_CNOTC_UNKNOWN_POLICIES:
            raise ValueError(
                f"Unknown CNOTC unknown policy {cnotc_unknown_policy!r}; "
                f"expected one of {sorted(VALID_CNOTC_UNKNOWN_POLICIES)}."
            )
        if not 0 < cnotc_cal_size < 1:
            raise ValueError("cnotc_cal_size must be in (0, 1)")

        self.include_cnotc = include_cnotc
        self.cnotc_alpha = cnotc_alpha
        self.cnotc_epsilon = cnotc_epsilon
        self.cnotc_cal_size = cnotc_cal_size
        self.cnotc_grouping = cnotc_grouping
        self.cnotc_unknown_policy = cnotc_unknown_policy
        self.budget_ks = tuple(int(x) for x in budget_ks)

        self.include_ir_metrics = include_ir_metrics
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

        out["_target_resource_level"] = out[self.target_col].map(resource_level)

        if (
            self.include_cnotc
            and self.cnotc_grouping == "resource_level"
            and self.cnotc_unknown_policy == "error"
        ):
            unknown_targets = sorted(
                out.loc[
                    out["_target_resource_level"] == "unknown",
                    self.target_col,
                ].astype(str).unique().tolist()
            )
            if unknown_targets:
                preview = ", ".join(unknown_targets[:20])
                suffix = "" if len(unknown_targets) <= 20 else ", ..."
                raise ValueError(
                    "Mondrian CTC requires every target to have a resource-level "
                    f"group. Unclassified targets: {preview}{suffix}. Either update "
                    "src/resource_level_langs.py or pass "
                    "--cnotc_unknown_policy separate."
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

    def _split_fitting_cnotc(
                             self,
                             train_val_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split non-test queries into ranker-fitting and CTC-calibration sets.

        With resource-level Mondrian CTC, the split is performed separately
        within HRL, MRL and LRL targets. This preserves calibration examples
        from each available group while keeping every calibration target out of
        ranker fitting. Groups containing only one non-test query remain in the
        fitting set because they cannot be split without emptying one side.
        """
        groups = train_val_data["_query_id"].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 3:
            raise ValueError(
                "CNOTC evaluation requires at least three non-held-out queries."
            )

        if self.cnotc_grouping == "none":
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.cnotc_cal_size,
                random_state=self.random_state,
            )
            fitting_idx, calibration_idx = next(
                splitter.split(train_val_data, groups=groups)
            )
            fitting_data = train_val_data.iloc[fitting_idx].copy()
            calibration_data = train_val_data.iloc[calibration_idx].copy()
        else:
            query_groups = (
                train_val_data[["_query_id", "_target_resource_level"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            group_counts = query_groups.groupby("_query_id")[
                "_target_resource_level"
            ].nunique()
            if bool((group_counts > 1).any()):
                raise ValueError(
                    "A query was assigned to more than one resource-level group."
                )

            rng = np.random.RandomState(self.random_state)
            calibration_ids: list[str] = []

            levels = list(RESOURCE_LEVELS)
            if self.cnotc_unknown_policy == "separate":
                levels.append("unknown")

            for level in levels:
                ids = np.asarray(
                    sorted(
                        query_groups.loc[
                            query_groups["_target_resource_level"] == level,
                            "_query_id",
                        ].astype(str).tolist()
                    ),
                    dtype=object,
                )
                n_group = int(ids.size)

                if n_group < 2:
                    continue

                n_calibration = int(math.ceil(self.cnotc_cal_size * n_group))
                n_calibration = max(1, min(n_calibration, n_group - 1))
                chosen = rng.permutation(ids)[:n_calibration]
                calibration_ids.extend(chosen.tolist())

            calibration_mask = train_val_data["_query_id"].astype(str).isin(
                set(calibration_ids)
            )
            fitting_data = train_val_data.loc[~calibration_mask].copy()
            calibration_data = train_val_data.loc[calibration_mask].copy()

        if fitting_data["_query_id"].nunique() == 0:
            raise ValueError("CNOTC split produced an empty fitting set.")

        if self.cnotc_grouping == "none" and calibration_data["_query_id"].nunique() == 0:
            raise ValueError("CNOTC split produced an empty calibration set.")

        overlap = set(fitting_data["_query_id"].unique()) & set(
            calibration_data["_query_id"].unique()
        )
        if overlap:
            raise RuntimeError(
                "CNOTC fitting/calibration leakage detected for queries: "
                + ", ".join(sorted(str(x) for x in overlap))
            )

        return fitting_data, calibration_data

    def _cnotc_calibration_subset(
            self,
            calibration_data: pd.DataFrame,
            test_data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        if self.cnotc_grouping == "none":
            return "all", calibration_data

        test_groups = test_data["_target_resource_level"].astype(str).unique()
        if test_groups.size != 1:
            raise ValueError(
                "Each held-out query must belong to exactly one Mondrian group."
            )

        test_group = str(test_groups[0])
        if test_group == "unknown" and self.cnotc_unknown_policy == "error":
            raise ValueError(
                "The held-out target has unknown resource level under Mondrian CTC."
            )

        subset = calibration_data.loc[
            calibration_data["_target_resource_level"].astype(str) == test_group
        ].copy()
        return test_group, subset

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

    def _evaluate_cnotc_and_budgets(
                                    self,
                                    *,
                                    fold_ranker: BaseRanker,
                                    calibration_data: pd.DataFrame,
                                    test_data: pd.DataFrame,
                                    feature_cols: list[str],
                                    y_pred: np.ndarray) -> tuple[dict[str, object], dict[str, float]]:
        per_fold: dict[str, object] = {}
        summary_values: dict[str, float] = {}

        cnotc_group, group_calibration_data = self._cnotc_calibration_subset(
            calibration_data,
            test_data,
        )

        calibration_scores = calibration_scores_from_queries(
            group_calibration_data,
            ranker=fold_ranker,
            feature_cols=feature_cols,
            performance_col=self.performance_col,
            query_col="_query_id",
            mode="rank_near_best",
            near_best_rule="relative",
            near_best_epsilon=self.cnotc_epsilon,
            near_best_std_multiplier=1.0,
        ) if not group_calibration_data.empty else []

        n_calibration = len(calibration_scores)
        min_finite_n = _minimum_finite_calibration_size(self.cnotc_alpha)

        # No same-group calibration target implies the only distribution-free
        # valid deterministic action is the full source pool. The existing
        # conformal_source_set implementation maps +inf to that full pool.
        if n_calibration == 0:
            threshold = float("inf")
        else:
            threshold = conformal_quantile(
                calibration_scores,
                alpha=self.cnotc_alpha,
            )

        test_perf = test_data[self.performance_col].to_numpy(dtype=float)
        test_sources = test_data[self.source_col].to_numpy()

        cnotc = evaluate_conformal_source_set(
            scores=y_pred,
            performances=test_perf,
            sources=test_sources,
            threshold=threshold,
            mode="rank_near_best",
            near_best_rule="relative",
            near_best_epsilon=self.cnotc_epsilon,
            near_best_std_multiplier=1.0,
            max_set_size=None,
        )

        if calibration_data.empty:
            calibration_group_counts = ""
        else:
            counts = (
                calibration_data[["_query_id", "_target_resource_level"]]
                .drop_duplicates()
                .groupby("_target_resource_level")["_query_id"]
                .nunique()
                .to_dict()
            )
            calibration_group_counts = ",".join(
                f"{level}:{int(counts.get(level, 0))}"
                for level in (*RESOURCE_LEVELS, "unknown")
                if int(counts.get(level, 0)) > 0
            )

        per_fold.update({
            "cnotc_alpha": self.cnotc_alpha,
            "cnotc_epsilon": self.cnotc_epsilon,
            "cnotc_grouping": self.cnotc_grouping,
            "cnotc_group": cnotc_group,
            "cnotc_calibration_size": n_calibration,
            "cnotc_calibration_group_counts": calibration_group_counts,
            "cnotc_min_finite_calibration_size": min_finite_n,
            "cnotc_finite_quantile_possible": float(n_calibration >= min_finite_n),
            "cnotc_threshold_is_infinite": float(np.isinf(threshold)),
            "cnotc_threshold": cnotc.threshold,
            "cnotc_trial_complexity": cnotc.effective_k,
            "cnotc_set_size": cnotc.set_size,
            "cnotc_test_pool_size": int(test_sources.shape[0]),
            "cnotc_pool_fraction": cnotc.set_size_fraction,
            "cnotc_near_oracle_coverage": cnotc.contains_near_best_source,
            "cnotc_exact_best_coverage": cnotc.contains_best_source,
            "cnotc_best_in_set_performance": cnotc.best_in_set_performance,
            "cnotc_best_in_set_performance_loss": cnotc.best_in_set_performance_loss,
            "cnotc_sources": serialize_sources(cnotc.set_sources),
        })

        summary_values.update({
            "cnotc_trial_complexity": float(cnotc.set_size),
            "cnotc_pool_fraction": cnotc.set_size_fraction,
            "cnotc_near_oracle_coverage": cnotc.contains_near_best_source,
            "cnotc_exact_best_coverage": cnotc.contains_best_source,
            "cnotc_best_in_set_performance_loss": cnotc.best_in_set_performance_loss,
            "cnotc_calibration_size": float(n_calibration),
            "cnotc_finite_quantile_possible": float(n_calibration >= min_finite_n),
            "cnotc_threshold_is_infinite": float(np.isinf(threshold)),
        })

        epsilon_tag = _percent_tag(self.cnotc_epsilon)

        for budget_k in self.budget_ks:
            budget_k = int(budget_k)
            budget = evaluate_conformal_source_set(
                scores=y_pred,
                performances=test_perf,
                sources=test_sources,
                threshold=float(budget_k),
                mode="rank_near_best",
                near_best_rule="relative",
                near_best_epsilon=self.cnotc_epsilon,
                near_best_std_multiplier=1.0,
                max_set_size=budget_k,
            )

            prefix = f"budget_{budget_k}_at_{epsilon_tag}"

            per_fold.update({
                f"{prefix}_size": budget.set_size,
                f"{prefix}_pool_fraction": budget.set_size_fraction,
                f"{prefix}_near_oracle_coverage": budget.contains_near_best_source,
                f"{prefix}_exact_best_coverage": budget.contains_best_source,
                f"{prefix}_best_in_set_performance": budget.best_in_set_performance,
                f"{prefix}_best_in_set_performance_loss": (
                    budget.best_in_set_performance_loss
                ),
                f"{prefix}_sources": serialize_sources(budget.set_sources),
            })

            summary_values.update({
                f"{prefix}_size": float(budget.set_size),
                f"{prefix}_pool_fraction": budget.set_size_fraction,
                f"{prefix}_near_oracle_coverage": budget.contains_near_best_source,
                f"{prefix}_exact_best_coverage": budget.contains_best_source,
                f"{prefix}_best_in_set_performance_loss": (
                    budget.best_in_set_performance_loss
                ),
            })

        return per_fold, summary_values

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
        shortlist_metric_lists: dict[str, list[float]] = {}

        iterator = tqdm(splits, desc="LOO-CV", disable=not self.verbose)

        for train_val_idx, test_idx in iterator:
            train_val_data = work_df.iloc[train_val_idx].copy()
            test_data = work_df.iloc[test_idx].copy()

            if self.include_cnotc:
                fitting_pool_data, cnotc_calibration_data = self._split_fitting_cnotc(
                    train_val_data
                )
            else:
                fitting_pool_data = train_val_data
                cnotc_calibration_data = None

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

            X_test = test_data[feature_cols].to_numpy(dtype=float)
            y_test = test_data["_relevance"].to_numpy(dtype=float)
            y_pred = fold_ranker.predict(X_test)

            fold_ndcg = ndcg_at_k(y_test, y_pred, k=self.k)
            ndcg_scores.append(fold_ndcg)

            if self.include_ir_metrics:
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
            else:
                fold_ir = {}

            test_perf = test_data[self.performance_col].to_numpy(dtype=float)

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
                "target_resource_level": test_data["_target_resource_level"].iloc[0],
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

            if self.include_cnotc:
                cnotc_per_fold, cnotc_summary_values = self._evaluate_cnotc_and_budgets(
                    fold_ranker=fold_ranker,
                    calibration_data=cnotc_calibration_data,
                    test_data=test_data,
                    feature_cols=feature_cols,
                    y_pred=y_pred,
                )

                record.update(cnotc_per_fold)

                for key, value in cnotc_summary_values.items():
                    if key not in shortlist_metric_lists:
                        shortlist_metric_lists[key] = []
                    if value is not None and not np.isnan(value):
                        shortlist_metric_lists[key].append(float(value))

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

        shortlist_metrics = {}
        for key, values in shortlist_metric_lists.items():
            if not values:
                shortlist_metrics[key] = float("nan")
            elif key.endswith("_coverage"):
                shortlist_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_pool_fraction"):
                shortlist_metrics[key] = float(np.mean(values) * 100)
            elif key.endswith("_performance_loss"):
                shortlist_metrics[key] = float(np.mean(values) * 100)
            else:
                shortlist_metrics[key] = float(np.mean(values))

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
            shortlist_metrics=shortlist_metrics,
        )

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
        row.update(result.shortlist_metrics)

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



def results_to_mondrian_ctc(results: list[EvaluationResult]) -> pd.DataFrame:
    """Summarise resource-level Mondrian CTC and hit rate.

    One row is returned for each method and target resource level. CTC is the
    mean realised top-k set size across held-out targets in that group. Hit-rate
    standard errors are the usual empirical standard errors across held-out
    target queries, matching the paper's ``mean +/- SE`` presentation.
    """
    rows: list[dict[str, object]] = []

    required = {
        "target_resource_level",
        "cnotc_trial_complexity",
        "cnotc_near_oracle_coverage",
        "cnotc_exact_best_coverage",
        "cnotc_pool_fraction",
        "cnotc_calibration_size",
        "cnotc_threshold_is_infinite",
        "cnotc_finite_quantile_possible",
    }

    for result in results:
        df = result.per_fold.copy()
        if not required.issubset(df.columns):
            continue

        for level in RESOURCE_LEVELS:
            group_df = df.loc[df["target_resource_level"] == level].copy()
            if group_df.empty:
                continue

            trial_complexity = pd.to_numeric(
                group_df["cnotc_trial_complexity"], errors="coerce"
            )
            near_hits = pd.to_numeric(
                group_df["cnotc_near_oracle_coverage"], errors="coerce"
            )
            exact_hits = pd.to_numeric(
                group_df["cnotc_exact_best_coverage"], errors="coerce"
            )
            calibration_sizes = pd.to_numeric(
                group_df["cnotc_calibration_size"], errors="coerce"
            )
            pool_sizes = pd.to_numeric(
                group_df.get("cnotc_test_pool_size"), errors="coerce"
            )
            pool_fractions = pd.to_numeric(
                group_df["cnotc_pool_fraction"], errors="coerce"
            )
            infinite = pd.to_numeric(
                group_df["cnotc_threshold_is_infinite"], errors="coerce"
            )
            finite_possible = pd.to_numeric(
                group_df["cnotc_finite_quantile_possible"], errors="coerce"
            )

            best_loss_col = "cnotc_best_in_set_performance_loss"
            if best_loss_col in group_df.columns:
                best_loss = pd.to_numeric(
                    group_df[best_loss_col], errors="coerce"
                )
            else:
                best_loss = pd.Series(dtype=float)

            rows.append({
                "method": result.method_name,
                "resource_level": level,
                "n_folds": int(group_df.shape[0]),
                "mean_trial_complexity": float(trial_complexity.mean()),
                "median_trial_complexity": float(trial_complexity.median()),
                "min_trial_complexity": float(trial_complexity.min()),
                "max_trial_complexity": float(trial_complexity.max()),
                "mean_test_pool_size": float(pool_sizes.mean()),
                "mean_pool_fraction_pct": float(pool_fractions.mean() * 100.0),
                "near_oracle_hit_rate_pct": float(near_hits.mean() * 100.0),
                "near_oracle_hit_rate_se_pct": float(_standard_error(near_hits) * 100.0),
                "exact_best_hit_rate_pct": float(exact_hits.mean() * 100.0),
                "exact_best_hit_rate_se_pct": float(_standard_error(exact_hits) * 100.0),
                "mean_calibration_size": float(calibration_sizes.mean()),
                "min_calibration_size": float(calibration_sizes.min()),
                "max_calibration_size": float(calibration_sizes.max()),
                "minimum_finite_calibration_size": int(
                    group_df["cnotc_min_finite_calibration_size"].iloc[0]
                ) if "cnotc_min_finite_calibration_size" in group_df.columns else float("nan"),
                "finite_quantile_possible_rate_pct": float(
                    finite_possible.mean() * 100.0
                ),
                "infinite_threshold_rate_pct": float(infinite.mean() * 100.0),
                "mean_best_in_set_performance_loss_pct": float(
                    best_loss.mean() * 100.0
                ) if not best_loss.empty else float("nan"),
                "alpha": float(group_df["cnotc_alpha"].iloc[0]),
                "epsilon": float(group_df["cnotc_epsilon"].iloc[0]),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["resource_level"] = pd.Categorical(
        out["resource_level"],
        categories=list(RESOURCE_LEVELS),
        ordered=True,
    )
    return out.sort_values(["method", "resource_level"]).reset_index(drop=True)

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