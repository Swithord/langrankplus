from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# Make the repository root importable when this file is run directly as
#     python scripts/evaluate_cross_task.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_methods import build_methods
from src.data import load_transfer_data, normalize_query_features
from src.evaluation import (
    EvaluationResult,
    TransferEvaluator,
    pairwise_comparisons,
    results_to_per_fold,
    results_to_summary,
    results_to_transfer_type_loss,
)
from src.metrics import (
    compute_ir_metrics,
    ndcg_at_k,
    performance_loss,
    top_k_accuracy,
)
from src.resource_level_langs import resource_level


STANDARD_PERFORMANCE_COL = "_cross_task_performance"
VALIDATION_UNITS = {"task", "query"}


def _canonical_language_ids(
    values: pd.Series,
    *,
    column_name: str,
    task_name: Optional[str] = None,
) -> pd.Series:
    """Return trimmed, case-insensitive language identifiers."""
    canonical = values.astype("string").str.strip()
    invalid = canonical.isna() | canonical.eq("")

    if bool(invalid.any()):
        bad_rows = values.index[invalid].tolist()[:10]
        location = f" in task {task_name!r}" if task_name is not None else ""
        raise ValueError(
            f"Column {column_name!r}{location} contains missing or empty "
            f"language identifiers; example row indices: {bad_rows}."
        )

    return canonical.str.casefold()


def _self_transfer_mask(
    df: pd.DataFrame,
    *,
    target_col: str,
    source_col: str,
    task_name: Optional[str] = None,
) -> pd.Series:
    """Identify rows whose source language is the target language."""
    target_ids = _canonical_language_ids(
        df[target_col],
        column_name=target_col,
        task_name=task_name,
    )
    source_ids = _canonical_language_ids(
        df[source_col],
        column_name=source_col,
        task_name=task_name,
    )
    return target_ids.eq(source_ids)


def _assert_no_self_transfer(
    df: pd.DataFrame,
    *,
    target_col: str,
    source_col: str,
    context: str,
) -> None:
    """Fail immediately if any same-language candidate remains."""
    mask = _self_transfer_mask(
        df,
        target_col=target_col,
        source_col=source_col,
    )
    if not bool(mask.any()):
        return

    examples = (
        df.loc[mask, [target_col, source_col]]
        .drop_duplicates()
        .head(10)
        .to_dict("records")
    )
    raise RuntimeError(
        f"Self-transfer rows remain in {context}. "
        f"Example source-target pairs: {examples}"
    )


def _remove_self_transfer_rows(
    frame: pd.DataFrame,
    *,
    task_name: str,
    target_col: str,
    source_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove source == target candidates before any preprocessing."""
    target_ids = _canonical_language_ids(
        frame[target_col],
        column_name=target_col,
        task_name=task_name,
    )
    mask = _self_transfer_mask(
        frame,
        target_col=target_col,
        source_col=source_col,
        task_name=task_name,
    )

    original_rows = int(frame.shape[0])
    original_targets = set(target_ids.tolist())
    removed_rows = int(mask.sum())
    affected_targets = int(target_ids.loc[mask].nunique())

    filtered = frame.loc[~mask].copy()

    remaining_target_ids = set(
        _canonical_language_ids(
            filtered[target_col],
            column_name=target_col,
            task_name=task_name,
        ).tolist()
    )
    targets_without_candidates = sorted(original_targets - remaining_target_ids)
    if targets_without_candidates:
        preview = ", ".join(targets_without_candidates[:20])
        suffix = "" if len(targets_without_candidates) <= 20 else ", ..."
        raise ValueError(
            f"After removing self-transfer from task {task_name!r}, "
            "the following targets have no cross-lingual source candidates: "
            f"{preview}{suffix}"
        )

    _assert_no_self_transfer(
        filtered,
        target_col=target_col,
        source_col=source_col,
        context=f"filtered task {task_name!r}",
    )

    stats = {
        "original_rows": original_rows,
        "removed_self_transfer_rows": removed_rows,
        "affected_targets": affected_targets,
        "remaining_rows": int(filtered.shape[0]),
        "remaining_targets": int(filtered[target_col].nunique()),
    }

    print(
        f"Self-transfer filter [{task_name}]: removed {removed_rows} rows "
        f"across {affected_targets} targets; "
        f"{filtered.shape[0]} rows remain."
    )

    return filtered, stats



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-language rankers under leave-one-task-out "
            "cross-task generalisation for one multilingual model."
        ),
    )

    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One transfer-performance CSV per task for a single model.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=None,
        help=(
            "Task names aligned with --csv. If omitted, CSV file stems are used."
        ),
    )
    parser.add_argument(
        "--performance_cols",
        nargs="+",
        required=True,
        help=(
            "Task-specific performance columns aligned with --csv, for example "
            "f1_score bleu las. All are renamed internally to one common column."
        ),
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model label written to every output row, for example mt5 or xlm-r.",
    )
    parser.add_argument(
        "--heldout_tasks",
        nargs="*",
        default=None,
        help=(
            "Optional subset of tasks to hold out. By default every supplied task "
            "is held out once."
        ),
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=["new_gen", "new_typ", "new_geo", "script"],
    )
    parser.add_argument("--target_col", default="task_lang")
    parser.add_argument("--source_col", default="transfer_lang")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument(
        "--normalizer",
        default="minmax",
        choices=["none", "minmax"],
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top_k_relevance", type=int, default=10)
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Fraction of non-test tasks or queries used for validation.",
    )
    parser.add_argument(
        "--validation_unit",
        choices=sorted(VALIDATION_UNITS),
        default="task",
        help=(
            "Use complete training tasks for validation ('task', recommended for "
            "cross-task generalisation) or split target-language queries ('query')."
        ),
    )
    parser.add_argument("--random_state", type=int, default=42)

    # These mirror scripts/evaluate_methods.py so that its build_methods()
    # function can be reused without duplicating ranker construction.
    parser.add_argument("--skip_random", action="store_true")
    parser.add_argument("--skip_single", action="store_true")
    parser.add_argument("--skip_learned", action="store_true")
    parser.add_argument(
        "--include_nested_offline",
        action="store_true",
        help=(
            "Also fit Composite-Pairwise and RRF-Pairwise inside each held-out-task "
            "fold using only the non-test tasks."
        ),
    )
    parser.add_argument("--n_opt_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_pairs_per_query", type=int, default=5000)
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument(
        "--rrf_k_grid",
        nargs="+",
        type=float,
        default=[1, 5, 10, 20, 40, 60, 100],
    )

    parser.add_argument("--include_ir_metrics", action="store_true")
    parser.add_argument(
        "--ir_cutoffs",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/cross_task_generalization",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Frozen calibration files cannot generally be used safely here because a
    # single frozen file may have been trained on the task currently held out.
    # build_methods() expects these attributes, so they are fixed to None.
    args.composite_calibration_json = None
    args.rrf_calibration_json = None

    return args


def _validate_cli_lists(args: argparse.Namespace) -> list[str]:
    n_csv = len(args.csv)

    if len(args.performance_cols) != n_csv:
        raise ValueError(
            "--performance_cols must contain exactly one entry per --csv path: "
            f"received {len(args.performance_cols)} columns for {n_csv} CSVs."
        )

    if args.dataset_names is None:
        dataset_names = [Path(path).stem for path in args.csv]
    else:
        dataset_names = list(args.dataset_names)
        if len(dataset_names) != n_csv:
            raise ValueError(
                "--dataset_names must contain exactly one entry per --csv path: "
                f"received {len(dataset_names)} names for {n_csv} CSVs."
            )

    if len(set(dataset_names)) != len(dataset_names):
        duplicates = sorted(
            name for name in set(dataset_names) if dataset_names.count(name) > 1
        )
        raise ValueError(
            "Task names must be unique. Duplicate names: " + ", ".join(duplicates)
        )

    if not 0.0 <= args.val_size < 1.0:
        raise ValueError("--val_size must be in [0, 1).")

    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    if args.top_k_relevance < 1:
        raise ValueError("--top_k_relevance must be at least 1.")

    return dataset_names


def load_cross_task_data(
    csv_paths: Sequence[str],
    dataset_names: Sequence[str],
    performance_cols: Sequence[str],
    *,
    dataset_col: str,
    feature_cols: Sequence[str],
    target_col: str,
    source_col: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, int]]]:
    """Load heterogeneous task CSVs and standardise their performance column.

    The existing load_transfer_data() function is deliberately reused one file
    at a time. This preserves its file-name handling and unnamed-index cleanup,
    while allowing each task to supply a different performance column.
    """
    frames: list[pd.DataFrame] = []
    performance_map: dict[str, str] = {}
    self_transfer_stats: dict[str, dict[str, int]] = {}

    required_common = {target_col, source_col, *feature_cols}

    for csv_path, task_name, performance_col in zip(
        csv_paths,
        dataset_names,
        performance_cols,
    ):
        frame = load_transfer_data(
            [csv_path],
            dataset_names=[task_name],
            dataset_col=dataset_col,
        )

        missing = sorted(required_common - set(frame.columns))
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing)}"
            )

        if performance_col not in frame.columns:
            raise ValueError(
                f"{csv_path} does not contain performance column "
                f"{performance_col!r}. Available columns: "
                + ", ".join(str(col) for col in frame.columns)
            )

        # Enforce a genuinely cross-lingual candidate pool before
        # normalisation, relevance construction, splitting, fitting, and
        # oracle evaluation.
        frame, filter_stats = _remove_self_transfer_rows(
            frame,
            task_name=task_name,
            target_col=target_col,
            source_col=source_col,
        )
        self_transfer_stats[task_name] = filter_stats

        performance = pd.to_numeric(frame[performance_col], errors="coerce")
        bad_mask = performance.isna()
        if bool(bad_mask.any()):
            bad_rows = frame.index[bad_mask].tolist()[:10]
            raise ValueError(
                f"{csv_path} contains non-numeric or missing values in "
                f"{performance_col!r}; example row indices: {bad_rows}."
            )

        frame[STANDARD_PERFORMANCE_COL] = performance.astype(float)
        frame["_original_performance_col"] = performance_col

        frames.append(frame)
        performance_map[task_name] = performance_col

    if not frames:
        raise ValueError("At least one CSV is required.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    _assert_no_self_transfer(
        combined,
        target_col=target_col,
        source_col=source_col,
        context="combined cross-task data",
    )

    if combined[dataset_col].nunique() < 2:
        raise ValueError(
            "Leave-one-task-out evaluation requires at least two distinct tasks."
        )

    return combined, performance_map, self_transfer_stats


def split_train_validation(
    evaluator: TransferEvaluator,
    fitting_pool: pd.DataFrame,
    *,
    dataset_col: str,
    validation_unit: str,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Split only the non-test data into fitting and validation partitions."""
    if val_size <= 0:
        return fitting_pool.copy(), None

    if validation_unit == "query":
        # Reuse the existing leakage-safe query-group split.
        return evaluator._split_train_val(fitting_pool)

    if validation_unit != "task":
        raise ValueError(
            f"Unknown validation unit {validation_unit!r}; "
            f"expected one of {sorted(VALIDATION_UNITS)}."
        )

    unique_tasks = fitting_pool[dataset_col].astype(str).nunique()
    if unique_tasks < 2:
        return fitting_pool.copy(), None

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )
    groups = fitting_pool[dataset_col].astype(str).to_numpy()
    train_idx, val_idx = next(splitter.split(fitting_pool, groups=groups))

    train_data = fitting_pool.iloc[train_idx].copy()
    val_data = fitting_pool.iloc[val_idx].copy()

    train_tasks = set(train_data[dataset_col].astype(str).unique())
    val_tasks = set(val_data[dataset_col].astype(str).unique())
    overlap = train_tasks & val_tasks
    if overlap:
        raise RuntimeError(
            "Task-level train/validation leakage detected for tasks: "
            + ", ".join(sorted(overlap))
        )

    if not train_tasks or not val_tasks:
        return fitting_pool.copy(), None

    return train_data, val_data


def _ir_metric_names(cutoffs: Iterable[int]) -> list[str]:
    names = ["mrr", "exact_best_rank", "r_precision", "relevant_count"]
    for cutoff in sorted(set(int(value) for value in cutoffs)):
        names.extend(
            [
                f"ndcg@{cutoff}",
                f"precision@{cutoff}",
                f"recall@{cutoff}",
                f"hit@{cutoff}",
                f"map@{cutoff}",
                f"err@{cutoff}",
                f"exact_best_hit@{cutoff}",
            ]
        )
    return names


def evaluate_held_out_task(
    *,
    ranker,
    method_name: str,
    model_name: str,
    held_out_task: str,
    performance_col_name: str,
    test_data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    source_col: str,
    dataset_col: str,
    k: int,
    include_ir_metrics: bool,
    ir_cutoffs: Sequence[int],
    n_fitting_tasks: int,
    n_validation_tasks: int,
    n_fitting_queries: int,
    n_validation_queries: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    _assert_no_self_transfer(
        test_data,
        target_col=target_col,
        source_col=source_col,
        context=f"held-out task {held_out_task!r}",
    )

    for query_id, query_df in test_data.groupby("_query_id", sort=False):
        X_test = query_df[list(feature_cols)].to_numpy(dtype=float)
        y_true = query_df["_relevance"].to_numpy(dtype=float)
        y_pred = np.asarray(ranker.predict(X_test), dtype=float)

        if y_pred.ndim != 1 or y_pred.shape[0] != query_df.shape[0]:
            raise ValueError(
                f"Method {method_name!r} returned predictions with shape "
                f"{y_pred.shape} for query {query_id!r}, expected "
                f"({query_df.shape[0]},)."
            )
        if not np.all(np.isfinite(y_pred)):
            raise ValueError(
                f"Method {method_name!r} returned non-finite predictions for "
                f"query {query_id!r}."
            )

        performances = query_df[STANDARD_PERFORMANCE_COL].to_numpy(dtype=float)
        predicted_best_idx = int(np.argmax(y_pred))
        actual_best_idx = int(np.argmax(performances))

        predicted_performance = float(performances[predicted_best_idx])
        actual_best_performance = float(performances[actual_best_idx])
        loss = performance_loss(
            predicted_performance,
            actual_best_performance,
        )

        target_language = query_df[target_col].iloc[0]
        predicted_source = query_df[source_col].iloc[predicted_best_idx]
        actual_best_source = query_df[source_col].iloc[actual_best_idx]

        record: dict[str, object] = {
            "model": model_name,
            "method": method_name,
            "held_out_task": held_out_task,
            "dataset": query_df[dataset_col].iloc[0],
            "performance_col": performance_col_name,
            "query_id": query_id,
            "target_lang": target_language,
            "target_resource_level": resource_level(target_language),
            "predicted_best_source": predicted_source,
            "predicted_source_resource_level": resource_level(predicted_source),
            "actual_best_source": actual_best_source,
            "actual_best_source_resource_level": resource_level(actual_best_source),
            "predicted_performance": predicted_performance,
            "actual_best_performance": actual_best_performance,
            "performance_loss": loss,
            "ndcg": ndcg_at_k(y_true, y_pred, k=k),
            "top_1_accuracy": top_k_accuracy(y_true, y_pred, k=1),
            "top_3_accuracy": top_k_accuracy(y_true, y_pred, k=3),
            "n_candidates": int(query_df.shape[0]),
            "n_fitting_tasks": int(n_fitting_tasks),
            "n_validation_tasks": int(n_validation_tasks),
            "n_fitting_queries": int(n_fitting_queries),
            "n_validation_queries": int(n_validation_queries),
        }

        if include_ir_metrics:
            record.update(
                compute_ir_metrics(
                    y_true,
                    y_pred,
                    cutoffs=ir_cutoffs,
                )
            )

        records.append(record)

    return records


def result_from_query_records(
    method_name: str,
    method_df: pd.DataFrame,
    *,
    k: int,
    include_ir_metrics: bool,
    ir_cutoffs: Sequence[int],
) -> EvaluationResult:
    method_df = method_df.sort_values(
        ["held_out_task", "query_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    ndcg_scores = pd.to_numeric(method_df["ndcg"], errors="coerce").tolist()
    performance_losses = (
        pd.to_numeric(method_df["performance_loss"], errors="coerce")
        .dropna()
        .tolist()
    )
    top_1 = pd.to_numeric(method_df["top_1_accuracy"], errors="coerce")
    top_3 = pd.to_numeric(method_df["top_3_accuracy"], errors="coerce")

    ir_metrics: dict[str, float] = {}
    if include_ir_metrics:
        for metric in _ir_metric_names(ir_cutoffs):
            if metric not in method_df.columns:
                continue
            values = pd.to_numeric(method_df[metric], errors="coerce").dropna()
            if values.empty:
                ir_metrics[metric] = float("nan")
            elif metric in {"exact_best_rank", "relevant_count"}:
                ir_metrics[metric] = float(values.mean())
            else:
                ir_metrics[metric] = float(values.mean() * 100.0)

    return EvaluationResult(
        method_name=method_name,
        mean_ndcg=float(np.nanmean(ndcg_scores) * 100.0),
        ndcg_scores=[float(value) for value in ndcg_scores],
        mean_performance_loss=(
            float(np.mean(performance_losses) * 100.0)
            if performance_losses
            else float("nan")
        ),
        performance_losses=[float(value) for value in performance_losses],
        mean_top_1_accuracy=float(top_1.mean() * 100.0),
        mean_top_3_accuracy=float(top_3.mean() * 100.0),
        per_fold=method_df.copy(),
        k=k,
        ir_metrics=ir_metrics,
        shortlist_metrics={},
    )


def summarize_by_held_out_task(
    per_query: pd.DataFrame,
    *,
    k: int,
    include_ir_metrics: bool,
    ir_cutoffs: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["model", "method", "held_out_task", "performance_col"]

    for group_key, group_df in per_query.groupby(group_cols, sort=False):
        model_name, method_name, held_out_task, performance_col = group_key

        row: dict[str, object] = {
            "model": model_name,
            "method": method_name,
            "held_out_task": held_out_task,
            "performance_col": performance_col,
            "n_queries": int(group_df.shape[0]),
            "mean_candidates": float(group_df["n_candidates"].mean()),
            "n_fitting_tasks": int(group_df["n_fitting_tasks"].iloc[0]),
            "n_validation_tasks": int(group_df["n_validation_tasks"].iloc[0]),
            "n_fitting_queries": int(group_df["n_fitting_queries"].iloc[0]),
            "n_validation_queries": int(group_df["n_validation_queries"].iloc[0]),
            f"ndcg@{k}": float(group_df["ndcg"].mean() * 100.0),
            "performance_loss": float(
                pd.to_numeric(
                    group_df["performance_loss"],
                    errors="coerce",
                ).mean()
                * 100.0
            ),
            "top_1_accuracy": float(
                group_df["top_1_accuracy"].mean() * 100.0
            ),
            "top_3_accuracy": float(
                group_df["top_3_accuracy"].mean() * 100.0
            ),
        }

        if include_ir_metrics:
            for metric in _ir_metric_names(ir_cutoffs):
                if metric not in group_df.columns:
                    continue
                values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
                if values.empty:
                    row[metric] = float("nan")
                elif metric in {"exact_best_rank", "relevant_count"}:
                    row[metric] = float(values.mean())
                else:
                    row[metric] = float(values.mean() * 100.0)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["held_out_task", "performance_loss", f"ndcg@{k}"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_macro_tasks(
    task_summary: pd.DataFrame,
    per_query: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    if task_summary.empty:
        return pd.DataFrame()

    metadata_cols = {
        "model",
        "method",
        "held_out_task",
        "performance_col",
        "n_queries",
        "mean_candidates",
        "n_fitting_tasks",
        "n_validation_tasks",
        "n_fitting_queries",
        "n_validation_queries",
    }
    metric_cols = [
        column
        for column in task_summary.columns
        if column not in metadata_cols
        and pd.api.types.is_numeric_dtype(task_summary[column])
    ]

    rows: list[dict[str, object]] = []
    for (model_name, method_name), group_df in task_summary.groupby(
        ["model", "method"],
        sort=False,
    ):
        query_count = int(
            per_query.loc[
                (per_query["model"] == model_name)
                & (per_query["method"] == method_name)
            ].shape[0]
        )

        row: dict[str, object] = {
            "model": model_name,
            "method": method_name,
            "n_tasks": int(group_df["held_out_task"].nunique()),
            "n_queries": query_count,
            "aggregation": "equal_weight_per_held_out_task",
        }
        for metric in metric_cols:
            row[metric] = float(
                pd.to_numeric(group_df[metric], errors="coerce").mean()
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["performance_loss", f"ndcg@{k}"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def macro_task_results_for_pairwise(
    task_summary: pd.DataFrame,
    *,
    method_order: Sequence[str],
    task_order: Sequence[str],
    k: int,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []

    for method_name in method_order:
        method_df = task_summary.loc[
            task_summary["method"] == method_name
        ].copy()
        method_df["held_out_task"] = pd.Categorical(
            method_df["held_out_task"],
            categories=list(task_order),
            ordered=True,
        )
        method_df = method_df.sort_values("held_out_task").reset_index(drop=True)

        observed_tasks = method_df["held_out_task"].astype(str).tolist()
        if observed_tasks != list(task_order):
            raise RuntimeError(
                f"Method {method_name!r} does not have exactly one summary row "
                "for every held-out task."
            )

        ndcg_scores = (
            pd.to_numeric(method_df[f"ndcg@{k}"], errors="coerce") / 100.0
        ).tolist()
        loss_scores = (
            pd.to_numeric(method_df["performance_loss"], errors="coerce") / 100.0
        ).tolist()
        top_1 = pd.to_numeric(
            method_df["top_1_accuracy"], errors="coerce"
        )
        top_3 = pd.to_numeric(
            method_df["top_3_accuracy"], errors="coerce"
        )

        results.append(
            EvaluationResult(
                method_name=method_name,
                mean_ndcg=float(np.nanmean(ndcg_scores) * 100.0),
                ndcg_scores=[float(value) for value in ndcg_scores],
                mean_performance_loss=float(np.nanmean(loss_scores) * 100.0),
                performance_losses=[float(value) for value in loss_scores],
                mean_top_1_accuracy=float(top_1.mean()),
                mean_top_3_accuracy=float(top_3.mean()),
                per_fold=method_df,
                k=k,
                ir_metrics={},
                shortlist_metrics={},
            )
        )

    return results


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_names: Sequence[str],
    performance_map: dict[str, str],
    self_transfer_stats: dict[str, dict[str, int]],
    heldout_tasks: Sequence[str],
    work_df: pd.DataFrame,
) -> None:
    task_counts = {
        str(task): {
            "rows": int(group_df.shape[0]),
            "queries": int(group_df["_query_id"].nunique()),
            "targets": int(group_df[args.target_col].nunique()),
            "sources": int(group_df[args.source_col].nunique()),
            "performance_col": performance_map[str(task)],
        }
        for task, group_df in work_df.groupby(args.dataset_col, sort=False)
    }

    payload = {
        "model_name": args.model_name,
        "csv": list(args.csv),
        "dataset_names": list(dataset_names),
        "performance_cols": list(args.performance_cols),
        "heldout_tasks": list(heldout_tasks),
        "features": list(args.features),
        "normalizer": args.normalizer,
        "k": args.k,
        "top_k_relevance": args.top_k_relevance,
        "val_size": args.val_size,
        "validation_unit": args.validation_unit,
        "random_state": args.random_state,
        "self_transfer_allowed": False,
        "self_transfer_filter_stats": self_transfer_stats,
        "task_counts": task_counts,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    dataset_names = _validate_cli_lists(args)

    raw_df, performance_map, self_transfer_stats = load_cross_task_data(
        args.csv,
        dataset_names,
        args.performance_cols,
        dataset_col=args.dataset_col,
        feature_cols=args.features,
        target_col=args.target_col,
        source_col=args.source_col,
    )

    normalised_df = normalize_query_features(
        raw_df,
        feature_cols=args.features,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        method=args.normalizer,
    )
    _assert_no_self_transfer(
        normalised_df,
        target_col=args.target_col,
        source_col=args.source_col,
        context="normalised cross-task data",
    )

    evaluator = TransferEvaluator(
        target_col=args.target_col,
        source_col=args.source_col,
        performance_col=STANDARD_PERFORMANCE_COL,
        dataset_col=args.dataset_col,
        k=args.k,
        top_k_relevance=args.top_k_relevance,
        val_size=args.val_size,
        random_state=args.random_state,
        verbose=args.verbose,
        include_cnotc=False,
        include_ir_metrics=args.include_ir_metrics,
        ir_cutoffs=args.ir_cutoffs,
    )

    # Reuse the existing evaluator's validation, relevance construction, query
    # IDs, and resource-level annotation.
    work_df = evaluator._prepare(normalised_df, list(args.features))
    _assert_no_self_transfer(
        work_df,
        target_col=args.target_col,
        source_col=args.source_col,
        context="prepared evaluation data",
    )

    all_tasks = dataset_names
    if args.heldout_tasks:
        unknown = sorted(set(args.heldout_tasks) - set(all_tasks))
        if unknown:
            raise ValueError(
                "Unknown --heldout_tasks values: " + ", ".join(unknown)
            )
        heldout_tasks = [
            task for task in all_tasks if task in set(args.heldout_tasks)
        ]
    else:
        heldout_tasks = list(all_tasks)

    if not heldout_tasks:
        raise ValueError("No held-out tasks were selected.")

    methods = build_methods(args)
    method_order = [method_name for method_name, _ in methods]

    all_records: list[dict[str, object]] = []
    heldout_iterator = tqdm(
        heldout_tasks,
        desc=f"Leave-one-task-out ({args.model_name})",
        disable=not args.verbose,
    )

    for fold_index, held_out_task in enumerate(heldout_iterator):
        heldout_iterator.set_postfix(task=held_out_task)

        test_data = work_df.loc[
            work_df[args.dataset_col].astype(str) == held_out_task
        ].copy()
        fitting_pool = work_df.loc[
            work_df[args.dataset_col].astype(str) != held_out_task
        ].copy()

        if test_data.empty:
            raise RuntimeError(f"Held-out task {held_out_task!r} has no rows.")
        if fitting_pool.empty:
            raise RuntimeError(
                f"Held-out task {held_out_task!r} leaves no fitting data."
            )

        train_data, val_data = split_train_validation(
            evaluator,
            fitting_pool,
            dataset_col=args.dataset_col,
            validation_unit=args.validation_unit,
            val_size=args.val_size,
            # Vary the validation partition by outer fold while remaining fully
            # deterministic across reruns.
            random_state=args.random_state + fold_index,
        )

        fitting_tasks = sorted(
            train_data[args.dataset_col].astype(str).unique().tolist()
        )
        validation_tasks = (
            sorted(val_data[args.dataset_col].astype(str).unique().tolist())
            if val_data is not None
            else []
        )
        n_fitting_queries = int(train_data["_query_id"].nunique())
        n_validation_queries = (
            int(val_data["_query_id"].nunique())
            if val_data is not None
            else 0
        )

        print("\n" + "=" * 72)
        print(f"Model: {args.model_name}")
        print(f"Held-out task: {held_out_task}")
        print(f"Fitting tasks ({len(fitting_tasks)}): {', '.join(fitting_tasks)}")
        print(
            f"Validation tasks ({len(validation_tasks)}): "
            + (", ".join(validation_tasks) if validation_tasks else "none")
        )
        print(f"Held-out queries: {test_data['_query_id'].nunique()}")

        method_iterator = tqdm(
            methods,
            desc=f"Methods: {held_out_task}",
            leave=False,
            disable=not args.verbose,
        )

        for method_name, base_ranker in method_iterator:
            method_iterator.set_postfix(method=method_name)
            print(f"Evaluating {method_name} on held-out task {held_out_task}")

            fold_ranker = clone(base_ranker)
            fold_ranker = evaluator._fit_fold_ranker(
                ranker=fold_ranker,
                train_data=train_data,
                val_data=val_data,
                feature_cols=list(args.features),
            )

            records = evaluate_held_out_task(
                ranker=fold_ranker,
                method_name=method_name,
                model_name=args.model_name,
                held_out_task=held_out_task,
                performance_col_name=performance_map[held_out_task],
                test_data=test_data,
                feature_cols=args.features,
                target_col=args.target_col,
                source_col=args.source_col,
                dataset_col=args.dataset_col,
                k=args.k,
                include_ir_metrics=args.include_ir_metrics,
                ir_cutoffs=args.ir_cutoffs,
                n_fitting_tasks=len(fitting_tasks),
                n_validation_tasks=len(validation_tasks),
                n_fitting_queries=n_fitting_queries,
                n_validation_queries=n_validation_queries,
            )
            all_records.extend(records)

    per_query = pd.DataFrame(all_records)
    if per_query.empty:
        raise RuntimeError("Cross-task evaluation produced no query records.")

    pooled_results = [
        result_from_query_records(
            method_name,
            per_query.loc[per_query["method"] == method_name].copy(),
            k=args.k,
            include_ir_metrics=args.include_ir_metrics,
            ir_cutoffs=args.ir_cutoffs,
        )
        for method_name in method_order
    ]

    pooled_summary = results_to_summary(pooled_results)
    pooled_summary.insert(0, "model", args.model_name)
    pooled_summary.insert(2, "aggregation", "equal_weight_per_target_query")

    exported_per_query = results_to_per_fold(pooled_results)
    task_summary = summarize_by_held_out_task(
        exported_per_query,
        k=args.k,
        include_ir_metrics=args.include_ir_metrics,
        ir_cutoffs=args.ir_cutoffs,
    )
    macro_summary = summarize_macro_tasks(
        task_summary,
        exported_per_query,
        k=args.k,
    )

    pooled_pairwise = pairwise_comparisons(pooled_results)
    pooled_pairwise.insert(0, "model", args.model_name)
    pooled_pairwise.insert(1, "aggregation", "paired_target_queries")

    macro_results = macro_task_results_for_pairwise(
        task_summary,
        method_order=method_order,
        task_order=heldout_tasks,
        k=args.k,
    )
    macro_pairwise = pairwise_comparisons(macro_results)
    macro_pairwise.insert(0, "model", args.model_name)
    macro_pairwise.insert(1, "aggregation", "paired_held_out_tasks")

    transfer_type_loss = results_to_transfer_type_loss(pooled_results)
    transfer_type_loss.insert(0, "model", args.model_name)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_macro_task": outdir / "summary_macro_task.csv",
        "summary_by_heldout_task": outdir / "summary_by_heldout_task.csv",
        "summary_pooled_query": outdir / "summary_pooled_query.csv",
        "per_query": outdir / "per_query.csv",
        "pairwise_macro_task": outdir / "pairwise_macro_task.csv",
        "pairwise_pooled_query": outdir / "pairwise_pooled_query.csv",
        "transfer_type_loss": outdir / "transfer_type_loss.csv",
        "manifest": outdir / "run_manifest.json",
    }

    macro_summary.to_csv(paths["summary_macro_task"], index=False)
    task_summary.to_csv(paths["summary_by_heldout_task"], index=False)
    pooled_summary.to_csv(paths["summary_pooled_query"], index=False)
    exported_per_query.to_csv(paths["per_query"], index=False)
    macro_pairwise.to_csv(paths["pairwise_macro_task"], index=False)
    pooled_pairwise.to_csv(paths["pairwise_pooled_query"], index=False)
    transfer_type_loss.to_csv(paths["transfer_type_loss"], index=False)
    write_manifest(
        paths["manifest"],
        args=args,
        dataset_names=dataset_names,
        performance_map=performance_map,
        self_transfer_stats=self_transfer_stats,
        heldout_tasks=heldout_tasks,
        work_df=work_df,
    )

    print("\nMacro-average across held-out tasks")
    print(macro_summary.to_string(index=False))
    print("\nWrote outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()