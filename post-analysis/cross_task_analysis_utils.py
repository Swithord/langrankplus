#!/usr/bin/env python3

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    FIXED_ENGLISH_PAIRS,
    PARADIGM_METHODS,
    PARADIGM_PAIRS,
    REVIEWER_OUTCOMES,
    SETUP_COLS,
    add_comparison_metadata,
    add_holm_adjustment,
    build_paradigm_target_values,
    classify_method,
    clean_method_rows,
    ensure_dir,
    grouped_bootstrap_summary,
    hierarchical_paired_bootstrap_differences,
    infer_score_column,
    load_data_matrices,
    make_lmm_tables,
    make_resource_tables,
    make_selection_opportunity_tables,
    method_label,
    metric_label,
    model_label,
    outcome_label,
    paired_bootstrap_differences,
    paired_bootstrap_selected_pairs,
    score_multiplier,
    write_csv,
    write_error,
)


AUXILIARY_FILENAMES = [
    "summary_by_heldout_task.csv",
    "summary_macro_task.csv",
    "summary_pooled_query.csv",
    "pairwise_macro_task.csv",
    "pairwise_pooled_query.csv",
    "transfer_type_loss.csv",
]

CROSS_TASK_PRIMARY_OUTCOME = "performance_loss_pct"
ZERO_BASELINE_METHOD = "__zero_baseline__"
VARIANTS = ["without_english", "with_english"]
ENGLISH_SOURCE_IDS = {"eng"}
ORACLE_TOLERANCE = 1e-10


class CrossTaskAnalysisError(RuntimeError):
    """Raised when cross-task artifacts cannot support the requested analysis."""


def _normalise_model_name(value: object) -> str:
    model = str(value).strip().lower().replace("_", "-")
    aliases = {
        "xlmr": "xlm-r",
        "xlm-r": "xlm-r",
        "mt5": "mt5",
        "m-t5": "mt5",
    }
    return aliases.get(model, model)


def _dataset_model_key(dataset: object, model: str) -> str:
    dataset_name = str(dataset).strip()
    suffix = f"_{model}"
    if dataset_name.endswith(suffix):
        return dataset_name
    return f"{dataset_name}{suffix}"


def _finite_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    return out.where(np.isfinite(out))


def _write_run_config(
    outdir: Path,
    root: Path,
    artifact_root: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    config = pd.DataFrame(
        [
            {
                "analysis": "cross_task_generalization",
                "repository_root": str(root.resolve()),
                "artifact_root": str(artifact_root.resolve()),
                "output_directory": str(outdir.resolve()),
                "n_bootstrap": int(n_bootstrap),
                "seed": int(seed),
                "variants": "; ".join(VARIANTS),
                "primary_cross_task_outcome": CROSS_TASK_PRIMARY_OUTCOME,
                "english_baseline_construction": (
                    "For each held-out task/model/target, read transfer_lang=eng "
                    "from data/<held_out_task>_<model>.csv; inherit the cross-task "
                    "artifact oracle and metric metadata; restrict all methods in "
                    "the with-English variant to the same English-supported targets."
                ),
                "raw_gap_note": (
                    "Raw oracle gaps are comparable within a held-out task. "
                    "Cross-task averages combine heterogeneous task-metric point scales "
                    "and should be treated as secondary descriptive results."
                ),
            }
        ]
    )
    write_csv(config, outdir / "run_config.csv")


def variant_tables_dir(outdir: Path, variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown cross-task variant: {variant}")
    return ensure_dir(outdir / "tables" / variant)


def discover_cross_task_model_dirs(artifact_root: str | Path) -> list[Path]:
    artifact_root = Path(artifact_root)
    if not artifact_root.exists():
        raise CrossTaskAnalysisError(
            f"Cross-task artifact directory does not exist: {artifact_root}"
        )

    model_dirs = [
        path
        for path in sorted(artifact_root.iterdir())
        if path.is_dir() and (path / "per_query.csv").is_file()
    ]

    if not model_dirs:
        raise CrossTaskAnalysisError(
            f"No model directories containing per_query.csv were found under {artifact_root}."
        )

    return model_dirs


def _standardise_cross_task_per_query(
    df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    required = {
        "method",
        "held_out_task",
        "dataset",
        "performance_col",
        "target_lang",
        "predicted_performance",
        "actual_best_performance",
        "performance_loss",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise CrossTaskAnalysisError(
            f"{model_dir / 'per_query.csv'} is missing required columns: {missing}"
        )

    out = df.copy()

    if "model" in out.columns and out["model"].notna().any():
        observed_models = sorted(
            {
                _normalise_model_name(value)
                for value in out["model"].dropna().astype(str)
            }
        )
        if len(observed_models) != 1:
            raise CrossTaskAnalysisError(
                f"Expected one model in {model_dir / 'per_query.csv'}, "
                f"found {observed_models}."
            )
        model = observed_models[0]
    else:
        model = _normalise_model_name(model_dir.name)

    directory_model = _normalise_model_name(model_dir.name)
    if directory_model != model:
        raise CrossTaskAnalysisError(
            f"Model directory {model_dir.name!r} conflicts with file model {model!r}."
        )

    out["source_dataset"] = out["dataset"].astype(str)
    out["held_out_task"] = out["held_out_task"].astype(str)
    out["task"] = out["held_out_task"]
    out["model"] = model
    out["model_name"] = model_label(model)
    out["dataset"] = out["held_out_task"].map(
        lambda value: _dataset_model_key(value, model)
    )
    out["dataset_model"] = out["dataset"]

    out["metric"] = out["performance_col"].astype(str)
    out["metric_name"] = out["metric"].map(metric_label)

    out["method_id"] = out["method"].astype(str)
    out["method"] = out["method_id"].map(method_label)
    out["method_type"] = out["method_id"].map(classify_method)
    out["method_group"] = out["method_type"]

    out["target_lang"] = out["target_lang"].astype(str)
    out["source_query_id"] = (
        out["query_id"].astype(str)
        if "query_id" in out.columns
        else out["held_out_task"] + "::" + out["target_lang"]
    )
    out["query_id"] = out["dataset"] + "::" + out["target_lang"]
    out["target_id"] = out["query_id"]

    for column in [
        "target_resource_level",
        "predicted_source_resource_level",
        "actual_best_source_resource_level",
    ]:
        if column in out.columns:
            out[column] = out[column].astype(str).str.strip().str.lower()

    out["predicted_performance"] = _finite_series(out["predicted_performance"])
    out["actual_best_performance"] = _finite_series(out["actual_best_performance"])
    out["performance_loss"] = _finite_series(out["performance_loss"])

    # The supplied cross-task per_query files store performance_loss as a
    # proportion (for example 0.52 means 52%).
    out["performance_loss_pct"] = 100.0 * out["performance_loss"]
    out["raw_oracle_gap"] = (
        out["actual_best_performance"] - out["predicted_performance"]
    )

    multiplier = out.groupby("dataset")["actual_best_performance"].transform(
        score_multiplier
    )
    out["raw_oracle_gap_points"] = multiplier * out["raw_oracle_gap"]
    out["actual_best_performance_points"] = (
        multiplier * out["actual_best_performance"]
    )
    out["predicted_performance_points"] = (
        multiplier * out["predicted_performance"]
    )

    if "ndcg" in out.columns:
        out["ndcg"] = _finite_series(out["ndcg"])
        out["ndcg_pct"] = 100.0 * out["ndcg"]

    out["source_artifact"] = str(model_dir / "per_query.csv")
    out["is_synthetic_english_baseline"] = False

    duplicated = out.duplicated(["query_id", "method"], keep=False)
    out["duplicate_query_method_row"] = duplicated

    return out


def load_cross_task_per_query(
    artifact_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dirs = discover_cross_task_model_dirs(artifact_root)
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for model_dir in model_dirs:
        path = model_dir / "per_query.csv"
        raw = pd.read_csv(path)
        standardised = _standardise_cross_task_per_query(raw, model_dir)
        frames.append(standardised)

        manifest_rows.append(
            {
                "model_directory": model_dir.name,
                "per_query_file": str(path),
                "n_rows": int(standardised.shape[0]),
                "n_methods": int(standardised["method"].nunique()),
                "n_held_out_tasks": int(standardised["task"].nunique()),
                "n_queries": int(standardised["query_id"].nunique()),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["model", "task", "target_lang", "method"],
        kind="stable",
    ).reset_index(drop=True)

    return combined, pd.DataFrame(manifest_rows)


def _align_matrix_scores_to_artifact_scale(
    scores: pd.Series,
    artifact_oracles: pd.Series,
) -> tuple[pd.Series, str]:
    matrix_values = pd.to_numeric(scores, errors="coerce")
    oracle_values = pd.to_numeric(artifact_oracles, errors="coerce").dropna()
    matrix_clean = matrix_values.dropna()

    if matrix_clean.empty or oracle_values.empty:
        return matrix_values, "unchanged"

    matrix_points = float(matrix_clean.max()) > 1.5
    artifact_points = float(oracle_values.max()) > 1.5

    if matrix_points and not artifact_points:
        return matrix_values / 100.0, "matrix divided by 100"
    if not matrix_points and artifact_points:
        return matrix_values * 100.0, "matrix multiplied by 100"
    return matrix_values, "unchanged"


def _english_rows_for_cross_task_setup(
    setup: pd.DataFrame,
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dataset = str(setup["dataset"].iloc[0])
    task = str(setup["task"].iloc[0])
    model = str(setup["model"].iloc[0])
    model_name = str(setup["model_name"].iloc[0])
    metric_values = setup["metric"].dropna().astype(str).unique().tolist()
    metric_name_values = setup["metric_name"].dropna().astype(str).unique().tolist()

    if len(metric_values) != 1:
        raise CrossTaskAnalysisError(
            f"Expected exactly one metric in cross-task setup {dataset}, "
            f"found {metric_values}."
        )

    metric = metric_values[0]
    metric_name = metric_name_values[0] if len(metric_name_values) == 1 else metric_label(metric)
    n_targets_full = int(setup["target_lang"].nunique())

    base_status: dict[str, object] = {
        "dataset": dataset,
        "held_out_task": task,
        "model": model,
        "model_name": model_name,
        "metric": metric,
        "metric_name": metric_name,
        "n_targets_full": n_targets_full,
        "n_targets_with_english": 0,
        "n_targets_missing_english": n_targets_full,
        "english_available": False,
        "used_in_with_english": False,
        "matrix_score_column": "",
        "score_scale_adjustment": "",
        "n_english_above_artifact_oracle": 0,
        "reason": "",
    }

    required_matrix_columns = {"task_lang", "transfer_lang"}
    if not required_matrix_columns.issubset(matrix.columns):
        base_status["reason"] = "matrix missing task_lang or transfer_lang"
        return pd.DataFrame(), base_status

    score_col = infer_score_column(matrix)
    base_status["matrix_score_column"] = score_col

    working = matrix[["task_lang", "transfer_lang", score_col]].copy()
    working["task_lang"] = working["task_lang"].astype(str)
    working["transfer_lang"] = (
        working["transfer_lang"].astype(str).str.strip().str.lower()
    )
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    working = working.dropna(subset=[score_col])

    english = working.loc[
        working["transfer_lang"].isin(ENGLISH_SOURCE_IDS)
    ].copy()
    if english.empty:
        base_status["reason"] = "no English transfer source in held-out task matrix"
        return pd.DataFrame(), base_status

    english = (
        english.groupby("task_lang", as_index=False)[score_col]
        .max()
        .rename(columns={"task_lang": "target_lang", score_col: "english_score"})
    )
    english["target_lang"] = english["target_lang"].astype(str)

    # Start from one actual cross-task row per target so all split metadata and
    # optional columns are preserved. Method-dependent fields are overwritten.
    canonical = (
        setup.sort_values(["target_lang", "method"], kind="stable")
        .drop_duplicates("target_lang", keep="first")
        .copy()
    )

    oracle_by_target = (
        setup.groupby("target_lang", as_index=False)["actual_best_performance"]
        .max()
        .rename(columns={"actual_best_performance": "artifact_oracle"})
    )
    canonical = canonical.drop(columns=["actual_best_performance"], errors="ignore")
    canonical = canonical.merge(oracle_by_target, on="target_lang", how="left")
    canonical = canonical.merge(english, on="target_lang", how="inner")

    if canonical.empty:
        base_status["reason"] = "English source has no shared cross-task target rows"
        return pd.DataFrame(), base_status

    aligned, scale_note = _align_matrix_scores_to_artifact_scale(
        canonical["english_score"],
        canonical["artifact_oracle"],
    )
    canonical["english_score"] = aligned
    base_status["score_scale_adjustment"] = scale_note

    canonical["actual_best_performance"] = pd.to_numeric(
        canonical["artifact_oracle"], errors="coerce"
    )
    canonical["predicted_performance"] = pd.to_numeric(
        canonical["english_score"], errors="coerce"
    )

    canonical["method_id"] = "always_eng"
    canonical["method"] = "Always English"
    canonical["method_type"] = "English"
    canonical["method_group"] = "English"
    canonical["predicted_best_source"] = "eng"
    canonical["predicted_source_resource_level"] = "hrl"
    canonical["is_synthetic_english_baseline"] = True
    canonical["source_artifact"] = str(matrix.get("data_path", pd.Series([""])).iloc[0])

    actual = canonical["actual_best_performance"]
    predicted = canonical["predicted_performance"]
    denominator = actual.where(actual > 0)
    canonical["performance_loss"] = (actual - predicted) / denominator
    canonical["performance_loss_pct"] = 100.0 * canonical["performance_loss"]
    canonical["raw_oracle_gap"] = actual - predicted

    multiplier = score_multiplier(actual)
    canonical["raw_oracle_gap_points"] = multiplier * canonical["raw_oracle_gap"]
    canonical["actual_best_performance_points"] = multiplier * actual
    canonical["predicted_performance_points"] = multiplier * predicted

    # Rank-based metrics are undefined for a fixed source baseline.
    rank_columns = [
        "ndcg",
        "ndcg_pct",
        "top_1_accuracy",
        "top_3_accuracy",
        "mrr",
        "exact_best_rank",
        "r_precision",
    ] + [
        column
        for column in canonical.columns
        if "@" in str(column)
    ]
    for column in rank_columns:
        if column in canonical.columns:
            canonical[column] = np.nan

    canonical["english_exceeds_artifact_oracle"] = (
        predicted > actual + ORACLE_TOLERANCE
    )
    canonical["duplicate_query_method_row"] = False
    canonical = canonical.drop(columns=["artifact_oracle", "english_score"], errors="ignore")

    n_english = int(canonical["target_lang"].nunique())
    n_above = int(canonical["english_exceeds_artifact_oracle"].sum())
    base_status.update(
        {
            "n_targets_with_english": n_english,
            "n_targets_missing_english": n_targets_full - n_english,
            "english_available": n_english > 0,
            "used_in_with_english": n_english > 0,
            "n_english_above_artifact_oracle": n_above,
            "reason": "" if n_english > 0 else "no shared English-supported targets",
        }
    )

    return canonical, base_status


def make_cross_task_variants(
    base_per_query: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Reconstruct Always English deterministically from the held-out task matrix.
    # Remove any pre-existing English rows first so stale or partially generated
    # artifacts cannot produce duplicate baseline rows.
    without_english = clean_method_rows(base_per_query.copy())
    if "method_type" in without_english.columns:
        without_english = without_english.loc[
            ~without_english["method_type"].eq("English")
        ].copy()
    if "method" in without_english.columns:
        without_english = without_english.loc[
            ~without_english["method"].eq("Always English")
        ].copy()

    with_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for dataset, setup in without_english.groupby("dataset", dropna=False, sort=False):
        dataset = str(dataset)
        matrix = data_matrices.get(dataset)

        if matrix is None:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "held_out_task": setup["task"].iloc[0],
                    "model": setup["model"].iloc[0],
                    "model_name": setup["model_name"].iloc[0],
                    "metric": setup["metric"].iloc[0],
                    "metric_name": setup["metric_name"].iloc[0],
                    "n_targets_full": int(setup["target_lang"].nunique()),
                    "n_targets_with_english": 0,
                    "n_targets_missing_english": int(setup["target_lang"].nunique()),
                    "english_available": False,
                    "used_in_with_english": False,
                    "matrix_score_column": "",
                    "score_scale_adjustment": "",
                    "n_english_above_artifact_oracle": 0,
                    "reason": "missing held-out task/model data matrix",
                }
            )
            continue

        english_rows, status = _english_rows_for_cross_task_setup(setup, matrix)
        coverage_rows.append(status)
        if english_rows.empty:
            continue

        supported_targets = set(english_rows["target_lang"].astype(str))
        method_rows = setup.loc[
            setup["target_lang"].astype(str).isin(supported_targets)
        ].copy()

        if method_rows.empty:
            continue

        with_frames.extend([method_rows, english_rows])

    coverage = pd.DataFrame(coverage_rows)

    if with_frames:
        with_english = pd.concat(with_frames, ignore_index=True, sort=False)
        with_english = clean_method_rows(with_english)
        with_english = with_english.sort_values(
            ["model", "task", "target_lang", "method"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        with_english = pd.DataFrame(columns=without_english.columns)

    return without_english, with_english, coverage


def combine_auxiliary_cross_task_files(
    artifact_root: str | Path,
    output_dir: Path,
) -> pd.DataFrame:
    model_dirs = discover_cross_task_model_dirs(artifact_root)
    output_dir = ensure_dir(output_dir)
    manifest_rows: list[dict[str, object]] = []

    for filename in AUXILIARY_FILENAMES:
        frames: list[pd.DataFrame] = []

        for model_dir in model_dirs:
            path = model_dir / filename
            if not path.is_file():
                manifest_rows.append(
                    {
                        "source_file": str(path),
                        "combined_file": "",
                        "status": "missing",
                        "n_rows": 0,
                    }
                )
                continue

            frame = pd.read_csv(path)
            model = _normalise_model_name(model_dir.name)

            if "model" not in frame.columns:
                frame.insert(0, "model", model)
            else:
                frame["model"] = frame["model"].fillna(model).map(
                    _normalise_model_name
                )

            if "model_name" not in frame.columns:
                frame.insert(1, "model_name", frame["model"].map(model_label))

            frame["source_artifact"] = str(path)
            frames.append(frame)

        output_path = output_dir / f"source_{Path(filename).stem}_combined.csv"

        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            write_csv(combined, output_path)
            manifest_rows.append(
                {
                    "source_file": filename,
                    "combined_file": str(output_path),
                    "status": "written",
                    "n_rows": int(combined.shape[0]),
                }
            )
        else:
            write_error(output_path, f"no {filename} files found")

    manifest = pd.DataFrame(manifest_rows)
    write_csv(manifest, output_dir / "source_artifact_combination_manifest.csv")
    return manifest


def _add_outcome_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["outcome_name"] = out["outcome"].map(outcome_label)
    out["cross_task_comparability"] = np.where(
        out["outcome"].eq("performance_loss_pct"),
        "scale-free; suitable for cross-task aggregation",
        "task-metric points; cross-task averages are secondary descriptive results",
    )
    return out


def make_cross_task_main_metric_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    df = clean_method_rows(per_query)
    if df.empty:
        write_error(folder / "main_metric_bootstrap_ci_by_heldout_task.csv", "no rows")
        return

    group_cols = SETUP_COLS + ["method", "method_type"]
    value_cols = [column for column in REVIEWER_OUTCOMES if column in df.columns]

    summary = grouped_bootstrap_summary(
        df,
        group_cols=group_cols,
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = _add_outcome_metadata(summary)
    write_csv(summary, folder / "main_metric_bootstrap_ci_by_heldout_task.csv")

    for outcome_index, outcome in enumerate(value_cols):
        pairwise = paired_bootstrap_differences(
            df,
            group_cols=SETUP_COLS,
            method_col="method",
            value_col=outcome,
            unit_col="query_id",
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 * outcome_index,
        )
        pairwise = add_comparison_metadata(
            pairwise,
            comparison_family="All methods within held-out task",
        )
        pairwise = add_holm_adjustment(
            pairwise,
            family_cols=SETUP_COLS + ["outcome"],
        )
        pairwise = _add_outcome_metadata(pairwise)
        write_csv(pairwise, folder / f"pairwise_{outcome}_by_heldout_task.csv")


def make_pooled_query_method_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    df = clean_method_rows(per_query)
    if df.empty:
        write_error(folder / "main_metric_bootstrap_ci_pooled_query.csv", "no rows")
        return

    group_cols = ["model", "model_name", "method", "method_type"]
    value_cols = [column for column in REVIEWER_OUTCOMES if column in df.columns]

    pooled = grouped_bootstrap_summary(
        df,
        group_cols=group_cols,
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    pooled = _add_outcome_metadata(pooled)
    pooled["weighting"] = "pooled queries; tasks with more targets receive more weight"
    write_csv(pooled, folder / "main_metric_bootstrap_ci_pooled_query.csv")


def _zero_augmented_method_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    base = df[SETUP_COLS + ["query_id", "target_lang", "method", outcome]].dropna(
        subset=["query_id", "method", outcome]
    )
    zero = (
        base[SETUP_COLS + ["query_id", "target_lang"]]
        .drop_duplicates()
        .assign(method=ZERO_BASELINE_METHOD, **{outcome: 0.0})
    )
    return pd.concat([base, zero], ignore_index=True, sort=False)


def make_macro_task_method_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    df = clean_method_rows(per_query)
    rows: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        if outcome not in df.columns:
            continue

        for model_index, ((model, model_name), model_df) in enumerate(
            df.groupby(["model", "model_name"], dropna=False, sort=False)
        ):
            methods = sorted(model_df["method"].dropna().astype(str).unique())
            if not methods:
                continue

            augmented = _zero_augmented_method_frame(model_df, outcome)
            comparisons = hierarchical_paired_bootstrap_differences(
                augmented,
                setup_cols=SETUP_COLS,
                method_col="method",
                value_col=outcome,
                unit_col="query_id",
                pairs=[(method, ZERO_BASELINE_METHOD) for method in methods],
                n_bootstrap=n_bootstrap,
                seed=seed + 10000 * outcome_index + 101 * model_index,
            )
            if comparisons.empty:
                continue

            comparisons.insert(0, "model", model)
            comparisons.insert(1, "model_name", model_name)
            comparisons["method"] = comparisons["method_a"]
            method_type_lookup = (
                model_df[["method", "method_type"]]
                .drop_duplicates("method")
                .set_index("method")["method_type"]
                .to_dict()
            )
            comparisons["method_type"] = comparisons["method"].map(method_type_lookup)
            comparisons["mean"] = comparisons["mean_difference_a_minus_b"]
            comparisons["bootstrap_p_value_vs_zero"] = comparisons[
                "bootstrap_p_value"
            ]
            comparisons["weighting"] = (
                "equal held-out-task weight with target resampling within task"
            )
            rows.append(comparisons)

    if not rows:
        write_error(
            folder / "main_metric_bootstrap_ci_macro_task.csv",
            "no macro-task method summaries available",
        )
        return

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = _add_outcome_metadata(out)
    preferred = [
        "model",
        "model_name",
        "method",
        "method_type",
        "outcome",
        "outcome_name",
        "mean",
        "ci_lower",
        "ci_upper",
        "n_setups",
        "n_paired_units",
        "min_units_per_setup",
        "median_units_per_setup",
        "max_units_per_setup",
        "weighting",
        "cross_task_comparability",
        "included_setups",
    ]
    remaining = [column for column in out.columns if column not in preferred]
    write_csv(out[preferred + remaining], folder / "main_metric_bootstrap_ci_macro_task.csv")


def _all_method_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    methods = sorted(df["method"].dropna().astype(str).unique())
    return list(itertools.combinations(methods, 2))


def make_macro_task_pairwise_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    df = clean_method_rows(per_query)
    by_model_frames: list[pd.DataFrame] = []
    all_model_frames: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        if outcome not in df.columns:
            continue

        for model_index, ((model, model_name), model_df) in enumerate(
            df.groupby(["model", "model_name"], dropna=False, sort=False)
        ):
            pairs = _all_method_pairs(model_df)
            if not pairs:
                continue

            result = hierarchical_paired_bootstrap_differences(
                model_df,
                setup_cols=SETUP_COLS,
                method_col="method",
                value_col=outcome,
                unit_col="query_id",
                pairs=pairs,
                n_bootstrap=n_bootstrap,
                seed=seed + 20000 * outcome_index + 401 * model_index,
            )
            if result.empty:
                continue
            result.insert(0, "model", model)
            result.insert(1, "model_name", model_name)
            by_model_frames.append(result)

        all_pairs = _all_method_pairs(df)
        if all_pairs:
            all_result = hierarchical_paired_bootstrap_differences(
                df,
                setup_cols=SETUP_COLS,
                method_col="method",
                value_col=outcome,
                unit_col="query_id",
                pairs=all_pairs,
                n_bootstrap=n_bootstrap,
                seed=seed + 30000 * outcome_index,
            )
            all_model_frames.append(all_result)

    if by_model_frames:
        by_model = pd.concat(by_model_frames, ignore_index=True, sort=False)
        by_model = add_comparison_metadata(
            by_model,
            comparison_family="All methods, macro held-out task, by model",
        )
        by_model = add_holm_adjustment(by_model, family_cols=["model", "outcome"])
        by_model = _add_outcome_metadata(by_model)
        write_csv(by_model, folder / "pairwise_macro_task_by_model.csv")
    else:
        write_error(
            folder / "pairwise_macro_task_by_model.csv",
            "no model-specific macro-task pairwise comparisons available",
        )

    if all_model_frames:
        overall = pd.concat(all_model_frames, ignore_index=True, sort=False)
        overall = add_comparison_metadata(
            overall,
            comparison_family="All methods, macro task-model setups",
        )
        overall = add_holm_adjustment(overall, family_cols=["outcome"])
        overall = _add_outcome_metadata(overall)
        write_csv(overall, folder / "pairwise_macro_task_all_models.csv")
    else:
        write_error(
            folder / "pairwise_macro_task_all_models.csv",
            "no all-model macro-task pairwise comparisons available",
        )


def make_cross_task_paradigm_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    by_setup_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []
    by_model_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        if outcome not in per_query.columns:
            continue

        target_values, coverage = build_paradigm_target_values(
            per_query,
            outcome=outcome,
        )
        if not coverage.empty:
            coverage_frames.append(coverage)
        if target_values.empty:
            continue

        by_setup = paired_bootstrap_selected_pairs(
            target_values,
            group_cols=SETUP_COLS,
            method_col="paradigm",
            value_col=outcome,
            unit_col="query_id",
            pairs=PARADIGM_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 40000 * outcome_index,
        )
        by_setup_frames.append(by_setup)

        overall = hierarchical_paired_bootstrap_differences(
            target_values,
            setup_cols=SETUP_COLS,
            method_col="paradigm",
            value_col=outcome,
            unit_col="query_id",
            pairs=PARADIGM_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 50000 * outcome_index,
        )
        overall_frames.append(overall)

        for model_index, ((model, model_name), model_df) in enumerate(
            target_values.groupby(["model", "model_name"], dropna=False, sort=False)
        ):
            result = hierarchical_paired_bootstrap_differences(
                model_df,
                setup_cols=SETUP_COLS,
                method_col="paradigm",
                value_col=outcome,
                unit_col="query_id",
                pairs=PARADIGM_PAIRS,
                n_bootstrap=n_bootstrap,
                seed=seed + 60000 * outcome_index + 503 * model_index,
            )
            if result.empty:
                continue
            result.insert(0, "model", model)
            result.insert(1, "model_name", model_name)
            by_model_frames.append(result)

    if coverage_frames:
        coverage = pd.concat(coverage_frames, ignore_index=True, sort=False)
        write_csv(coverage, folder / "paradigm_target_coverage.csv")
    else:
        write_error(folder / "paradigm_target_coverage.csv", "no paradigm target coverage available")

    def finish(
        frames: list[pd.DataFrame],
        path: Path,
        family_cols: list[str],
        family_name: str,
    ) -> None:
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            write_error(path, f"no {family_name.lower()} comparisons available")
            return
        out = pd.concat(nonempty, ignore_index=True, sort=False)
        out = add_comparison_metadata(out, comparison_family=family_name)
        out["method_a_components"] = out["method_a"].map(
            lambda value: "; ".join(PARADIGM_METHODS.get(str(value), []))
        )
        out["method_b_components"] = out["method_b"].map(
            lambda value: "; ".join(PARADIGM_METHODS.get(str(value), []))
        )
        out = add_holm_adjustment(out, family_cols=family_cols)
        out = _add_outcome_metadata(out)
        write_csv(out, path)

    finish(
        by_setup_frames,
        folder / "paradigm_paired_tests_by_heldout_task.csv",
        SETUP_COLS + ["outcome"],
        "Between paradigms within held-out task",
    )
    finish(
        by_model_frames,
        folder / "paradigm_paired_tests_macro_task_by_model.csv",
        ["model", "outcome"],
        "Between paradigms, macro held-out task, by model",
    )
    finish(
        overall_frames,
        folder / "paradigm_paired_tests_macro_task_all_models.csv",
        ["outcome"],
        "Between paradigms, macro task-model setups",
    )


def make_cross_task_fixed_english_tables(
    per_query: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    by_setup_frames: list[pd.DataFrame] = []
    by_model_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        if outcome not in per_query.columns:
            continue

        by_setup = paired_bootstrap_selected_pairs(
            per_query,
            group_cols=SETUP_COLS,
            method_col="method",
            value_col=outcome,
            unit_col="query_id",
            pairs=FIXED_ENGLISH_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 * outcome_index,
        )
        by_setup_frames.append(by_setup)

        for model_index, ((model, model_name), model_df) in enumerate(
            per_query.groupby(["model", "model_name"], dropna=False, sort=False)
        ):
            by_model = hierarchical_paired_bootstrap_differences(
                model_df,
                setup_cols=SETUP_COLS,
                method_col="method",
                value_col=outcome,
                unit_col="query_id",
                pairs=FIXED_ENGLISH_PAIRS,
                n_bootstrap=n_bootstrap,
                seed=seed + 10000 * outcome_index + 307 * model_index,
            )
            if by_model.empty:
                continue
            by_model.insert(0, "model", model)
            by_model.insert(1, "model_name", model_name)
            by_model_frames.append(by_model)

        overall = hierarchical_paired_bootstrap_differences(
            per_query,
            setup_cols=SETUP_COLS,
            method_col="method",
            value_col=outcome,
            unit_col="query_id",
            pairs=FIXED_ENGLISH_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 20000 * outcome_index,
        )
        overall_frames.append(overall)

    def finish(
        frames: list[pd.DataFrame],
        path: Path,
        family_cols: list[str],
        family_name: str,
    ) -> pd.DataFrame:
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            write_error(path, f"no {family_name.lower()} comparisons available")
            return pd.DataFrame()
        out = pd.concat(nonempty, ignore_index=True, sort=False)
        out = add_comparison_metadata(out, comparison_family=family_name)
        out = add_holm_adjustment(out, family_cols=family_cols)
        out = _add_outcome_metadata(out)
        write_csv(out, path)
        return out

    finish(
        by_setup_frames,
        folder / "fixed_methods_vs_english_by_heldout_task.csv",
        SETUP_COLS + ["outcome"],
        "Fixed method versus Always English within held-out task",
    )
    finish(
        by_model_frames,
        folder / "fixed_methods_vs_english_macro_task_by_model.csv",
        ["model", "outcome"],
        "Fixed method versus Always English, macro held-out task, by model",
    )
    overall = finish(
        overall_frames,
        folder / "fixed_methods_vs_english_overall.csv",
        ["outcome"],
        "Fixed method versus Always English, macro task-model setups",
    )
    if not overall.empty:
        write_csv(
            overall,
            folder / "fixed_methods_vs_english_macro_task_all_models.csv",
        )


def make_cross_task_resource_tables(per_query: pd.DataFrame, folder: Path) -> None:
    make_resource_tables(per_query, folder)
    make_lmm_tables(per_query, folder)


def make_cross_task_selection_opportunity_tables(
    root: Path,
    per_query: pd.DataFrame,
    folder: Path,
) -> None:
    data_matrices = load_data_matrices(root)
    make_selection_opportunity_tables(per_query, data_matrices, folder)


def write_cross_task_status_tables(
    per_query: pd.DataFrame,
    folder: Path,
    variant: str,
) -> None:
    methods = set(per_query.get("method", pd.Series(dtype=str)).dropna().astype(str))
    english_present = "Always English" in methods
    n_english_rows = int(
        per_query.loc[
            per_query.get("method", pd.Series(index=per_query.index, dtype=str))
            .eq("Always English")
        ].shape[0]
    )
    n_english_setups = int(
        per_query.loc[
            per_query.get("method", pd.Series(index=per_query.index, dtype=str))
            .eq("Always English"),
            "dataset",
        ].nunique()
    ) if "dataset" in per_query.columns else 0

    english_status = pd.DataFrame(
        [
            {
                "analysis": "Always English baseline",
                "variant": variant,
                "generated": english_present,
                "n_english_rows": n_english_rows,
                "n_english_setups": n_english_setups,
                "reason": (
                    "Always English was reconstructed from held-out task matrices."
                    if english_present
                    else (
                        "Intentionally omitted from the without-English variant."
                        if variant == "without_english"
                        else "No held-out task/model/target rows supported an English source."
                    )
                ),
            }
        ]
    )
    write_csv(english_status, folder / "english_comparison_status.csv")

    ctc_columns = sorted(
        column for column in per_query.columns if str(column).startswith("cnotc_")
    )
    ctc_status = pd.DataFrame(
        [
            {
                "analysis": "Mondrian CTC",
                "variant": variant,
                "generated": bool(ctc_columns),
                "available_columns": "; ".join(ctc_columns),
                "reason": (
                    "CTC columns are available."
                    if ctc_columns
                    else "Not generated: cross-task per_query.csv contains no CTC outputs."
                ),
            }
        ]
    )
    write_csv(ctc_status, folder / "ctc_status.csv")


def write_cross_task_variant_tables(
    variant: str,
    per_query: pd.DataFrame,
    root: Path,
    outdir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    folder = variant_tables_dir(outdir, variant)
    write_csv(per_query, outdir / f"combined_cross_task_per_query_{variant}.csv")

    if per_query.empty:
        write_error(folder / "variant_status.csv", "no rows in variant")
        write_cross_task_status_tables(per_query, folder, variant)
        return

    make_cross_task_main_metric_tables(per_query, folder, n_bootstrap, seed)
    make_pooled_query_method_tables(per_query, folder, n_bootstrap, seed + 70000)
    make_macro_task_method_tables(per_query, folder, n_bootstrap, seed + 80000)
    make_macro_task_pairwise_tables(per_query, folder, n_bootstrap, seed + 90000)
    make_cross_task_paradigm_tables(per_query, folder, n_bootstrap, seed + 100000)
    make_cross_task_resource_tables(per_query, folder)
    make_cross_task_selection_opportunity_tables(root, per_query, folder)

    if variant == "with_english":
        make_cross_task_fixed_english_tables(
            per_query,
            folder,
            n_bootstrap,
            seed + 110000,
        )

    write_cross_task_status_tables(per_query, folder, variant)


def write_cross_task_manifest(outdir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(outdir.rglob("*.csv")):
        if path.name == "table_manifest.csv":
            continue
        try:
            frame = pd.read_csv(path)
            n_rows = int(frame.shape[0])
            n_columns = int(frame.shape[1])
        except Exception:
            n_rows = -1
            n_columns = -1
        relative = path.relative_to(outdir)
        variant = (
            relative.parts[1]
            if len(relative.parts) >= 3
            and relative.parts[0] == "tables"
            and relative.parts[1] in VARIANTS
            else "shared"
        )
        rows.append(
            {
                "variant": variant,
                "file": str(relative),
                "n_rows": n_rows,
                "n_columns": n_columns,
            }
        )
    write_csv(pd.DataFrame(rows), outdir / "table_manifest.csv")


def run_cross_task_revision_analysis(
    root: str | Path,
    artifact_root: str | Path | None = None,
    outdir: str | Path = "post-analysis/cross-task-outputs",
    n_bootstrap: int = 20000,
    seed: int = 42,
) -> None:
    root = Path(root).resolve()
    artifact_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else (root / "artifacts" / "cross_task_generalization").resolve()
    )
    outdir = ensure_dir(Path(outdir).resolve())
    ensure_dir(outdir / "tables")
    for variant in VARIANTS:
        variant_tables_dir(outdir, variant)

    _write_run_config(
        outdir=outdir,
        root=root,
        artifact_root=artifact_root,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    print("[1/8] Loading cross-task per-query artifacts", flush=True)
    base_per_query, input_manifest = load_cross_task_per_query(artifact_root)
    write_csv(input_manifest, outdir / "input_manifest.csv")
    write_csv(base_per_query, outdir / "combined_cross_task_per_query_all_methods.csv")

    print("[2/8] Loading held-out task matrices and constructing English variants", flush=True)
    data_matrices = load_data_matrices(root)
    without_english, with_english, english_coverage = make_cross_task_variants(
        base_per_query,
        data_matrices,
    )
    write_csv(
        english_coverage,
        variant_tables_dir(outdir, "with_english") / "english_subset_coverage.csv",
    )

    print("[3/8] Combining original cross-task summary artifacts", flush=True)
    combine_auxiliary_cross_task_files(
        artifact_root,
        ensure_dir(outdir / "tables" / "source_artifacts"),
    )

    print("[4/8] Generating full-benchmark tables without English", flush=True)
    write_cross_task_variant_tables(
        variant="without_english",
        per_query=without_english,
        root=root,
        outdir=outdir,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    print("[5/8] Generating common-subset tables with Always English", flush=True)
    write_cross_task_variant_tables(
        variant="with_english",
        per_query=with_english,
        root=root,
        outdir=outdir,
        n_bootstrap=n_bootstrap,
        seed=seed + 200000,
    )

    print("[6/8] Validating English target pairing", flush=True)
    if not with_english.empty:
        english_targets = set(
            with_english.loc[
                with_english["method"].eq("Always English"), "query_id"
            ].astype(str)
        )
        nonenglish = with_english.loc[
            ~with_english["method"].eq("Always English")
        ]
        unsupported = set(nonenglish["query_id"].astype(str)) - english_targets
        if unsupported:
            raise CrossTaskAnalysisError(
                "The with-English variant contains ordinary method targets without "
                f"an English baseline row. Example: {sorted(unsupported)[:5]}"
            )

    # print("[7/8] Writing manifest", flush=True)
    # write_cross_task_manifest(outdir)

    # print("[8/8] Validating outputs", flush=True)
    # generated = [path for path in outdir.rglob("*.csv")]
    # if not generated:
    #     raise CrossTaskAnalysisError(
    #         f"Analysis completed without generating CSV files under {outdir}."
    #     )

    # print(f"Generated {len(generated)} CSV files under: {outdir}", flush=True)