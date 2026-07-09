#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    RESOURCE_LEVELS,
    ensure_dir,
    finite_numeric,
    grouped_bootstrap_summary,
    infer_score_column,
    infer_task_model,
    load_data_matrices,
    paired_bootstrap_differences,
    target_resource_lookup,
    write_csv,
)


REVIEWER_OUTCOMES = [
    "performance_loss_pct",
    "raw_oracle_gap_points",
]

MAIN_LMM_METHOD_TYPES = [
    "Individual",
    "Composite",
    "Trained",
    "English",
]

CTC_COLS = [
    "cnotc_trial_complexity",
    "cnotc_pool_fraction",
    "cnotc_near_oracle_coverage",
    "cnotc_exact_best_coverage",
    "cnotc_best_in_set_performance_loss",
]

DISTANCE_METHODS = {
    "single_new_gen",
    "single_new_typ",
    "single_new_geo",
    "single_script",
    "single_distals_asjp",
    "single_distals_wiki_size",
    "new_gen",
    "new_typ",
    "new_geo",
    "script",
    "distals_asjp",
    "distals_wiki_size",
}

METHOD_LABELS = {
    "always_eng": "Always English",
    "always_english": "Always English",
    "english": "Always English",
    "composite_equal": "Composite-Equal",
    "composite_equal_weight": "Composite-Equal",
    "rrf_equal": "Composite-RRF",
    "composite_rrf": "Composite-RRF",
    "lightgbm_lambdarank": "LightGBM",
    "lightgbm": "LightGBM",
    "mlp_listnet": "MLP",
    "mlp": "MLP",
    "nnrank": "NNRank",
    "single_new_gen": "Genetic",
    "new_gen": "Genetic",
    "single_new_typ": "Typological",
    "new_typ": "Typological",
    "single_new_geo": "Geographic",
    "new_geo": "Geographic",
    "single_script": "Script",
    "script": "Script",
    "single_distals_asjp": "ASJP",
    "distals_asjp": "ASJP",
    "single_distals_wiki_size": "Wikipedia size",
    "distals_wiki_size": "Wikipedia size",
    "random": "Random",
}

MODEL_LABELS = {
    "mt5": "mT5",
    "xlm-r": "XLM-R",
}

METRIC_LABELS = {
    "f1_score": "F1/LAS",
    "f1": "F1/LAS",
    "bleu": "BLEU",
}

OUTCOME_LABELS = {
    "performance_loss_pct": "PL (%)",
    "raw_oracle_gap_points": "Raw gap",
}

RESOURCE_LABELS = {
    "hrl": "HRL",
    "mrl": "MRL",
    "lrl": "LRL",
}

VARIANTS = [
    "with_english",
    "without_english",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-retraining reviewer-response tables for LangRankPlus."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("post-analysis/outputs"))
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-nnrank", action="store_true")
    parser.add_argument("--min-ctc-targets", type=int, default=50)
    return parser.parse_args()


def tables_dir(outdir: Path, variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown table variant: {variant}")

    return ensure_dir(outdir / "tables" / variant)


def normalise_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def method_label(method: object) -> str:
    key = normalise_name(method)
    return METHOD_LABELS.get(key, str(method).strip())


def classify_method(method: object) -> str:
    key = normalise_name(method)

    if key == "random":
        return "Random"

    if key in {"always_eng", "always_english", "english"}:
        return "English"

    if "nnrank" in key:
        return "NNRank"

    if key in DISTANCE_METHODS or key.startswith("single_"):
        return "Individual"

    if "composite" in key or "rrf" in key or key in {"equal", "rrf_equal"}:
        return "Composite"

    if (
        "lightgbm" in key
        or "lambdarank" in key
        or "lambda" in key
        or "mlp" in key
        or "listnet" in key
    ):
        return "Trained"

    return "Other"


def model_label(model: object) -> str:
    key = str(model).strip()
    return MODEL_LABELS.get(key, key)


def metric_label(metric: object) -> str:
    key = str(metric).strip()
    return METRIC_LABELS.get(key, key)


def outcome_label(outcome: object) -> str:
    key = str(outcome).strip()
    return OUTCOME_LABELS.get(key, key)


def resource_label(resource: object) -> str:
    key = str(resource).strip().lower()
    return RESOURCE_LABELS.get(key, str(resource))


def metric_from_per_fold_path(path: Path) -> str:
    stem = path.stem

    if stem.startswith("per_fold_"):
        return stem.removeprefix("per_fold_")

    return stem


def score_multiplier(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return 100.0

    if float(clean.max()) > 1.5:
        return 1.0

    return 100.0


def clean_method_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "method_type" not in df.columns:
        return df.copy()

    return df.query("method_type != 'Random'").copy()


def write_error(path: Path, message: str) -> None:
    write_csv(pd.DataFrame({"error": [message]}), path)


def standardise_artifact_metadata(
    df: pd.DataFrame,
    artifact_dataset: str,
    metric: str,
) -> pd.DataFrame:
    out = df.copy()

    task, model = infer_task_model(artifact_dataset)

    out["dataset"] = artifact_dataset
    out["dataset_model"] = artifact_dataset
    out["task"] = task
    out["model"] = model
    out["model_name"] = model_label(model)
    out["metric"] = metric
    out["metric_name"] = metric_label(metric)

    if "method" not in out.columns:
        raise ValueError(f"Missing required column 'method' in artifact {artifact_dataset}.")

    out["method_id"] = out["method"].astype(str)
    out["method"] = out["method_id"].map(method_label)
    out["method_type"] = out["method_id"].map(classify_method)
    out["method_group"] = out["method_type"]

    if "target_lang" not in out.columns:
        if "query_id" in out.columns:
            out["target_lang"] = out["query_id"].astype(str)
        else:
            raise ValueError(
                f"Missing both 'target_lang' and 'query_id' in artifact {artifact_dataset}."
            )

    out["target_lang"] = out["target_lang"].astype(str)
    out["query_id"] = out["dataset"].astype(str) + "::" + out["target_lang"].astype(str)
    out["target_id"] = out["query_id"]

    for col in [
        "target_resource_level",
        "predicted_source_resource_level",
        "actual_best_source_resource_level",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower()

    return out


def recompute_scale_columns(per_fold: pd.DataFrame) -> pd.DataFrame:
    df = per_fold.copy()

    if "performance_loss" in df.columns:
        pl = pd.to_numeric(df["performance_loss"], errors="coerce")

        if pl.dropna().empty:
            df["performance_loss_pct"] = np.nan
        elif float(pl.dropna().max()) > 1.5:
            df["performance_loss_pct"] = pl
        else:
            df["performance_loss_pct"] = 100.0 * pl

    needed = {"dataset", "actual_best_performance", "predicted_performance"}
    if needed.issubset(df.columns):
        actual = pd.to_numeric(df["actual_best_performance"], errors="coerce")
        predicted = pd.to_numeric(df["predicted_performance"], errors="coerce")

        df["actual_best_performance"] = actual
        df["predicted_performance"] = predicted
        df["raw_oracle_gap"] = actual - predicted

        mult_by_dataset = (
            df.groupby("dataset")["actual_best_performance"]
            .transform(score_multiplier)
        )

        df["raw_oracle_gap_points"] = mult_by_dataset * df["raw_oracle_gap"]
        df["actual_best_performance_points"] = mult_by_dataset * actual
        df["predicted_performance_points"] = mult_by_dataset * predicted

    return df


def load_per_fold_artifacts(root: Path, nnrank: bool = False) -> pd.DataFrame:
    root = Path(root)

    if nnrank:
        artifact_root = root / "artifacts" / "nnrank"
    else:
        artifact_root = root / "artifacts"

    if not artifact_root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for dataset_dir in sorted(artifact_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        if not nnrank and dataset_dir.name == "nnrank":
            continue

        for path in sorted(dataset_dir.glob("per_fold_*.csv")):
            metric = metric_from_per_fold_path(path)
            df = pd.read_csv(path)

            df = standardise_artifact_metadata(
                df=df,
                artifact_dataset=dataset_dir.name,
                metric=metric,
            )
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    return recompute_scale_columns(out)


def assert_no_unknown_models(df: pd.DataFrame, label: str) -> None:
    if "model" not in df.columns:
        raise RuntimeError(f"{label} has no 'model' column.")

    unknown = (
        df.loc[df["model"].astype(str).eq("unknown"), ["dataset", "task", "model"]]
        .drop_duplicates()
        .sort_values(["dataset", "task", "model"])
    )

    if not unknown.empty:
        msg = unknown.to_string(index=False)
        raise RuntimeError(
            f"{label} still contains model='unknown'. "
            f"This should not happen after directory-based metadata repair.\n{msg}"
        )


def matrix_has_english(matrix: pd.DataFrame) -> bool:
    if not {"task_lang", "transfer_lang"}.issubset(matrix.columns):
        return False

    return "eng" in set(matrix["transfer_lang"].astype(str))


def english_rows_for_dataset(
    per_fold_dataset: pd.DataFrame,
    matrix: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    if not matrix_has_english(matrix):
        return pd.DataFrame()

    score_col = infer_score_column(matrix)
    task, model = infer_task_model(dataset)

    if model == "unknown":
        return pd.DataFrame()

    working = matrix[["task_lang", "transfer_lang", score_col]].copy()
    working["task_lang"] = working["task_lang"].astype(str)
    working["transfer_lang"] = working["transfer_lang"].astype(str)
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    working = working.dropna(subset=[score_col])

    english = (
        working
        .query("transfer_lang == 'eng'")
        .groupby("task_lang", as_index=False)[score_col]
        .max()
        .rename(columns={"task_lang": "target_lang", score_col: "predicted_performance"})
    )
    english["target_lang"] = english["target_lang"].astype(str)

    target_oracle = (
        per_fold_dataset[
            [
                "target_lang",
                "actual_best_performance",
                "target_resource_level",
            ]
        ]
        .dropna(subset=["target_lang", "actual_best_performance"])
        .copy()
    )
    target_oracle["target_lang"] = target_oracle["target_lang"].astype(str)
    target_oracle["actual_best_performance"] = pd.to_numeric(
        target_oracle["actual_best_performance"],
        errors="coerce",
    )

    target_oracle = (
        target_oracle
        .groupby("target_lang", as_index=False)
        .agg(
            actual_best_performance=("actual_best_performance", "max"),
            target_resource_level=("target_resource_level", "first"),
        )
    )

    joined = target_oracle.merge(english, on="target_lang", how="inner")

    if joined.empty:
        return pd.DataFrame()

    joined["dataset"] = dataset
    joined["dataset_model"] = dataset
    joined["task"] = task
    joined["model"] = model
    joined["model_name"] = model_label(model)
    joined["metric"] = score_col
    joined["metric_name"] = metric_label(score_col)

    joined["method_id"] = "always_eng"
    joined["method"] = "Always English"
    joined["method_type"] = "English"
    joined["method_group"] = "English"

    joined["query_id"] = joined["dataset"].astype(str) + "::" + joined["target_lang"].astype(str)
    joined["target_id"] = joined["query_id"]

    joined["predicted_best_source"] = "eng"
    joined["predicted_source_resource_level"] = "hrl"
    joined["actual_best_source"] = np.nan
    joined["actual_best_source_resource_level"] = np.nan

    actual = pd.to_numeric(joined["actual_best_performance"], errors="coerce")
    predicted = pd.to_numeric(joined["predicted_performance"], errors="coerce")
    denom = actual.where(actual > 0)

    joined["performance_loss"] = (actual - predicted) / denom
    joined["performance_loss_pct"] = 100.0 * joined["performance_loss"]

    joined = recompute_scale_columns(joined)

    return joined


def english_availability_table(
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, group in per_fold.groupby("dataset", dropna=False):
        dataset = str(dataset)
        matrix = data_matrices.get(dataset)

        n_targets = int(group["target_lang"].nunique())

        if matrix is None:
            rows.append(
                {
                    "dataset": dataset,
                    "task": group["task"].iloc[0],
                    "model": group["model_name"].iloc[0],
                    "english_available": False,
                    "n_targets_full": n_targets,
                    "n_targets_with_english": 0,
                    "used_in_with_english": False,
                    "reason": "missing data matrix",
                }
            )
            continue

        if not matrix_has_english(matrix):
            rows.append(
                {
                    "dataset": dataset,
                    "task": group["task"].iloc[0],
                    "model": group["model_name"].iloc[0],
                    "english_available": False,
                    "n_targets_full": n_targets,
                    "n_targets_with_english": 0,
                    "used_in_with_english": False,
                    "reason": "no English transfer source",
                }
            )
            continue

        english_rows = english_rows_for_dataset(group, matrix, dataset)
        n_english = int(english_rows["target_lang"].nunique()) if not english_rows.empty else 0

        rows.append(
            {
                "dataset": dataset,
                "task": group["task"].iloc[0],
                "model": group["model_name"].iloc[0],
                "english_available": n_english > 0,
                "n_targets_full": n_targets,
                "n_targets_with_english": n_english,
                "used_in_with_english": n_english > 0,
                "reason": "" if n_english > 0 else "English source has no shared target rows",
            }
        )

    return pd.DataFrame(rows)


def make_without_english_per_fold(per_fold: pd.DataFrame) -> pd.DataFrame:
    return clean_method_rows(per_fold.copy())


def make_with_english_per_fold(
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage = english_availability_table(per_fold, data_matrices)

    for dataset, group in per_fold.groupby("dataset", dropna=False):
        dataset = str(dataset)
        matrix = data_matrices.get(dataset)

        if matrix is None or not matrix_has_english(matrix):
            continue

        english_rows = english_rows_for_dataset(group, matrix, dataset)

        if english_rows.empty:
            continue

        english_targets = set(english_rows["target_lang"].astype(str))
        ranker_subset = group.loc[group["target_lang"].astype(str).isin(english_targets)].copy()

        if ranker_subset.empty:
            continue

        frames.append(ranker_subset)
        frames.append(english_rows)

    if not frames:
        return pd.DataFrame(), coverage

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = clean_method_rows(out)
    out = recompute_scale_columns(out)

    return out, coverage


def add_outcome_names(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "outcome" not in summary.columns:
        return summary

    out = summary.copy()
    out.insert(
        out.columns.get_loc("outcome") + 1,
        "outcome_name",
        out["outcome"].map(outcome_label),
    )

    return out


def make_main_metric_tables(
    per_fold: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    df = clean_method_rows(per_fold)

    if df.empty:
        write_error(folder / "main_metric_bootstrap_ci.csv", "no rows")
        write_error(folder / "pairwise_performance_loss_bootstrap.csv", "no rows")
        return

    group_cols = [
        "dataset",
        "task",
        "model",
        "model_name",
        "metric",
        "metric_name",
        "method",
        "method_type",
    ]
    value_cols = [c for c in REVIEWER_OUTCOMES if c in df.columns]

    summary = grouped_bootstrap_summary(
        df,
        group_cols=group_cols,
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = add_outcome_names(summary)
    write_csv(summary, folder / "main_metric_bootstrap_ci.csv")

    pairwise = paired_bootstrap_differences(
        df,
        group_cols=["dataset", "task", "model", "model_name", "metric", "metric_name"],
        method_col="method",
        value_col="performance_loss_pct",
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(pairwise, folder / "pairwise_performance_loss_bootstrap.csv")


def task_balanced_resource_summary_local(
    per_fold: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    if level not in {"method", "method_type"}:
        raise ValueError("level must be either 'method' or 'method_type'.")

    df = clean_method_rows(per_fold)
    df = df.query("target_resource_level in @RESOURCE_LEVELS").copy()

    if df.empty:
        return pd.DataFrame()

    if level == "method":
        id_cols = ["model", "model_name", "method", "method_type"]
    else:
        id_cols = ["model", "model_name", "method_type"]

    needed = [
        "dataset",
        "task",
        "model",
        "model_name",
        "metric",
        "metric_name",
        "target_resource_level",
        *id_cols,
        "target_lang",
        "performance_loss_pct",
        "raw_oracle_gap_points",
    ]
    needed = list(dict.fromkeys(needed))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns for resource summary: {missing}")

    cell_group = [
        "dataset",
        "task",
        "model",
        "model_name",
        "metric",
        "metric_name",
        *id_cols,
        "target_resource_level",
    ]
    cell_group = list(dict.fromkeys(cell_group))

    cell = (
        df[needed]
        .dropna(subset=["performance_loss_pct", "raw_oracle_gap_points"])
        .groupby(cell_group, dropna=False)
        .agg(
            n_targets=("target_lang", "nunique"),
            performance_loss_pct=("performance_loss_pct", "mean"),
            raw_oracle_gap_points=("raw_oracle_gap_points", "mean"),
        )
        .reset_index()
    )

    if cell.empty:
        return pd.DataFrame()

    rows = []

    for key, group in cell.groupby(id_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)

        base = dict(zip(id_cols, key))
        row: dict[str, object] = dict(base)

        for resource in RESOURCE_LEVELS:
            sub = group.query("target_resource_level == @resource")
            suffix = resource

            row[f"n_cells_{suffix}"] = int(sub["dataset"].nunique()) if not sub.empty else 0
            row[f"pl_{suffix}"] = (
                float(sub["performance_loss_pct"].mean()) if not sub.empty else np.nan
            )
            row[f"raw_gap_{suffix}"] = (
                float(sub["raw_oracle_gap_points"].mean()) if not sub.empty else np.nan
            )

        row["pl_mrl_minus_hrl"] = row.get("pl_mrl", np.nan) - row.get("pl_hrl", np.nan)
        row["pl_lrl_minus_hrl"] = row.get("pl_lrl", np.nan) - row.get("pl_hrl", np.nan)
        row["raw_gap_mrl_minus_hrl"] = row.get("raw_gap_mrl", np.nan) - row.get("raw_gap_hrl", np.nan)
        row["raw_gap_lrl_minus_hrl"] = row.get("raw_gap_lrl", np.nan) - row.get("raw_gap_hrl", np.nan)

        rows.append(row)

    out = pd.DataFrame(rows)

    if "model_name" in out.columns:
        out = out.drop(columns=["model"]).rename(columns={"model_name": "model"})

    if level == "method":
        ordered_cols = ["model", "method", "method_type"]
    else:
        ordered_cols = ["model", "method_type"]

    metric_cols = [
        "n_cells_hrl",
        "n_cells_mrl",
        "n_cells_lrl",
        "pl_hrl",
        "pl_mrl",
        "pl_lrl",
        "raw_gap_hrl",
        "raw_gap_mrl",
        "raw_gap_lrl",
        "pl_mrl_minus_hrl",
        "pl_lrl_minus_hrl",
        "raw_gap_mrl_minus_hrl",
        "raw_gap_lrl_minus_hrl",
    ]

    return out[ordered_cols + metric_cols]


def make_resource_tables(per_fold: pd.DataFrame, folder: Path) -> None:
    if per_fold.empty:
        write_error(folder / "task_balanced_resource_by_method.csv", "no rows")
        write_error(folder / "task_balanced_resource_by_method_type.csv", "no rows")
        write_error(folder / "target_resource_counts.csv", "no rows")
        return

    by_method = task_balanced_resource_summary_local(
        per_fold,
        level="method",
    )
    write_csv(by_method, folder / "task_balanced_resource_by_method.csv")

    by_type = task_balanced_resource_summary_local(
        per_fold,
        level="method_type",
    )
    write_csv(by_type, folder / "task_balanced_resource_by_method_type.csv")

    target_counts = (
        per_fold[["dataset", "task", "model_name", "target_lang", "target_resource_level"]]
        .drop_duplicates()
        .assign(
            is_hrl=lambda x: x["target_resource_level"].eq("hrl").astype(int),
            is_mrl=lambda x: x["target_resource_level"].eq("mrl").astype(int),
            is_lrl=lambda x: x["target_resource_level"].eq("lrl").astype(int),
            is_unknown=lambda x: (
                ~x["target_resource_level"].isin(RESOURCE_LEVELS)
            ).astype(int),
        )
        .groupby(["dataset", "task", "model_name"], dropna=False)
        .agg(
            hrl_targets=("is_hrl", "sum"),
            mrl_targets=("is_mrl", "sum"),
            lrl_targets=("is_lrl", "sum"),
            unknown_targets=("is_unknown", "sum"),
            total_targets=("target_lang", "nunique"),
        )
        .reset_index()
        .rename(columns={"model_name": "model"})
    )
    write_csv(target_counts, folder / "target_resource_counts.csv")


def lmm_term_label(term: object) -> str:
    term = str(term)

    mapping = {
        "Intercept": "Baseline: Individual, HRL, mT5",
        'C(method_type, Treatment(reference="Individual"))[T.Composite]': "Composite vs Individual, HRL",
        'C(method_type, Treatment(reference="Individual"))[T.Trained]': "Trained vs Individual, HRL",
        'C(method_type, Treatment(reference="Individual"))[T.English]': "English vs Individual, HRL",
        'C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "LRL vs HRL, Individual",
        'C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "MRL vs HRL, Individual",
        "C(model)[T.xlm-r]": "XLM-R vs mT5",
        'C(method_type, Treatment(reference="Individual"))[T.Composite]:C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "Composite × LRL",
        'C(method_type, Treatment(reference="Individual"))[T.Trained]:C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "Trained × LRL",
        'C(method_type, Treatment(reference="Individual"))[T.English]:C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "English × LRL",
        'C(method_type, Treatment(reference="Individual"))[T.Composite]:C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "Composite × MRL",
        'C(method_type, Treatment(reference="Individual"))[T.Trained]:C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "Trained × MRL",
        'C(method_type, Treatment(reference="Individual"))[T.English]:C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "English × MRL",
        "target Var": "Target variance",
    }

    return mapping.get(term, term)


def fit_lmm_table(
    per_fold: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": ["statsmodels is not installed"],
            }
        )

    needed = [
        outcome,
        "method_type",
        "target_resource_level",
        "model",
        "dataset",
        "target_id",
    ]
    missing = [c for c in needed if c not in per_fold.columns]
    if missing:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": [f"missing columns: {missing}"],
            }
        )

    model_df = (
        per_fold[needed]
        .dropna()
        .query("method_type in @MAIN_LMM_METHOD_TYPES")
        .query("target_resource_level in @RESOURCE_LEVELS")
        .copy()
    )

    if model_df.empty:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": ["no rows after filtering to main method types and hrl/mrl/lrl"],
            }
        )

    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df[np.isfinite(model_df[outcome])].copy()

    if model_df.empty:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": ["no finite outcome values"],
            }
        )

    formula = (
        f"{outcome} ~ "
        'C(method_type, Treatment(reference="Individual"))'
        ' * C(target_resource_level, Treatment(reference="hrl"))'
        ' + C(model)'
    )

    fit_note = "dataset random intercept plus target variance component"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["dataset"],
                vc_formula={"target": "0 + C(target_id)"},
            )
            result = model.fit(method="lbfgs", reml=True, maxiter=500, disp=False)
        except Exception:
            fit_note = "dataset random intercept only"
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["dataset"],
            )
            result = model.fit(method="lbfgs", reml=True, maxiter=500, disp=False)

    conf = result.conf_int()
    rows = []

    for raw_term in result.params.index:
        if raw_term.lower().startswith("group"):
            continue

        rows.append(
            {
                "outcome": outcome,
                "outcome_name": outcome_label(outcome),
                "coefficient": lmm_term_label(raw_term),
                "estimate": result.params.get(raw_term, np.nan),
                "se": result.bse.get(raw_term, np.nan),
                "ci_lower": conf.loc[raw_term, 0] if raw_term in conf.index else np.nan,
                "ci_upper": conf.loc[raw_term, 1] if raw_term in conf.index else np.nan,
                "p_value": result.pvalues.get(raw_term, np.nan),
                "n_rows": int(model_df.shape[0]),
                "n_datasets": int(model_df["dataset"].nunique()),
                "n_targets": int(model_df["target_id"].nunique()),
                "fit_note": fit_note,
                "raw_term": raw_term,
            }
        )

    return pd.DataFrame(rows)


def make_lmm_tables(per_fold: pd.DataFrame, folder: Path) -> None:
    if per_fold.empty:
        write_error(folder / "resource_lmm_fixed_effects.csv", "no rows")
        return

    frames = []
    for outcome in REVIEWER_OUTCOMES:
        if outcome in per_fold.columns:
            frames.append(fit_lmm_table(per_fold, outcome))

    if frames:
        write_csv(
            pd.concat(frames, ignore_index=True),
            folder / "resource_lmm_fixed_effects.csv",
        )
    else:
        write_error(folder / "resource_lmm_fixed_effects.csv", "no available outcomes")


def make_selection_opportunity_tables(
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
    folder: Path,
) -> None:
    if per_fold.empty:
        write_error(folder / "selection_opportunity_by_dataset_resource.csv", "no rows")
        write_error(folder / "selection_opportunity_task_balanced.csv", "no rows")
        return

    available_datasets = set(per_fold["dataset"].dropna().astype(str).unique())

    lookup = (
        per_fold[["dataset", "target_lang", "target_resource_level"]]
        .drop_duplicates()
        .copy()
    )
    lookup["dataset"] = lookup["dataset"].astype(str)
    lookup["target_lang"] = lookup["target_lang"].astype(str)

    rows = []

    for dataset, matrix in data_matrices.items():
        dataset = str(dataset)

        if dataset not in available_datasets:
            continue

        if not {"task_lang", "transfer_lang"}.issubset(matrix.columns):
            continue

        score_col = infer_score_column(matrix)
        task, model = infer_task_model(dataset)

        if model == "unknown":
            continue

        score = pd.to_numeric(matrix[score_col], errors="coerce")
        mult = score_multiplier(score)

        working = matrix[["task_lang", "transfer_lang", score_col]].copy()
        working["task_lang"] = working["task_lang"].astype(str)
        working["transfer_lang"] = working["transfer_lang"].astype(str)
        working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
        working = working.dropna(subset=[score_col])

        if working.empty:
            continue

        stats = (
            working
            .groupby("task_lang", as_index=False)
            .agg(
                oracle_score_raw=(score_col, "max"),
                median_source_score_raw=(score_col, "median"),
                source_score_sd_raw=(score_col, "std"),
                n_sources=("transfer_lang", "nunique"),
            )
            .rename(columns={"task_lang": "target_lang"})
        )

        stats["dataset"] = dataset
        stats["target_lang"] = stats["target_lang"].astype(str)

        stats = stats.merge(
            lookup,
            on=["dataset", "target_lang"],
            how="inner",
        )

        if stats.empty:
            continue

        stats["task"] = task
        stats["model"] = model_label(model)
        stats["metric"] = metric_label(score_col)
        stats["resource"] = stats["target_resource_level"].map(resource_label)
        stats["oracle_score"] = mult * stats["oracle_score_raw"]
        stats["source_sd"] = mult * stats["source_score_sd_raw"]
        stats["oracle_minus_median"] = mult * (
            stats["oracle_score_raw"] - stats["median_source_score_raw"]
        )

        rows.append(stats)

    if not rows:
        write_error(folder / "selection_opportunity_by_dataset_resource.csv", "no selection-opportunity rows found")
        write_error(folder / "selection_opportunity_task_balanced.csv", "no selection-opportunity rows found")
        return

    target_level = pd.concat(rows, ignore_index=True)

    dataset_resource = (
        target_level
        .query("target_resource_level in @RESOURCE_LEVELS")
        .groupby(["dataset", "task", "model", "metric", "resource"], dropna=False)
        .agg(
            oracle_score=("oracle_score", "mean"),
            source_sd=("source_sd", "mean"),
            oracle_minus_median=("oracle_minus_median", "mean"),
            mean_sources=("n_sources", "mean"),
            n_targets=("target_lang", "nunique"),
        )
        .reset_index()
    )
    write_csv(dataset_resource, folder / "selection_opportunity_by_dataset_resource.csv")

    task_balanced = (
        dataset_resource
        .groupby(["model", "resource"], dropna=False)
        .agg(
            oracle_score=("oracle_score", "mean"),
            source_sd=("source_sd", "mean"),
            oracle_minus_median=("oracle_minus_median", "mean"),
            mean_sources=("mean_sources", "mean"),
            n_dataset_cells=("dataset", "nunique"),
        )
        .reset_index()
    )
    write_csv(task_balanced, folder / "selection_opportunity_task_balanced.csv")


def make_ctc_tables(
    per_fold: pd.DataFrame,
    folder: Path,
    min_ctc_targets: int,
) -> None:
    present = [c for c in CTC_COLS if c in per_fold.columns]

    if per_fold.empty:
        write_error(folder / "ctc_resource_by_dataset_method.csv", "no rows")
        write_error(folder / "ctc_main_resource_by_method.csv", "no rows")
        return

    if not present:
        write_error(folder / "ctc_resource_by_dataset_method.csv", "no CTC columns found")
        write_error(folder / "ctc_main_resource_by_method.csv", "no CTC columns found")
        return

    ctc = clean_method_rows(per_fold)
    ctc = ctc.query("method_type != 'English'").copy()
    ctc = finite_numeric(ctc, present)
    ctc = ctc.query("target_resource_level in @RESOURCE_LEVELS").copy()

    if ctc.empty:
        write_error(folder / "ctc_resource_by_dataset_method.csv", "no CTC rows after filtering")
        write_error(folder / "ctc_main_resource_by_method.csv", "no CTC rows after filtering")
        return

    dataset_sizes = (
        ctc[["dataset", "target_lang"]]
        .drop_duplicates()
        .groupby("dataset", as_index=False)
        .agg(n_targets_total=("target_lang", "nunique"))
    )

    ctc = ctc.merge(dataset_sizes, on="dataset", how="left")
    ctc["large_enough_for_main_ctc"] = ctc["n_targets_total"] >= min_ctc_targets
    ctc["resource"] = ctc["target_resource_level"].map(resource_label)

    by_dataset_method = (
        ctc
        .groupby(
            [
                "dataset",
                "task",
                "model_name",
                "metric_name",
                "method",
                "method_type",
                "resource",
            ],
            dropna=False,
        )
        .agg(
            n_targets=("target_lang", "nunique"),
            trial_complexity=("cnotc_trial_complexity", "mean"),
            pool_fraction=("cnotc_pool_fraction", "mean"),
            near_oracle_coverage=("cnotc_near_oracle_coverage", "mean"),
            exact_best_coverage=("cnotc_exact_best_coverage", "mean"),
            best_in_set_pl=("cnotc_best_in_set_performance_loss", "mean"),
            large_enough_for_main_ctc=("large_enough_for_main_ctc", "max"),
        )
        .reset_index()
        .rename(columns={"model_name": "model", "metric_name": "metric"})
    )
    write_csv(by_dataset_method, folder / "ctc_resource_by_dataset_method.csv")

    main_by_method = (
        ctc
        .query("large_enough_for_main_ctc")
        .groupby(["model_name", "method", "method_type", "resource"], dropna=False)
        .agg(
            n_dataset_cells=("dataset", "nunique"),
            n_targets=("target_lang", "nunique"),
            trial_complexity=("cnotc_trial_complexity", "mean"),
            pool_fraction=("cnotc_pool_fraction", "mean"),
            near_oracle_coverage=("cnotc_near_oracle_coverage", "mean"),
            exact_best_coverage=("cnotc_exact_best_coverage", "mean"),
            best_in_set_pl=("cnotc_best_in_set_performance_loss", "mean"),
        )
        .reset_index()
        .rename(columns={"model_name": "model"})
    )
    write_csv(main_by_method, folder / "ctc_main_resource_by_method.csv")


def make_nnr_tables(
    root: Path,
    data_matrices: dict[str, pd.DataFrame],
    outdir: Path,
    n_bootstrap: int,
    seed: int,
    min_ctc_targets: int,
) -> None:
    nnr_base = load_per_fold_artifacts(root, nnrank=True)

    for variant in VARIANTS:
        folder = tables_dir(outdir, variant)

        if nnr_base.empty:
            write_error(folder / "nnrank_status.csv", "no NNRank artifacts found")
            continue

        assert_no_unknown_models(nnr_base, "NNRank artifacts")

        if variant == "with_english":
            nnr, coverage = make_with_english_per_fold(nnr_base, data_matrices)
            write_csv(coverage, folder / "nnrank_english_subset_coverage.csv")
        else:
            nnr = make_without_english_per_fold(nnr_base)

        write_csv(nnr, outdir / f"combined_nnr_per_fold_results_{variant}.csv")

        if nnr.empty:
            write_error(folder / "nnrank_restricted_metric_bootstrap_ci.csv", "no rows")
            write_error(folder / "nnrank_restricted_pairwise_bootstrap.csv", "no rows")
            write_error(folder / "nnrank_restricted_task_balanced_resource_by_method.csv", "no rows")
            write_error(folder / "nnrank_restricted_task_balanced_resource_by_method_type.csv", "no rows")
            continue

        group_cols = [
            "dataset",
            "task",
            "model",
            "model_name",
            "metric",
            "metric_name",
            "method",
            "method_type",
        ]
        value_cols = [c for c in REVIEWER_OUTCOMES if c in nnr.columns]

        summary = grouped_bootstrap_summary(
            nnr,
            group_cols=group_cols,
            value_cols=value_cols,
            unit_col="query_id",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        summary = add_outcome_names(summary)
        write_csv(summary, folder / "nnrank_restricted_metric_bootstrap_ci.csv")

        pairwise = paired_bootstrap_differences(
            nnr,
            group_cols=["dataset", "task", "model", "model_name", "metric", "metric_name"],
            method_col="method",
            value_col="performance_loss_pct",
            unit_col="query_id",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        write_csv(pairwise, folder / "nnrank_restricted_pairwise_bootstrap.csv")

        resource_by_method = task_balanced_resource_summary_local(nnr, level="method")
        write_csv(resource_by_method, folder / "nnrank_restricted_task_balanced_resource_by_method.csv")

        resource_by_type = task_balanced_resource_summary_local(nnr, level="method_type")
        write_csv(resource_by_type, folder / "nnrank_restricted_task_balanced_resource_by_method_type.csv")

        make_ctc_tables(
            per_fold=nnr,
            folder=folder,
            min_ctc_targets=min_ctc_targets,
        )


def write_variant_tables(
    variant: str,
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
    outdir: Path,
    n_bootstrap: int,
    seed: int,
    min_ctc_targets: int,
) -> None:
    folder = tables_dir(outdir, variant)

    write_csv(per_fold, outdir / f"combined_per_fold_results_{variant}.csv")

    make_main_metric_tables(
        per_fold=per_fold,
        folder=folder,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    make_resource_tables(
        per_fold=per_fold,
        folder=folder,
    )
    make_lmm_tables(
        per_fold=per_fold,
        folder=folder,
    )
    make_selection_opportunity_tables(
        per_fold=per_fold,
        data_matrices=data_matrices,
        folder=folder,
    )
    make_ctc_tables(
        per_fold=per_fold,
        folder=folder,
        min_ctc_targets=min_ctc_targets,
    )


def write_manifest(outdir: Path) -> None:
    manifest_rows = []

    for variant in VARIANTS:
        folder = tables_dir(outdir, variant)

        for path in sorted(folder.glob("*.csv")):
            manifest_rows.append(
                {
                    "variant": variant,
                    "table_file": str(path.relative_to(outdir)),
                }
            )

    for path in sorted(outdir.glob("combined*_per_fold_results_*.csv")):
        manifest_rows.append(
            {
                "variant": "combined",
                "table_file": str(path.relative_to(outdir)),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    write_csv(manifest, outdir / "table_manifest.csv")


def main() -> None:
    args = parse_args()

    outdir = ensure_dir(args.outdir)
    ensure_dir(outdir / "tables")
    for variant in VARIANTS:
        tables_dir(outdir, variant)

    base_per_fold = load_per_fold_artifacts(args.root, nnrank=False)
    if base_per_fold.empty:
        raise RuntimeError("No main per-fold artifacts found.")

    assert_no_unknown_models(base_per_fold, "Main artifacts")

    data_matrices = load_data_matrices(args.root)

    without_english = make_without_english_per_fold(base_per_fold)
    with_english, english_coverage = make_with_english_per_fold(
        base_per_fold,
        data_matrices,
    )

    write_csv(
        english_coverage,
        tables_dir(outdir, "with_english") / "english_subset_coverage.csv",
    )

    write_variant_tables(
        variant="without_english",
        per_fold=without_english,
        data_matrices=data_matrices,
        outdir=outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        min_ctc_targets=args.min_ctc_targets,
    )

    write_variant_tables(
        variant="with_english",
        per_fold=with_english,
        data_matrices=data_matrices,
        outdir=outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        min_ctc_targets=args.min_ctc_targets,
    )

    if args.include_nnrank:
        make_nnr_tables(
            root=args.root,
            data_matrices=data_matrices,
            outdir=outdir,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            min_ctc_targets=args.min_ctc_targets,
        )

    write_manifest(outdir)

    print(f"Wrote without-English tables to: {outdir / 'tables' / 'without_english'}")
    print(f"Wrote with-English tables to: {outdir / 'tables' / 'with_english'}")
    print(f"Wrote manifest to: {outdir / 'table_manifest.csv'}")


if __name__ == "__main__":
    main()