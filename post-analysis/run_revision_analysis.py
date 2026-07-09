#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    MAIN_LMM_METHOD_GROUPS,
    RESOURCE_LEVELS,
    ensure_dir,
    finite_numeric,
    grouped_bootstrap_summary,
    infer_score_column,
    infer_task_model,
    load_data_matrices,
    paired_bootstrap_differences,
    target_resource_lookup,
    task_balanced_resource_summary,
    write_csv,
)


REVIEWER_OUTCOMES = [
    "performance_loss_pct",
    "raw_oracle_gap_points",
]

ENGLISH_OUTCOMES = [
    "performance_loss_pct",
    "raw_oracle_gap_points",
    "actual_best_performance_points",
    "predicted_performance_points",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concise no-retraining reviewer-response tables for LangRankPlus."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("post-analysis/outputs"))
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-nnrank", action="store_true")
    parser.add_argument("--min-ctc-targets", type=int, default=50)
    return parser.parse_args()


def necessary_dir(outdir: Path) -> Path:
    return ensure_dir(outdir / "tables" / "necessary")


def other_dir(outdir: Path) -> Path:
    return ensure_dir(outdir / "tables" / "other")


def classify_method(method: object) -> str:
    name = str(method).strip()
    lower = name.lower()

    if lower == "random":
        return "random"

    if "nnrank" in lower:
        return "nnrank"

    if lower in DISTANCE_METHODS or lower.startswith("single_"):
        return "individual"

    if "composite" in lower or "rrf" in lower or lower in {"equal", "rrf_equal"}:
        return "composite"

    if (
        "lightgbm" in lower
        or "lambdarank" in lower
        or "lambda" in lower
        or "mlp" in lower
        or "listnet" in lower
    ):
        return "trained"

    return "other"


def metric_from_per_fold_path(path: Path) -> str:
    stem = path.stem

    if stem.startswith("per_fold_"):
        return stem.removeprefix("per_fold_")

    return stem


def score_multiplier(values: pd.Series) -> float:
    """
    Convert score differences to metric points.

    If scores are already in point units, for example F1 in [0, 100],
    leave differences unchanged. If scores are proportions, for example
    F1 in [0, 1], multiply differences by 100.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return 100.0

    if float(clean.max()) > 1.5:
        return 1.0

    return 100.0


def clean_method_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "method_group" not in df.columns:
        return df.copy()

    return df.query("method_group != 'random'").copy()


def standardise_artifact_metadata(
    df: pd.DataFrame,
    artifact_dataset: str,
    metric: str,
) -> pd.DataFrame:
    """
    Force metadata to be derived from the artifact directory name.

    This avoids cases where an internal CSV column says dataset='opus100'
    even though the artifact directory is 'opus100_mt5'. The latter is the
    reliable dataset-model identifier.
    """
    out = df.copy()

    task, model = infer_task_model(artifact_dataset)

    out["dataset"] = artifact_dataset
    out["dataset_model"] = artifact_dataset
    out["task"] = task
    out["model"] = model
    out["metric"] = metric

    if "method" not in out.columns:
        raise ValueError(f"Missing required column 'method' in artifact {artifact_dataset}.")

    out["method"] = out["method"].astype(str)
    out["method_group"] = out["method"].map(classify_method)

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


def load_per_fold_artifacts(root: Path, nnrank: bool = False) -> pd.DataFrame:
    """
    Load per-fold artifact CSVs directly from the artifact directory tree.

    For main artifacts:
      root/artifacts/<dataset_model>/per_fold_*.csv

    For NNRank artifacts:
      root/artifacts/nnrank/<dataset_model>/per_fold_*.csv

    The dataset, task, and model fields are always inferred from <dataset_model>.
    Internal CSV metadata cannot create model='unknown'.
    """
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


def recompute_scale_columns(per_fold: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute all scale-sensitive columns inside this driver.

    This prevents stale or incorrect scaling from external helpers from
    entering raw-gap, English-baseline, resource, or LMM tables.
    """
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


def make_main_metric_tables(
    per_fold: pd.DataFrame,
    outdir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    necessary = necessary_dir(outdir)
    other = other_dir(outdir)

    df = clean_method_rows(per_fold)

    group_cols = ["dataset", "task", "model", "metric", "method", "method_group"]
    value_cols = [c for c in REVIEWER_OUTCOMES if c in df.columns]

    summary = grouped_bootstrap_summary(
        df,
        group_cols=group_cols,
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(summary, necessary / "main_metric_bootstrap_ci.csv")

    pairwise = paired_bootstrap_differences(
        df,
        group_cols=["dataset", "task", "model", "metric"],
        method_col="method",
        value_col="performance_loss_pct",
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(pairwise, other / "pairwise_performance_loss_bootstrap.csv")


def make_resource_tables(per_fold: pd.DataFrame, outdir: Path) -> None:
    necessary = necessary_dir(outdir)

    df = clean_method_rows(per_fold)

    resource_by_method = task_balanced_resource_summary(
        df,
        method_level="method",
    )
    write_csv(resource_by_method, necessary / "task_balanced_resource_by_method.csv")

    target_counts = (
        per_fold[["dataset", "task", "model", "target_lang", "target_resource_level"]]
        .drop_duplicates()
        .assign(
            is_hrl=lambda x: x["target_resource_level"].eq("hrl").astype(int),
            is_mrl=lambda x: x["target_resource_level"].eq("mrl").astype(int),
            is_lrl=lambda x: x["target_resource_level"].eq("lrl").astype(int),
            is_unknown=lambda x: (
                ~x["target_resource_level"].isin(RESOURCE_LEVELS)
            ).astype(int),
        )
        .groupby(["dataset", "task", "model"], dropna=False)
        .agg(
            hrl_targets=("is_hrl", "sum"),
            mrl_targets=("is_mrl", "sum"),
            lrl_targets=("is_lrl", "sum"),
            unknown_targets=("is_unknown", "sum"),
            total_targets=("target_lang", "nunique"),
        )
        .reset_index()
    )
    write_csv(target_counts, necessary / "target_resource_counts.csv")


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
                "error": ["statsmodels is not installed"],
            }
        )

    needed = [
        outcome,
        "method_group",
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
                "error": [f"missing columns: {missing}"],
            }
        )

    model_df = (
        per_fold[needed]
        .dropna()
        .query("method_group in @MAIN_LMM_METHOD_GROUPS")
        .query("target_resource_level in @RESOURCE_LEVELS")
        .copy()
    )

    if model_df.empty:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "error": ["no rows after filtering to main method groups and hrl/mrl/lrl"],
            }
        )

    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df[np.isfinite(model_df[outcome])].copy()

    if model_df.empty:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "error": ["no finite outcome values"],
            }
        )

    formula = (
        f"{outcome} ~ "
        'C(method_group, Treatment(reference="individual"))'
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

    for term in result.params.index:
        if term.lower().startswith("group"):
            continue

        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "estimate": result.params.get(term, np.nan),
                "se": result.bse.get(term, np.nan),
                "ci_lower": conf.loc[term, 0] if term in conf.index else np.nan,
                "ci_upper": conf.loc[term, 1] if term in conf.index else np.nan,
                "p_value": result.pvalues.get(term, np.nan),
                "n_rows": int(model_df.shape[0]),
                "n_datasets": int(model_df["dataset"].nunique()),
                "n_targets": int(model_df["target_id"].nunique()),
                "fit_note": fit_note,
            }
        )

    return pd.DataFrame(rows)


def make_lmm_tables(per_fold: pd.DataFrame, outdir: Path) -> None:
    necessary = necessary_dir(outdir)

    frames = []
    for outcome in REVIEWER_OUTCOMES:
        if outcome in per_fold.columns:
            frames.append(fit_lmm_table(per_fold, outcome))

    if frames:
        write_csv(
            pd.concat(frames, ignore_index=True),
            necessary / "resource_lmm_fixed_effects.csv",
        )


def make_english_baseline_tables(
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
    outdir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    necessary = necessary_dir(outdir)

    available_datasets = set(per_fold["dataset"].dropna().unique())
    lookup = target_resource_lookup(per_fold)

    english_rows = []

    for dataset, matrix in data_matrices.items():
        if dataset not in available_datasets:
            continue

        if not {"task_lang", "transfer_lang"}.issubset(matrix.columns):
            continue

        if "eng" not in set(matrix["transfer_lang"].astype(str)):
            continue

        score_col = infer_score_column(matrix)
        task, model = infer_task_model(dataset)

        if model == "unknown":
            continue

        score = pd.to_numeric(matrix[score_col], errors="coerce")
        mult = score_multiplier(score)

        working = matrix[["task_lang", "transfer_lang", score_col]].copy()
        working[score_col] = pd.to_numeric(working[score_col], errors="coerce")

        oracle = (
            working
            .dropna(subset=[score_col])
            .groupby("task_lang", as_index=False)[score_col]
            .max()
            .rename(columns={score_col: "actual_best_performance"})
        )

        english = (
            working
            .query("transfer_lang == 'eng'")
            .dropna(subset=[score_col])
            [["task_lang", score_col]]
            .rename(columns={score_col: "predicted_performance"})
        )

        joined = oracle.merge(english, on="task_lang", how="inner")
        if joined.empty:
            continue

        joined["dataset"] = dataset
        joined["dataset_model"] = dataset
        joined["task"] = task
        joined["model"] = model
        joined["metric"] = score_col
        joined["method"] = "always_eng"
        joined["method_group"] = "baseline"
        joined["target_lang"] = joined["task_lang"].astype(str)
        joined["query_id"] = joined["dataset"].astype(str) + "::" + joined["target_lang"]
        joined["target_id"] = joined["query_id"]

        joined["performance_loss_pct"] = (
            100.0
            * (joined["actual_best_performance"] - joined["predicted_performance"])
            / joined["actual_best_performance"]
        )
        joined["raw_oracle_gap_points"] = (
            mult * (joined["actual_best_performance"] - joined["predicted_performance"])
        )
        joined["actual_best_performance_points"] = mult * joined["actual_best_performance"]
        joined["predicted_performance_points"] = mult * joined["predicted_performance"]

        joined = joined.merge(
            lookup,
            on=["dataset", "target_lang"],
            how="left",
        )

        english_rows.append(joined)

    if not english_rows:
        write_csv(
            pd.DataFrame({"error": ["no English baseline rows found"]}),
            necessary / "english_baseline_comparison_bootstrap.csv",
        )
        return

    english_target = pd.concat(english_rows, ignore_index=True)
    shared_targets = english_target[["dataset", "target_lang"]].drop_duplicates()

    ranker_shared = (
        clean_method_rows(per_fold)
        .merge(shared_targets, on=["dataset", "target_lang"], how="inner")
    )

    common_cols = ranker_shared.columns.intersection(english_target.columns).tolist()

    comparable = pd.concat(
        [
            english_target[common_cols],
            ranker_shared,
        ],
        ignore_index=True,
        sort=False,
    )

    value_cols = [c for c in ENGLISH_OUTCOMES if c in comparable.columns]

    summary = grouped_bootstrap_summary(
        comparable,
        group_cols=["dataset", "task", "model", "metric", "method", "method_group"],
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    write_csv(summary, necessary / "english_baseline_comparison_bootstrap.csv")


def make_selection_opportunity_tables(
    per_fold: pd.DataFrame,
    data_matrices: dict[str, pd.DataFrame],
    outdir: Path,
) -> None:
    necessary = necessary_dir(outdir)
    other = other_dir(outdir)

    available_datasets = set(per_fold["dataset"].dropna().unique())
    lookup = target_resource_lookup(per_fold)

    rows = []

    for dataset, matrix in data_matrices.items():
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
        working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
        working = working.dropna(subset=[score_col])

        if working.empty:
            continue

        stats = (
            working
            .groupby("task_lang", as_index=False)
            .agg(
                oracle_score=(score_col, "max"),
                median_source_score=(score_col, "median"),
                source_score_sd=(score_col, "std"),
                n_sources=("transfer_lang", "nunique"),
            )
        )

        stats["dataset"] = dataset
        stats["dataset_model"] = dataset
        stats["task"] = task
        stats["model"] = model
        stats["metric"] = score_col
        stats["target_lang"] = stats["task_lang"].astype(str)
        stats["oracle_score_points"] = mult * stats["oracle_score"]
        stats["source_score_sd_points"] = mult * stats["source_score_sd"]
        stats["oracle_minus_median_points"] = mult * (
            stats["oracle_score"] - stats["median_source_score"]
        )

        stats = stats.merge(
            lookup,
            on=["dataset", "target_lang"],
            how="left",
        )

        rows.append(stats)

    if not rows:
        write_csv(
            pd.DataFrame({"error": ["no selection-opportunity rows found"]}),
            necessary / "selection_opportunity_task_balanced.csv",
        )
        return

    target_level = pd.concat(rows, ignore_index=True)

    dataset_resource = (
        target_level
        .query("target_resource_level in @RESOURCE_LEVELS")
        .groupby(
            ["dataset", "task", "model", "metric", "target_resource_level"],
            dropna=False,
        )
        .agg(
            oracle_score_points=("oracle_score_points", "mean"),
            source_score_sd_points=("source_score_sd_points", "mean"),
            oracle_minus_median_points=("oracle_minus_median_points", "mean"),
            mean_n_sources=("n_sources", "mean"),
            n_targets=("target_lang", "nunique"),
        )
        .reset_index()
    )
    write_csv(dataset_resource, other / "selection_opportunity_by_dataset_resource.csv")

    task_balanced = (
        dataset_resource
        .groupby(["model", "target_resource_level"], dropna=False)
        .agg(
            oracle_score_points=("oracle_score_points", "mean"),
            source_score_sd_points=("source_score_sd_points", "mean"),
            oracle_minus_median_points=("oracle_minus_median_points", "mean"),
            mean_n_sources=("mean_n_sources", "mean"),
            n_dataset_cells=("dataset", "nunique"),
        )
        .reset_index()
    )
    write_csv(task_balanced, necessary / "selection_opportunity_task_balanced.csv")


def make_ctc_tables(
    per_fold: pd.DataFrame,
    outdir: Path,
    min_ctc_targets: int,
) -> None:
    necessary = necessary_dir(outdir)
    other = other_dir(outdir)

    present = [c for c in CTC_COLS if c in per_fold.columns]
    if not present:
        write_csv(
            pd.DataFrame({"error": ["no CTC columns found"]}),
            necessary / "ctc_main_resource_by_method.csv",
        )
        return

    ctc = finite_numeric(clean_method_rows(per_fold), present)
    ctc = ctc.query("target_resource_level in @RESOURCE_LEVELS").copy()

    if ctc.empty:
        write_csv(
            pd.DataFrame({"error": ["no CTC rows after resource filtering"]}),
            necessary / "ctc_main_resource_by_method.csv",
        )
        return

    dataset_sizes = (
        ctc[["dataset", "target_lang"]]
        .drop_duplicates()
        .groupby("dataset", as_index=False)
        .agg(n_targets_total=("target_lang", "nunique"))
    )

    ctc = ctc.merge(dataset_sizes, on="dataset", how="left")
    ctc["large_enough_for_main_ctc"] = ctc["n_targets_total"] >= min_ctc_targets

    by_dataset_method = (
        ctc
        .groupby(
            [
                "dataset",
                "task",
                "model",
                "metric",
                "method",
                "method_group",
                "target_resource_level",
            ],
            dropna=False,
        )
        .agg(
            n_targets=("target_lang", "nunique"),
            cnotc_trial_complexity=("cnotc_trial_complexity", "mean"),
            cnotc_pool_fraction=("cnotc_pool_fraction", "mean"),
            cnotc_near_oracle_coverage=("cnotc_near_oracle_coverage", "mean"),
            cnotc_exact_best_coverage=("cnotc_exact_best_coverage", "mean"),
            cnotc_best_in_set_performance_loss=("cnotc_best_in_set_performance_loss", "mean"),
            large_enough_for_main_ctc=("large_enough_for_main_ctc", "max"),
        )
        .reset_index()
    )
    write_csv(by_dataset_method, other / "ctc_resource_by_dataset_method.csv")

    main_by_method = (
        ctc
        .query("large_enough_for_main_ctc")
        .groupby(["model", "method", "method_group", "target_resource_level"], dropna=False)
        .agg(
            n_dataset_cells=("dataset", "nunique"),
            n_targets=("target_lang", "nunique"),
            cnotc_trial_complexity=("cnotc_trial_complexity", "mean"),
            cnotc_pool_fraction=("cnotc_pool_fraction", "mean"),
            cnotc_near_oracle_coverage=("cnotc_near_oracle_coverage", "mean"),
            cnotc_exact_best_coverage=("cnotc_exact_best_coverage", "mean"),
            cnotc_best_in_set_performance_loss=("cnotc_best_in_set_performance_loss", "mean"),
        )
        .reset_index()
    )
    write_csv(main_by_method, necessary / "ctc_main_resource_by_method.csv")


def make_nnr_tables(
    root: Path,
    outdir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    nnr = load_per_fold_artifacts(root, nnrank=True)
    necessary = necessary_dir(outdir)
    other = other_dir(outdir)

    if nnr.empty:
        write_csv(
            pd.DataFrame({"error": ["no NNRank artifacts found"]}),
            other / "nnrank_status.csv",
        )
        return

    assert_no_unknown_models(nnr, "NNRank artifacts")

    nnr = clean_method_rows(nnr)

    write_csv(nnr, outdir / "combined_nnr_per_fold_results.csv")

    group_cols = ["dataset", "task", "model", "metric", "method", "method_group"]
    value_cols = [c for c in REVIEWER_OUTCOMES if c in nnr.columns]

    summary = grouped_bootstrap_summary(
        nnr,
        group_cols=group_cols,
        value_cols=value_cols,
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(summary, necessary / "nnrank_restricted_metric_bootstrap_ci.csv")

    resource = task_balanced_resource_summary(nnr, method_level="method")
    write_csv(resource, necessary / "nnrank_restricted_task_balanced_resource.csv")

    pairwise = paired_bootstrap_differences(
        nnr,
        group_cols=["dataset", "task", "model", "metric"],
        method_col="method",
        value_col="performance_loss_pct",
        unit_col="query_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(pairwise, other / "nnrank_restricted_pairwise_bootstrap.csv")


def write_manifest(outdir: Path) -> None:
    necessary = necessary_dir(outdir)
    other = other_dir(outdir)

    manifest_rows = []

    for category, directory in [
        ("necessary", necessary),
        ("other", other),
    ]:
        for path in sorted(directory.glob("*.csv")):
            manifest_rows.append(
                {
                    "category": category,
                    "table_file": str(path.relative_to(outdir)),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    write_csv(manifest, outdir / "table_manifest.csv")


def main() -> None:
    args = parse_args()

    outdir = ensure_dir(args.outdir)
    ensure_dir(outdir / "tables")
    necessary_dir(outdir)
    other_dir(outdir)

    per_fold = load_per_fold_artifacts(args.root, nnrank=False)
    if per_fold.empty:
        raise RuntimeError("No main per-fold artifacts found.")

    assert_no_unknown_models(per_fold, "Main artifacts")

    write_csv(per_fold, outdir / "combined_per_fold_results.csv")

    data_matrices = load_data_matrices(args.root)

    make_main_metric_tables(
        per_fold=per_fold,
        outdir=outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    make_resource_tables(
        per_fold=per_fold,
        outdir=outdir,
    )
    make_lmm_tables(
        per_fold=per_fold,
        outdir=outdir,
    )
    make_english_baseline_tables(
        per_fold=per_fold,
        data_matrices=data_matrices,
        outdir=outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    make_selection_opportunity_tables(
        per_fold=per_fold,
        data_matrices=data_matrices,
        outdir=outdir,
    )
    make_ctc_tables(
        per_fold=per_fold,
        outdir=outdir,
        min_ctc_targets=args.min_ctc_targets,
    )

    if args.include_nnrank:
        make_nnr_tables(
            root=args.root,
            outdir=outdir,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    write_manifest(outdir)

    print(f"Wrote combined main data to: {outdir / 'combined_per_fold_results.csv'}")
    print(f"Wrote necessary tables to: {outdir / 'tables' / 'necessary'}")
    print(f"Wrote other tables to: {outdir / 'tables' / 'other'}")
    print(f"Wrote manifest to: {outdir / 'table_manifest.csv'}")


if __name__ == "__main__":
    main()