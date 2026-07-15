#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    "cnotc_group_calibration_size",
    "cnotc_finite_threshold",
    "cnotc_full_pool_fallback",
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
    "las": "F1/LAS",
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

RESOURCE_LEVELS = ["hrl", "mrl", "lrl"]

VARIANTS = [
    "with_english",
    "without_english",
]

SETUP_COLS = [
    "dataset",
    "task",
    "model",
    "model_name",
    "metric",
    "metric_name",
]

FIXED_ENGLISH_PAIRS = [
    ("Composite-Equal", "Always English"),
    ("LightGBM", "Always English"),
]

PARADIGM_METHODS = {
    "Individual": [
        "Genetic",
        "Typological",
        "Geographic",
        "Script",
        "ASJP",
        "Wikipedia size",
    ],
    "Composite": [
        "Composite-Equal",
        "Composite-RRF",
    ],
    "Trained": [
        "LightGBM",
        "MLP",
    ],
}

PARADIGM_PAIRS = [
    ("Composite", "Individual"),
    ("Trained", "Individual"),
    ("Trained", "Composite"),
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def infer_metric_from_filename(path: str | Path) -> str:
    name = Path(path).name
    match = re.match(r"per_fold_(.+)\.csv$", name)
    if match:
        return match.group(1)
    match = re.match(r"summary_(.+)\.csv$", name)
    if match:
        return match.group(1)
    match = re.match(r"pairwise_(.+)\.csv$", name)
    if match:
        return match.group(1)
    return "unknown"


def infer_task_model(dataset: str) -> tuple[str, str]:
    if dataset.endswith("_xlm-r"):
        return dataset.removesuffix("_xlm-r"), "xlm-r"
    if dataset.endswith("_mt5"):
        return dataset.removesuffix("_mt5"), "mt5"
    if dataset.endswith("_llm"):
        return dataset.removesuffix("_llm"), "llm"
    if dataset.endswith("_decoder"):
        return dataset.removesuffix("_decoder"), "decoder"
    return dataset, "unknown"


def infer_score_column(df: pd.DataFrame) -> str:
    candidates = [
        "f1_score",
        "bleu",
        "accuracy",
        "las",
        "score",
        "performance",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    excluded = {
        "new_gen",
        "new_typ",
        "new_geo",
        "script",
        "distals_asjp",
        "distals_wiki_size",
    }
    numeric_cols = [c for c in numeric_cols if c not in excluded]
    if not numeric_cols:
        raise ValueError("Could not infer score column.")
    return numeric_cols[0]


def load_data_matrices(root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    out: dict[str, pd.DataFrame] = {}

    for path in sorted((root / "data").glob("*.csv")):
        dataset = path.stem
        df = read_csv(path)
        df["dataset"] = dataset
        task, model = infer_task_model(dataset)
        df["task"] = task
        df["model"] = model
        df["data_path"] = str(path)
        out[dataset] = df

    return out


def target_resource_lookup(per_fold: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "target_lang", "target_resource_level"]
    if not set(cols).issubset(per_fold.columns):
        return pd.DataFrame(columns=cols)

    lookup = per_fold[cols].dropna().drop_duplicates().copy()
    lookup = lookup.loc[
        lookup["target_resource_level"].isin(RESOURCE_LEVELS)
    ].copy()

    duplicated = lookup.duplicated(["dataset", "target_lang"], keep=False)
    if duplicated.any():
        lookup = (
            lookup
            .sort_values(["dataset", "target_lang", "target_resource_level"])
            .drop_duplicates(["dataset", "target_lang"], keep="first")
        )

    return lookup


def unit_level_values(
    df: pd.DataFrame,
    value_col: str,
    unit_col: str,
) -> np.ndarray:
    values = (
        df[[unit_col, value_col]]
        .dropna()
        .groupby(unit_col, as_index=False)[value_col]
        .mean()[value_col]
        .to_numpy(dtype=float)
    )
    return values


def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return math.nan, math.nan, math.nan, 0

    mean = float(np.mean(arr))

    if arr.size == 1 or n_bootstrap <= 0:
        return mean, math.nan, math.nan, int(arr.size)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot = arr[idx].mean(axis=1)

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi), int(arr.size)


def grouped_bootstrap_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    unit_col: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        for value_col in value_cols:
            if value_col not in group.columns:
                continue

            values = unit_level_values(group, value_col=value_col, unit_col=unit_col)
            mean, lo, hi, n_units = bootstrap_mean_ci(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            row = {
                **base,
                "outcome": value_col,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_units": n_units,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def paired_bootstrap_differences(
    df: pd.DataFrame,
    group_cols: list[str],
    method_col: str,
    value_col: str,
    unit_col: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        pivot = (
            group[[unit_col, method_col, value_col]]
            .dropna()
            .groupby([unit_col, method_col], as_index=False)[value_col]
            .mean()
            .pivot(index=unit_col, columns=method_col, values=value_col)
        )

        methods = list(pivot.columns)

        for method_a, method_b in itertools.combinations(methods, 2):
            pair = pivot[[method_a, method_b]].dropna()
            if pair.empty:
                continue

            diff = pair[method_a].to_numpy(dtype=float) - pair[method_b].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]

            if diff.size == 0:
                continue

            mean = float(np.mean(diff))

            if diff.size == 1 or n_bootstrap <= 0:
                lo = math.nan
                hi = math.nan
                p_boot = math.nan
            else:
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, diff.size, size=(n_bootstrap, diff.size))
                boot = diff[idx].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                p_boot = bootstrap_two_sided_p_value(boot)

            rows.append(
                {
                    **base,
                    "method_a": method_a,
                    "method_b": method_b,
                    "outcome": value_col,
                    "mean_difference_a_minus_b": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "bootstrap_p_value": p_boot,
                    "n_paired_units": int(diff.size),
                }
            )

    return pd.DataFrame(rows)


def bootstrap_two_sided_p_value(
    bootstrap_values: np.ndarray,
    null_value: float = 0.0,
) -> float:
    """
    Compute a two-sided sign-based bootstrap p-value with a plus-one correction.
    """
    boot = np.asarray(bootstrap_values, dtype=float)
    boot = boot[np.isfinite(boot)]

    if boot.size == 0:
        return math.nan

    lower = (np.count_nonzero(boot <= null_value) + 1.0) / (boot.size + 1.0)
    upper = (np.count_nonzero(boot >= null_value) + 1.0) / (boot.size + 1.0)

    return float(min(1.0, 2.0 * min(lower, upper)))


def paired_bootstrap_selected_pairs(
    df: pd.DataFrame,
    group_cols: list[str],
    method_col: str,
    value_col: str,
    unit_col: str,
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run paired target-level bootstrap tests for pre-specified method pairs.

    The reported difference is method_a minus method_b. For loss outcomes,
    a negative value therefore favours method_a.
    """
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        pivot = (
            group[[unit_col, method_col, value_col]]
            .dropna()
            .groupby([unit_col, method_col], as_index=False)[value_col]
            .mean()
            .pivot(index=unit_col, columns=method_col, values=value_col)
        )

        for method_a, method_b in pairs:
            if method_a not in pivot.columns or method_b not in pivot.columns:
                continue

            pair = pivot[[method_a, method_b]].dropna()
            if pair.empty:
                continue

            diff = (
                pair[method_a].to_numpy(dtype=float)
                - pair[method_b].to_numpy(dtype=float)
            )
            diff = diff[np.isfinite(diff)]

            if diff.size == 0:
                continue

            mean = float(np.mean(diff))

            if diff.size == 1 or n_bootstrap <= 0:
                lo = math.nan
                hi = math.nan
                p_boot = math.nan
            else:
                boot = _bootstrap_means_chunked(
                    diff,
                    n_draws=n_bootstrap,
                    rng=rng,
                )
                lo, hi = np.quantile(boot, [0.025, 0.975])
                p_boot = bootstrap_two_sided_p_value(boot)

            rows.append(
                {
                    **base,
                    "comparison": f"{method_a} - {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "outcome": value_col,
                    "mean_difference_a_minus_b": mean,
                    "ci_lower": float(lo) if np.isfinite(lo) else math.nan,
                    "ci_upper": float(hi) if np.isfinite(hi) else math.nan,
                    "bootstrap_p_value": p_boot,
                    "n_paired_units": int(diff.size),
                }
            )

    return pd.DataFrame(rows)


def _bootstrap_means_chunked(
    values: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    chunk_size: int = 2000,
) -> np.ndarray:
    """
    Generate bootstrap sample means without allocating one very large index array.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if n_draws <= 0:
        return np.empty(0, dtype=float)

    if arr.size == 0:
        return np.full(n_draws, np.nan, dtype=float)

    if arr.size == 1:
        return np.full(n_draws, float(arr[0]), dtype=float)

    out = np.empty(n_draws, dtype=float)

    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        idx = rng.integers(
            0,
            arr.size,
            size=(stop - start, arr.size),
        )
        out[start:stop] = arr[idx].mean(axis=1)

    return out


def hierarchical_paired_bootstrap_differences(
    df: pd.DataFrame,
    setup_cols: list[str],
    method_col: str,
    value_col: str,
    unit_col: str,
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compute task-balanced paired comparisons across dataset-model setups.

    Point estimate:
      1. Compute paired target-level differences within each setup.
      2. Average within each setup.
      3. Average setup means equally.

    Bootstrap:
      1. Resample setups with replacement.
      2. Independently resample paired target differences within every sampled setup.
      3. Average the sampled setup means equally.

    The reported difference is method_a minus method_b. For loss outcomes,
    negative values favour method_a.
    """
    rows: list[dict[str, object]] = []

    if df.empty:
        return pd.DataFrame()

    grouped_setups = list(df.groupby(setup_cols, dropna=False, sort=False))

    for pair_index, (method_a, method_b) in enumerate(pairs):
        setup_differences: list[np.ndarray] = []
        setup_names: list[str] = []

        for keys, group in grouped_setups:
            if not isinstance(keys, tuple):
                keys = (keys,)

            pivot = (
                group[[unit_col, method_col, value_col]]
                .dropna()
                .groupby([unit_col, method_col], as_index=False)[value_col]
                .mean()
                .pivot(index=unit_col, columns=method_col, values=value_col)
            )

            if method_a not in pivot.columns or method_b not in pivot.columns:
                continue

            pair = pivot[[method_a, method_b]].dropna()
            if pair.empty:
                continue

            diff = (
                pair[method_a].to_numpy(dtype=float)
                - pair[method_b].to_numpy(dtype=float)
            )
            diff = diff[np.isfinite(diff)]

            if diff.size == 0:
                continue

            setup_differences.append(diff)
            setup_names.append(" | ".join(str(x) for x in keys))

        if not setup_differences:
            continue

        setup_means = np.asarray(
            [float(np.mean(diff)) for diff in setup_differences],
            dtype=float,
        )
        estimate = float(np.mean(setup_means))
        n_setups = len(setup_differences)
        target_counts = np.asarray(
            [diff.size for diff in setup_differences],
            dtype=int,
        )

        if n_bootstrap <= 0:
            lo = math.nan
            hi = math.nan
            p_boot = math.nan
        elif n_setups == 1:
            rng = np.random.default_rng(seed + 100003 * pair_index)
            boot = _bootstrap_means_chunked(
                setup_differences[0],
                n_draws=n_bootstrap,
                rng=rng,
            )
            lo, hi = np.quantile(boot, [0.025, 0.975])
            p_boot = bootstrap_two_sided_p_value(boot)
        else:
            rng = np.random.default_rng(seed + 100003 * pair_index)
            bootstrap_sum = np.zeros(n_bootstrap, dtype=float)

            # Each position in the resampled setup list is generated separately.
            # Repeated selections of the same setup therefore receive independent
            # within-setup target resamples.
            for _ in range(n_setups):
                selected = rng.integers(0, n_setups, size=n_bootstrap)
                position_means = np.empty(n_bootstrap, dtype=float)

                for setup_index, diff in enumerate(setup_differences):
                    locations = np.flatnonzero(selected == setup_index)
                    if locations.size == 0:
                        continue

                    position_means[locations] = _bootstrap_means_chunked(
                        diff,
                        n_draws=int(locations.size),
                        rng=rng,
                    )

                bootstrap_sum += position_means

            boot = bootstrap_sum / float(n_setups)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            p_boot = bootstrap_two_sided_p_value(boot)

        rows.append(
            {
                "comparison": f"{method_a} - {method_b}",
                "method_a": method_a,
                "method_b": method_b,
                "outcome": value_col,
                "mean_difference_a_minus_b": estimate,
                "ci_lower": float(lo) if np.isfinite(lo) else math.nan,
                "ci_upper": float(hi) if np.isfinite(hi) else math.nan,
                "bootstrap_p_value": p_boot,
                "n_setups": int(n_setups),
                "n_paired_units": int(target_counts.sum()),
                "min_units_per_setup": int(target_counts.min()),
                "median_units_per_setup": float(np.median(target_counts)),
                "max_units_per_setup": int(target_counts.max()),
                "setup_weighting": "equal",
                "setup_definition": " + ".join(setup_cols),
                "included_setups": "; ".join(setup_names),
            }
        )

    return pd.DataFrame(rows)


def finite_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]), col] = np.nan
    return out


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
            metric = infer_metric_from_filename(path)
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

    # The matrix score column is used only to read English performance.
    # The English method must inherit the metric metadata used by the
    # existing per-fold ranker rows. This ensures that paired comparisons
    # place all methods in the same task-model-metric group.
    if "metric" not in per_fold_dataset.columns:
        raise ValueError(
            f"Per-fold rows for {dataset!r} do not contain a 'metric' column."
        )

    metric_keys = (
        per_fold_dataset["metric"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(metric_keys) != 1:
        raise ValueError(
            f"Expected exactly one artifact metric for {dataset!r}, "
            f"but found {metric_keys}."
        )

    artifact_metric = metric_keys[0]

    if "metric_name" in per_fold_dataset.columns:
        metric_names = (
            per_fold_dataset.loc[
                per_fold_dataset["metric"].astype(str).eq(artifact_metric),
                "metric_name",
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    else:
        metric_names = []

    if len(metric_names) > 1:
        raise ValueError(
            f"Expected one metric label for {dataset!r}, "
            f"but found {metric_names}."
        )

    artifact_metric_name = (
        metric_names[0]
        if metric_names
        else metric_label(artifact_metric)
    )

    working = matrix[
        ["task_lang", "transfer_lang", score_col]
    ].copy()

    working["task_lang"] = working["task_lang"].astype(str)
    working["transfer_lang"] = working["transfer_lang"].astype(str)
    working[score_col] = pd.to_numeric(
        working[score_col],
        errors="coerce",
    )
    working = working.dropna(subset=[score_col])

    english = (
        working
        .query("transfer_lang == 'eng'")
        .groupby("task_lang", as_index=False)[score_col]
        .max()
        .rename(
            columns={
                "task_lang": "target_lang",
                score_col: "predicted_performance",
            }
        )
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
        .dropna(
            subset=[
                "target_lang",
                "actual_best_performance",
            ]
        )
        .copy()
    )

    target_oracle["target_lang"] = (
        target_oracle["target_lang"].astype(str)
    )
    target_oracle["actual_best_performance"] = pd.to_numeric(
        target_oracle["actual_best_performance"],
        errors="coerce",
    )

    target_oracle = (
        target_oracle
        .groupby("target_lang", as_index=False)
        .agg(
            actual_best_performance=(
                "actual_best_performance",
                "max",
            ),
            target_resource_level=(
                "target_resource_level",
                "first",
            ),
        )
    )

    joined = target_oracle.merge(
        english,
        on="target_lang",
        how="inner",
    )

    if joined.empty:
        return pd.DataFrame()

    joined["dataset"] = dataset
    joined["dataset_model"] = dataset
    joined["task"] = task
    joined["model"] = model
    joined["model_name"] = model_label(model)

    # Use the artifact metric key rather than score_col.
    # For UD-Dep, this assigns f1_score instead of las.
    joined["metric"] = artifact_metric
    joined["metric_name"] = artifact_metric_name

    joined["method_id"] = "always_eng"
    joined["method"] = "Always English"
    joined["method_type"] = "English"
    joined["method_group"] = "English"

    joined["query_id"] = (
        joined["dataset"].astype(str)
        + "::"
        + joined["target_lang"].astype(str)
    )
    joined["target_id"] = joined["query_id"]

    joined["predicted_best_source"] = "eng"
    joined["predicted_source_resource_level"] = "hrl"
    joined["actual_best_source"] = np.nan
    joined["actual_best_source_resource_level"] = np.nan

    actual = pd.to_numeric(
        joined["actual_best_performance"],
        errors="coerce",
    )
    predicted = pd.to_numeric(
        joined["predicted_performance"],
        errors="coerce",
    )
    denom = actual.where(actual > 0)

    joined["performance_loss"] = (
        actual - predicted
    ) / denom
    joined["performance_loss_pct"] = (
        100.0 * joined["performance_loss"]
    )

    return recompute_scale_columns(joined)


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


def add_holm_adjustment(
    df: pd.DataFrame,
    p_col: str = "bootstrap_p_value",
    family_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add Holm-adjusted p-values within each stated testing family.
    """
    if df.empty or p_col not in df.columns:
        return df.copy()

    out = df.copy()
    out["holm_p_value"] = np.nan

    if family_cols:
        grouped = out.groupby(family_cols, dropna=False, sort=False)
    else:
        grouped = [(None, out)]

    for _, group in grouped:
        valid = pd.to_numeric(group[p_col], errors="coerce").dropna()
        if valid.empty:
            continue

        ordered = valid.sort_values()
        m = len(ordered)
        adjusted = np.empty(m, dtype=float)
        running = 0.0

        for rank, (_, p_value) in enumerate(ordered.items()):
            candidate = (m - rank) * float(p_value)
            running = max(running, candidate)
            adjusted[rank] = min(1.0, running)

        out.loc[ordered.index, "holm_p_value"] = adjusted

    return out


def add_comparison_metadata(
    df: pd.DataFrame,
    comparison_family: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["outcome_name"] = out["outcome"].map(outcome_label)
    out["comparison_family"] = comparison_family
    out["difference_definition"] = "method_a minus method_b"
    out["negative_difference_favours"] = out["method_a"]
    out["positive_difference_favours"] = out["method_b"]

    return out


def make_fixed_methods_vs_english_tables(
    per_fold: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    """
    Compare two pre-specified methods with Always English.

    These comparisons are only valid on the common English-available subset.
    """
    by_setup_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        if outcome not in per_fold.columns:
            continue

        by_setup = paired_bootstrap_selected_pairs(
            per_fold,
            group_cols=SETUP_COLS,
            method_col="method",
            value_col=outcome,
            unit_col="query_id",
            pairs=FIXED_ENGLISH_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 * outcome_index,
        )
        by_setup_frames.append(by_setup)

        overall = hierarchical_paired_bootstrap_differences(
            per_fold,
            setup_cols=SETUP_COLS,
            method_col="method",
            value_col=outcome,
            unit_col="query_id",
            pairs=FIXED_ENGLISH_PAIRS,
            n_bootstrap=n_bootstrap,
            seed=seed + 10000 * outcome_index,
        )
        overall_frames.append(overall)

    if by_setup_frames:
        by_setup = pd.concat(by_setup_frames, ignore_index=True)
        by_setup = add_comparison_metadata(
            by_setup,
            comparison_family="Fixed method vs English",
        )
        by_setup = add_holm_adjustment(
            by_setup,
            family_cols=SETUP_COLS + ["outcome"],
        )
        write_csv(
            by_setup,
            folder / "fixed_methods_vs_english_by_setup.csv",
        )
    else:
        write_error(
            folder / "fixed_methods_vs_english_by_setup.csv",
            "no fixed-method English comparisons available",
        )

    if overall_frames:
        overall = pd.concat(overall_frames, ignore_index=True)
        overall = add_comparison_metadata(
            overall,
            comparison_family="Fixed method vs English",
        )
        overall = add_holm_adjustment(
            overall,
            family_cols=["outcome"],
        )
        write_csv(
            overall,
            folder / "fixed_methods_vs_english_overall.csv",
        )
    else:
        write_error(
            folder / "fixed_methods_vs_english_overall.csv",
            "no fixed-method English comparisons available",
        )


def build_paradigm_target_values(
    per_fold: pd.DataFrame,
    outcome: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct pre-specified equal-weight paradigm means for every target.

    A target contributes to a paradigm only when every pre-specified method
    in that paradigm is observed. Between-paradigm tests then use complete
    paired targets for the two paradigms being compared.
    """
    if outcome not in per_fold.columns:
        return pd.DataFrame(), pd.DataFrame()

    required_methods = {
        method
        for methods in PARADIGM_METHODS.values()
        for method in methods
    }

    base = (
        per_fold.loc[
            per_fold["method"].isin(required_methods),
            SETUP_COLS + ["query_id", "target_lang", "method", outcome],
        ]
        .dropna(subset=["query_id", "method", outcome])
        .copy()
    )

    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    method_values = (
        base
        .groupby(
            SETUP_COLS + ["query_id", "target_lang", "method"],
            dropna=False,
            as_index=False,
        )[outcome]
        .mean()
    )

    total_targets = (
        method_values
        .groupby(SETUP_COLS, dropna=False)["query_id"]
        .nunique()
        .rename("n_targets_any_paradigm_method")
        .reset_index()
    )

    value_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for setup_keys, setup_df in method_values.groupby(
        SETUP_COLS,
        dropna=False,
        sort=False,
    ):
        if not isinstance(setup_keys, tuple):
            setup_keys = (setup_keys,)

        setup_meta = dict(zip(SETUP_COLS, setup_keys))
        n_any = int(setup_df["query_id"].nunique())
        observed_setup_methods = sorted(set(setup_df["method"].astype(str)))

        pivot = setup_df.pivot_table(
            index=["query_id", "target_lang"],
            columns="method",
            values=outcome,
            aggfunc="mean",
        )

        for paradigm, methods in PARADIGM_METHODS.items():
            missing_setup_methods = [
                method for method in methods
                if method not in pivot.columns
            ]

            if missing_setup_methods:
                n_complete = 0
                complete_fraction = 0.0
                coverage_rows.append(
                    {
                        **setup_meta,
                        "outcome": outcome,
                        "outcome_name": outcome_label(outcome),
                        "paradigm": paradigm,
                        "expected_methods": "; ".join(methods),
                        "observed_methods": "; ".join(observed_setup_methods),
                        "missing_methods": "; ".join(missing_setup_methods),
                        "n_targets_any_paradigm_method": n_any,
                        "n_complete_targets": n_complete,
                        "complete_target_fraction": complete_fraction,
                    }
                )
                continue

            complete = pivot[methods].dropna()
            n_complete = int(complete.shape[0])
            complete_fraction = (
                float(n_complete / n_any)
                if n_any > 0
                else np.nan
            )

            coverage_rows.append(
                {
                    **setup_meta,
                    "outcome": outcome,
                    "outcome_name": outcome_label(outcome),
                    "paradigm": paradigm,
                    "expected_methods": "; ".join(methods),
                    "observed_methods": "; ".join(observed_setup_methods),
                    "missing_methods": "",
                    "n_targets_any_paradigm_method": n_any,
                    "n_complete_targets": n_complete,
                    "complete_target_fraction": complete_fraction,
                }
            )

            if complete.empty:
                continue

            values = complete.mean(axis=1).rename(outcome).reset_index()
            for col, value in setup_meta.items():
                values[col] = value

            values["paradigm"] = paradigm
            values["paradigm_methods"] = "; ".join(methods)
            value_frames.append(values)

    target_values = (
        pd.concat(value_frames, ignore_index=True, sort=False)
        if value_frames
        else pd.DataFrame()
    )
    coverage = pd.DataFrame(coverage_rows)

    if not target_values.empty:
        target_values = target_values.merge(
            total_targets,
            on=SETUP_COLS,
            how="left",
        )

    return target_values, coverage


def make_paradigm_paired_test_tables(
    per_fold: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    """
    Test pre-specified between-paradigm comparisons.

    Each target-level paradigm score is the equal-weight mean of all methods
    assigned to that paradigm. No best method is selected within a setup.
    """
    by_setup_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []

    for outcome_index, outcome in enumerate(REVIEWER_OUTCOMES):
        target_values, coverage = build_paradigm_target_values(
            per_fold,
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
            seed=seed + 2000 * outcome_index,
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
            seed=seed + 20000 * outcome_index,
        )
        overall_frames.append(overall)

    if coverage_frames:
        coverage = pd.concat(coverage_frames, ignore_index=True)
        write_csv(
            coverage,
            folder / "paradigm_target_coverage.csv",
        )
    else:
        write_error(
            folder / "paradigm_target_coverage.csv",
            "no paradigm target coverage available",
        )

    if by_setup_frames:
        by_setup = pd.concat(by_setup_frames, ignore_index=True)
        by_setup = add_comparison_metadata(
            by_setup,
            comparison_family="Between paradigms",
        )
        by_setup["method_a_components"] = by_setup["method_a"].map(
            lambda x: "; ".join(PARADIGM_METHODS.get(str(x), []))
        )
        by_setup["method_b_components"] = by_setup["method_b"].map(
            lambda x: "; ".join(PARADIGM_METHODS.get(str(x), []))
        )
        by_setup = add_holm_adjustment(
            by_setup,
            family_cols=SETUP_COLS + ["outcome"],
        )
        write_csv(
            by_setup,
            folder / "paradigm_paired_tests_by_setup.csv",
        )
    else:
        write_error(
            folder / "paradigm_paired_tests_by_setup.csv",
            "no between-paradigm comparisons available",
        )

    if overall_frames:
        overall = pd.concat(overall_frames, ignore_index=True)
        overall = add_comparison_metadata(
            overall,
            comparison_family="Between paradigms",
        )
        overall["method_a_components"] = overall["method_a"].map(
            lambda x: "; ".join(PARADIGM_METHODS.get(str(x), []))
        )
        overall["method_b_components"] = overall["method_b"].map(
            lambda x: "; ".join(PARADIGM_METHODS.get(str(x), []))
        )
        overall = add_holm_adjustment(
            overall,
            family_cols=["outcome"],
        )
        write_csv(
            overall,
            folder / "paradigm_paired_tests_overall.csv",
        )
    else:
        write_error(
            folder / "paradigm_paired_tests_overall.csv",
            "no between-paradigm comparisons available",
        )


def make_reviewer_paired_test_tables(
    variant: str,
    per_fold: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    """
    Write the paired tests requested by the reviewer.
    """
    make_paradigm_paired_test_tables(
        per_fold=per_fold,
        folder=folder,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if variant == "with_english":
        make_fixed_methods_vs_english_tables(
            per_fold=per_fold,
            folder=folder,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )


def task_balanced_resource_summary_local(
    per_fold: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    if level not in {"method", "method_type"}:
        raise ValueError("level must be either 'method' or 'method_type'.")

    df = clean_method_rows(per_fold)
    df = df.loc[df["target_resource_level"].isin(RESOURCE_LEVELS)].copy()

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
            sub = group.loc[group["target_resource_level"].eq(resource)]
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
    """
    Fit the resource-level mixed model.

    Failure to fit the preferred or fallback model is returned as an error row
    rather than terminating the complete post-analysis pipeline.
    """
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

    model_df = per_fold[needed].dropna().copy()
    model_df = model_df.loc[
        model_df["method_type"].isin(MAIN_LMM_METHOD_TYPES)
        & model_df["target_resource_level"].isin(RESOURCE_LEVELS)
    ].copy()
    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df[np.isfinite(model_df[outcome])].copy()

    if model_df.empty:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": ["no finite rows after LMM filtering"],
            }
        )

    formula = (
        f"{outcome} ~ "
        'C(method_type, Treatment(reference="Individual"))'
        ' * C(target_resource_level, Treatment(reference="hrl"))'
        ' + C(model)'
    )

    result = None
    fit_note = ""
    fit_errors: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["dataset"],
                vc_formula={"target": "0 + C(target_id)"},
            )
            result = model.fit(
                method="lbfgs",
                reml=True,
                maxiter=500,
                disp=False,
            )
            fit_note = "dataset random intercept plus target variance component"
        except Exception as exc:
            fit_errors.append(
                "preferred model: "
                + f"{type(exc).__name__}: {exc}"
            )

        if result is None:
            try:
                model = smf.mixedlm(
                    formula,
                    data=model_df,
                    groups=model_df["dataset"],
                )
                result = model.fit(
                    method="lbfgs",
                    reml=True,
                    maxiter=500,
                    disp=False,
                )
                fit_note = "dataset random intercept only"
            except Exception as exc:
                fit_errors.append(
                    "fallback model: "
                    + f"{type(exc).__name__}: {exc}"
                )

    if result is None:
        return pd.DataFrame(
            {
                "outcome": [outcome],
                "outcome_name": [outcome_label(outcome)],
                "coefficient": ["error"],
                "error": ["; ".join(fit_errors)],
                "n_rows": [int(model_df.shape[0])],
                "n_datasets": [int(model_df["dataset"].nunique())],
                "n_targets": [int(model_df["target_id"].nunique())],
            }
        )

    conf = result.conf_int()
    rows: list[dict[str, object]] = []

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

    target_level = target_level.loc[
        target_level["target_resource_level"].isin(RESOURCE_LEVELS)
    ].copy()

    dataset_resource = (
        target_level
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
    prefix: str = "",
) -> None:
    """
    Summarise resource-Mondrian CTC outputs.

    CTC must have been computed during evaluation. If the artifacts contain
    only the older marginal CTC fields, this function writes explicit error
    tables and allows the rest of the post-analysis to finish.
    """
    dataset_path = folder / f"{prefix}ctc_resource_by_dataset_method.csv"
    main_path = folder / f"{prefix}ctc_main_resource_by_method.csv"

    if per_fold.empty:
        write_error(dataset_path, "no rows")
        write_error(main_path, "no rows")
        return

    ctc = clean_method_rows(per_fold)
    ctc = ctc.query("method_type != 'English'").copy()
    ctc = ctc.loc[ctc["target_resource_level"].isin(RESOURCE_LEVELS)].copy()

    if ctc.empty:
        write_error(dataset_path, "no CTC rows after filtering")
        write_error(main_path, "no CTC rows after filtering")
        return

    if "cnotc_scheme" not in ctc.columns:
        message = (
            "old marginal CTC artifacts detected: cnotc_scheme is absent; "
            "rerun evaluation with resource-Mondrian CTC"
        )
        write_error(dataset_path, message)
        write_error(main_path, message)
        return

    schemes = set(ctc["cnotc_scheme"].dropna().astype(str))
    if schemes != {"mondrian_resource"}:
        message = (
            "expected cnotc_scheme=mondrian_resource, found "
            + ", ".join(sorted(schemes))
        )
        write_error(dataset_path, message)
        write_error(main_path, message)
        return

    missing = [c for c in CTC_COLS if c not in ctc.columns]
    if missing:
        message = "Mondrian CTC artifacts are missing: " + ", ".join(missing)
        write_error(dataset_path, message)
        write_error(main_path, message)
        return

    numeric_cols = [*CTC_COLS, "cnotc_quantile_index"]
    present_numeric = [c for c in numeric_cols if c in ctc.columns]
    ctc = finite_numeric(ctc, present_numeric)
    ctc = ctc.dropna(subset=CTC_COLS).copy()

    if ctc.empty:
        write_error(dataset_path, "no complete Mondrian CTC rows")
        write_error(main_path, "no complete Mondrian CTC rows")
        return

    ctc["resource"] = ctc["target_resource_level"].map(resource_label)

    group_cols = [
        "dataset",
        "task",
        "model_name",
        "metric_name",
        "method",
        "method_type",
        "resource",
    ]

    agg_spec: dict[str, tuple[str, str]] = {
        "n_targets": ("target_lang", "nunique"),
        "mean_group_calibration_size": ("cnotc_group_calibration_size", "mean"),
        "min_group_calibration_size": ("cnotc_group_calibration_size", "min"),
        "finite_threshold_rate": ("cnotc_finite_threshold", "mean"),
        "full_pool_fallback_rate": ("cnotc_full_pool_fallback", "mean"),
        "trial_complexity": ("cnotc_trial_complexity", "mean"),
        "pool_fraction": ("cnotc_pool_fraction", "mean"),
        "near_oracle_coverage": ("cnotc_near_oracle_coverage", "mean"),
        "exact_best_coverage": ("cnotc_exact_best_coverage", "mean"),
        "best_in_set_pl": ("cnotc_best_in_set_performance_loss", "mean"),
    }

    if "cnotc_quantile_index" in ctc.columns:
        agg_spec["mean_quantile_index"] = ("cnotc_quantile_index", "mean")

    by_dataset_method = (
        ctc
        .groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"model_name": "model", "metric_name": "metric"})
    )

    by_dataset_method["main_eligible"] = (
        by_dataset_method["n_targets"] >= int(min_ctc_targets)
    )
    write_csv(by_dataset_method, dataset_path)

    eligible = by_dataset_method.query("main_eligible").copy()
    if eligible.empty:
        write_error(
            main_path,
            f"no dataset-resource cells satisfy min_ctc_targets={min_ctc_targets}",
        )
        return

    main_agg: dict[str, tuple[str, str]] = {
        "n_dataset_cells": ("dataset", "nunique"),
        "n_targets": ("n_targets", "sum"),
        "mean_group_calibration_size": ("mean_group_calibration_size", "mean"),
        "min_group_calibration_size": ("min_group_calibration_size", "min"),
        "finite_threshold_rate": ("finite_threshold_rate", "mean"),
        "full_pool_fallback_rate": ("full_pool_fallback_rate", "mean"),
        "trial_complexity": ("trial_complexity", "mean"),
        "pool_fraction": ("pool_fraction", "mean"),
        "near_oracle_coverage": ("near_oracle_coverage", "mean"),
        "exact_best_coverage": ("exact_best_coverage", "mean"),
        "best_in_set_pl": ("best_in_set_pl", "mean"),
    }

    if "mean_quantile_index" in eligible.columns:
        main_agg["mean_quantile_index"] = ("mean_quantile_index", "mean")

    main_by_method = (
        eligible
        .groupby(["model", "method", "method_type", "resource"], dropna=False)
        .agg(**main_agg)
        .reset_index()
    )
    write_csv(main_by_method, main_path)


def make_nnr_tables(
    root: Path,
    data_matrices: dict[str, pd.DataFrame],
    outdir: Path,
    n_bootstrap: int,
    seed: int,
    min_ctc_targets: int,
) -> None:
    """Write NNRank-restricted outputs without overwriting main CTC tables."""
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
            write_error(
                folder / "nnrank_restricted_task_balanced_resource_by_method.csv",
                "no rows",
            )
            write_error(
                folder / "nnrank_restricted_task_balanced_resource_by_method_type.csv",
                "no rows",
            )
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
            group_cols=SETUP_COLS,
            method_col="method",
            value_col="performance_loss_pct",
            unit_col="query_id",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        write_csv(pairwise, folder / "nnrank_restricted_pairwise_bootstrap.csv")

        resource_by_method = task_balanced_resource_summary_local(
            nnr,
            level="method",
        )
        write_csv(
            resource_by_method,
            folder / "nnrank_restricted_task_balanced_resource_by_method.csv",
        )

        resource_by_type = task_balanced_resource_summary_local(
            nnr,
            level="method_type",
        )
        write_csv(
            resource_by_type,
            folder / "nnrank_restricted_task_balanced_resource_by_method_type.csv",
        )

        make_ctc_tables(
            per_fold=nnr,
            folder=folder,
            min_ctc_targets=min_ctc_targets,
            prefix="nnrank_restricted_",
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
    make_reviewer_paired_test_tables(
        variant=variant,
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


def run_revision_analysis(
    root: str | Path = ".",
    outdir: str | Path = "post-analysis/outputs",
    n_bootstrap: int = 20000,
    seed: int = 42,
    include_nnrank: bool = False,
    min_ctc_targets: int = 50,
) -> None:
    """Run the complete reviewer-response post-analysis pipeline."""
    root = Path(root).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()

    ensure_dir(outdir)
    ensure_dir(outdir / "tables")
    for variant in VARIANTS:
        tables_dir(outdir, variant)

    # Write immediately so a successful entry into the pipeline is visible
    # even if a later analysis fails.
    write_csv(
        pd.DataFrame(
            [
                {
                    "root": str(root),
                    "outdir": str(outdir),
                    "n_bootstrap": int(n_bootstrap),
                    "seed": int(seed),
                    "include_nnrank": bool(include_nnrank),
                    "min_ctc_targets": int(min_ctc_targets),
                }
            ]
        ),
        outdir / "run_config.csv",
    )

    print(f"[1/7] Loading main artifacts from {root / 'artifacts'}", flush=True)
    base_per_fold = load_per_fold_artifacts(root, nnrank=False)
    if base_per_fold.empty:
        raise RuntimeError(
            "No main per-fold artifacts found under "
            f"{root / 'artifacts'}."
        )

    assert_no_unknown_models(base_per_fold, "Main artifacts")
    write_csv(base_per_fold, outdir / "combined_per_fold_results_all.csv")

    print(
        "[2/7] Loading data matrices and constructing English subsets",
        flush=True,
    )
    data_matrices = load_data_matrices(root)

    without_english = make_without_english_per_fold(base_per_fold)
    with_english, english_coverage = make_with_english_per_fold(
        base_per_fold,
        data_matrices,
    )

    write_csv(
        english_coverage,
        tables_dir(outdir, "with_english") / "english_subset_coverage.csv",
    )

    print("[3/7] Writing full-benchmark tables", flush=True)
    write_variant_tables(
        variant="without_english",
        per_fold=without_english,
        data_matrices=data_matrices,
        outdir=outdir,
        n_bootstrap=n_bootstrap,
        seed=seed,
        min_ctc_targets=min_ctc_targets,
    )

    print("[4/7] Writing English-available tables", flush=True)
    write_variant_tables(
        variant="with_english",
        per_fold=with_english,
        data_matrices=data_matrices,
        outdir=outdir,
        n_bootstrap=n_bootstrap,
        seed=seed,
        min_ctc_targets=min_ctc_targets,
    )

    if include_nnrank:
        print("[5/7] Writing NNRank-restricted tables", flush=True)
        make_nnr_tables(
            root=root,
            data_matrices=data_matrices,
            outdir=outdir,
            n_bootstrap=n_bootstrap,
            seed=seed,
            min_ctc_targets=min_ctc_targets,
        )
    else:
        print("[5/7] NNRank-restricted tables skipped", flush=True)

    print("[6/7] Writing manifest", flush=True)
    write_manifest(outdir)

    files = sorted(path for path in outdir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(
            f"Analysis completed without writing files to {outdir}."
        )

    write_csv(
        pd.DataFrame(
            {
                "file": [
                    str(path.relative_to(outdir))
                    for path in files
                ]
            }
        ),
        outdir / "generated_files.csv",
    )

    print(
        f"[7/7] Complete: wrote {len(files) + 1} files to {outdir}",
        flush=True,
    )
    print(
        f"Without-English tables: {outdir / 'tables' / 'without_english'}",
        flush=True,
    )
    print(
        f"With-English tables: {outdir / 'tables' / 'with_english'}",
        flush=True,
    )
    print(f"Manifest: {outdir / 'table_manifest.csv'}", flush=True)