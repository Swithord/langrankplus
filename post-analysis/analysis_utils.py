#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DISTANCE_FEATURES = [
    "new_gen",
    "new_typ",
    "new_geo",
    "script",
    "distals_asjp",
    "distals_wiki_size",
]

RESOURCE_LEVELS = ["hrl", "mrl", "lrl"]

MAIN_LMM_METHOD_GROUPS = ["individual", "composite", "trained"]


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


def metric_multiplier(metric: str, values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 100.0
    max_value = clean.max()
    if metric == "bleu" and max_value > 5:
        return 1.0
    return 100.0


def method_group(method: str) -> str:
    m = str(method).lower()

    if m == "random":
        return "random"

    if m.startswith("single_"):
        return "individual"

    if any(token in m for token in ["composite", "rrf", "equal"]):
        return "composite"

    if any(token in m for token in ["lightgbm", "lgbm", "mlp", "listnet", "lambda"]):
        return "trained"

    if "nnrank" in m:
        return "nnrank"

    if any(token in m for token in ["new_gen", "new_typ", "new_geo", "script", "asjp", "wiki"]):
        return "individual"

    return "other"


def discover_per_fold_files(root: str | Path, nnrank: bool = False) -> list[Path]:
    root = Path(root)
    if nnrank:
        files = sorted((root / "artifacts" / "nnrank").glob("*/per_fold_*.csv"))
    else:
        files = sorted((root / "artifacts").glob("*/per_fold_*.csv"))
    return files


def load_per_fold_results(root: str | Path, nnrank: bool = False) -> pd.DataFrame:
    files = discover_per_fold_files(root, nnrank=nnrank)
    frames: list[pd.DataFrame] = []

    for path in files:
        df = read_csv(path)
        dataset = path.parent.name
        metric = infer_metric_from_filename(path)

        if "dataset" not in df.columns:
            df["dataset"] = dataset

        df["dataset"] = df["dataset"].fillna(dataset)
        df["artifact_path"] = str(path)
        df["metric"] = metric
        df["source_collection"] = "nnrank" if nnrank else "main"

        task_model = df["dataset"].map(infer_task_model)
        df["task"] = [x[0] for x in task_model]
        df["model"] = [x[1] for x in task_model]

        if "performance_loss" in df.columns:
            df["performance_loss_pct"] = 100.0 * pd.to_numeric(df["performance_loss"], errors="coerce")

        if {"actual_best_performance", "predicted_performance"}.issubset(df.columns):
            df["raw_oracle_gap"] = (
                pd.to_numeric(df["actual_best_performance"], errors="coerce")
                - pd.to_numeric(df["predicted_performance"], errors="coerce")
            )
            mult = metric_multiplier(metric, df["actual_best_performance"])
            df["raw_oracle_gap_points"] = mult * df["raw_oracle_gap"]
            df["actual_best_performance_points"] = mult * pd.to_numeric(df["actual_best_performance"], errors="coerce")
            df["predicted_performance_points"] = mult * pd.to_numeric(df["predicted_performance"], errors="coerce")

        if "ndcg" in df.columns:
            df["ndcg_pct"] = 100.0 * pd.to_numeric(df["ndcg"], errors="coerce")

        if "method" in df.columns:
            df["method_group"] = df["method"].map(method_group)

        if "target_resource_level" in df.columns:
            df["target_resource_level"] = df["target_resource_level"].astype(str).str.lower()

        if "query_id" not in df.columns and "target_lang" in df.columns:
            df["query_id"] = df["dataset"].astype(str) + "::" + df["target_lang"].astype(str)

        if "target_lang" in df.columns:
            df["target_id"] = df["dataset"].astype(str) + "::" + df["target_lang"].astype(str)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


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

    lookup = (
        per_fold[cols]
        .dropna()
        .drop_duplicates()
        .query("target_resource_level in @RESOURCE_LEVELS")
    )

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
                p_boot = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
                p_boot = min(p_boot, 1.0)

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


def simple_group_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    available = [c for c in value_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return (
        df.groupby(group_cols, dropna=False)[available]
        .mean()
        .reset_index()
    )


def task_balanced_resource_summary(
    df: pd.DataFrame,
    method_level: str = "method_group",
) -> pd.DataFrame:
    needed = [
        "dataset",
        "task",
        "model",
        method_level,
        "target_resource_level",
        "performance_loss_pct",
        "raw_oracle_gap_points",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame({"error": [f"missing columns: {missing}"]})

    base = (
        df.query("target_resource_level in @RESOURCE_LEVELS")
        .groupby(["dataset", "task", "model", method_level, "target_resource_level"], dropna=False)
        [["performance_loss_pct", "raw_oracle_gap_points"]]
        .mean()
        .reset_index()
    )

    balanced = (
        base
        .groupby(["model", method_level, "target_resource_level"], dropna=False)
        .agg(
            performance_loss_pct=("performance_loss_pct", "mean"),
            raw_oracle_gap_points=("raw_oracle_gap_points", "mean"),
            n_dataset_cells=("dataset", "nunique"),
        )
        .reset_index()
    )

    wide = balanced.pivot_table(
        index=["model", method_level],
        columns="target_resource_level",
        values=["performance_loss_pct", "raw_oracle_gap_points", "n_dataset_cells"],
        aggfunc="first",
    )

    wide.columns = [f"{metric}_{resource}" for metric, resource in wide.columns]
    wide = wide.reset_index()

    for metric in ["performance_loss_pct", "raw_oracle_gap_points"]:
        hrl = f"{metric}_hrl"
        mrl = f"{metric}_mrl"
        lrl = f"{metric}_lrl"
        if hrl in wide.columns and mrl in wide.columns:
            wide[f"{metric}_mrl_minus_hrl"] = wide[mrl] - wide[hrl]
        if hrl in wide.columns and lrl in wide.columns:
            wide[f"{metric}_lrl_minus_hrl"] = wide[lrl] - wide[hrl]

    return wide


def finite_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]), col] = np.nan
    return out