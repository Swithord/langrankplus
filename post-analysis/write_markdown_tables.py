#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAIN_METHODS = [
    "composite_equal",
    "rrf_equal",
    "lightgbm_lambdarank",
    "mlp_listnet",
]

METHOD_LABELS = {
    "always_eng": "English",
    "composite_equal": "Composite-Equal",
    "rrf_equal": "Composite-RRF",
    "lightgbm_lambdarank": "LightGBM",
    "mlp_listnet": "MLP",
    "nnrank": "NNRank",
    "single_new_gen": "Genetic",
    "single_new_typ": "Typological",
    "single_new_geo": "Geographic",
    "single_script": "Script",
    "single_distals_asjp": "ASJP",
    "single_distals_wiki_size": "Wikipedia size",
    "random": "Random",
}

METHOD_ORDER = {
    "always_eng": 0,
    "single_new_gen": 10,
    "single_new_typ": 11,
    "single_new_geo": 12,
    "single_script": 13,
    "single_distals_asjp": 14,
    "single_distals_wiki_size": 15,
    "composite_equal": 20,
    "rrf_equal": 21,
    "lightgbm_lambdarank": 30,
    "mlp_listnet": 31,
    "nnrank": 40,
}

RESOURCE_ORDER = {
    "hrl": 0,
    "mrl": 1,
    "lrl": 2,
}

MODEL_ORDER = {
    "mt5": 0,
    "xlm-r": 1,
}

OUTCOME_LABELS = {
    "performance_loss_pct": "Performance loss (%)",
    "raw_oracle_gap_points": "Raw oracle gap",
    "actual_best_performance_points": "Oracle score",
    "predicted_performance_points": "Selected-source score",
}

CTC_LABELS = {
    "cnotc_trial_complexity": "Trial complexity",
    "cnotc_pool_fraction": "Pool fraction",
    "cnotc_near_oracle_coverage": "Near-oracle coverage",
    "cnotc_exact_best_coverage": "Exact-best coverage",
    "cnotc_best_in_set_performance_loss": "Best-in-set PL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write paper-ready markdown tables from post-analysis CSV outputs."
    )
    parser.add_argument("--outdir", type=Path, default=Path("post-analysis/outputs"))
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help="Defaults to <outdir>/markdown_tables.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places for reported estimates.",
    )
    return parser.parse_args()


def necessary_dir(outdir: Path) -> Path:
    return outdir / "tables" / "necessary"


def other_dir(outdir: Path) -> Path:
    return outdir / "tables" / "other"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    return df


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalise_method(method: object) -> str:
    return str(method).strip()


def method_label(method: object) -> str:
    method = normalise_method(method)
    return METHOD_LABELS.get(method, method)


def model_sort_key(model: object) -> int:
    return MODEL_ORDER.get(str(model), 99)


def method_sort_key(method: object) -> int:
    return METHOD_ORDER.get(str(method), 99)


def resource_sort_key(resource: object) -> int:
    return RESOURCE_ORDER.get(str(resource), 99)


def drop_unknown_models(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df.copy()

    return df.loc[~df["model"].astype(str).eq("unknown")].copy()


def finite(x: object) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def fmt_num(x: object, digits: int = 2) -> str:
    if not finite(x):
        return "—"

    return f"{float(x):.{digits}f}"


def fmt_int(x: object) -> str:
    if not finite(x):
        return "—"

    return str(int(round(float(x))))


def fmt_signed(x: object, digits: int = 2) -> str:
    if not finite(x):
        return "—"

    value = float(x)
    return f"{value:+.{digits}f}"


def fmt_p(x: object) -> str:
    if not finite(x):
        return "—"

    p = float(x)

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def fmt_ci(
    mean: object,
    lower: object,
    upper: object,
    digits: int = 2,
    bold: bool = False,
) -> str:
    if not finite(mean):
        text = "—"
    elif finite(lower) and finite(upper):
        text = f"{float(mean):.{digits}f} [{float(lower):.{digits}f}, {float(upper):.{digits}f}]"
    else:
        text = f"{float(mean):.{digits}f}"

    if bold and text != "—":
        return f"**{text}**"

    return text


def fmt_effect(
    estimate: object,
    lower: object,
    upper: object,
    p_value: object,
    digits: int = 2,
) -> str:
    bold = finite(p_value) and float(p_value) < 0.05
    return fmt_ci(estimate, lower, upper, digits=digits, bold=bold)


def fmt_percent_from_fraction(x: object, digits: int = 1, bold: bool = False) -> str:
    if not finite(x):
        text = "—"
    else:
        text = f"{100.0 * float(x):.{digits}f}%"

    if bold and text != "—":
        return f"**{text}**"

    return text


def fmt_maybe_percent_loss(series: pd.Series, x: object, digits: int = 2, bold: bool = False) -> str:
    if not finite(x):
        text = "—"
    else:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        value = float(x)

        if clean.empty:
            text = f"{value:.{digits}f}"
        elif float(clean.abs().max()) <= 1.5:
            text = f"{100.0 * value:.{digits}f}"
        else:
            text = f"{value:.{digits}f}"

    if bold and text != "—":
        return f"**{text}**"

    return text


def escape_md(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"

    cols = list(df.columns)
    header = "| " + " | ".join(escape_md(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(escape_md(row[c]) for c in cols) + " |")

    return "\n".join([header, sep, *rows]) + "\n"


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    content = f"# {title}\n\n{body.strip()}\n"

    path.write_text(content + "\n", encoding="utf-8")


def outcome_df(summary: pd.DataFrame, outcome: str) -> pd.DataFrame:
    df = summary.loc[summary["outcome"].eq(outcome)].copy()
    df = drop_unknown_models(df)
    df["method"] = df["method"].astype(str)

    return df


def row_for_method(group: pd.DataFrame, method: str) -> pd.Series | None:
    sub = group.loc[group["method"].eq(method)].copy()

    if sub.empty:
        return None

    return sub.sort_values("mean", ascending=True).iloc[0]


def best_row(group: pd.DataFrame, method_group: str | None = None) -> pd.Series | None:
    sub = group.copy()

    if method_group is not None:
        sub = sub.loc[sub["method_group"].eq(method_group)].copy()

    if sub.empty:
        return None

    return sub.sort_values("mean", ascending=True).iloc[0]


def cell_from_row(
    row: pd.Series | None,
    best_mean: float | None,
    digits: int,
    include_method_name: bool = False,
) -> str:
    if row is None:
        return "—"

    bold = best_mean is not None and finite(row["mean"]) and np.isclose(
        float(row["mean"]),
        float(best_mean),
        rtol=0.0,
        atol=1e-12,
    )

    value = fmt_ci(
        row["mean"],
        row["ci_lower"],
        row["ci_upper"],
        digits=digits,
        bold=bold,
    )

    if include_method_name:
        return f"{method_label(row['method'])}: {value}"

    return value


def make_compact_main_table(
    summary: pd.DataFrame,
    outcome: str,
    digits: int,
) -> pd.DataFrame:
    df = outcome_df(summary, outcome)

    rows = []
    key_cols = ["dataset", "task", "model", "metric"]

    for key, group in df.groupby(key_cols, dropna=False, sort=False):
        dataset, task, model, metric = key

        candidates: dict[str, pd.Series | None] = {
            "Best individual": best_row(group, "individual"),
            "Composite-Equal": row_for_method(group, "composite_equal"),
            "Composite-RRF": row_for_method(group, "rrf_equal"),
            "LightGBM": row_for_method(group, "lightgbm_lambdarank"),
            "MLP": row_for_method(group, "mlp_listnet"),
        }

        present = [r for r in candidates.values() if r is not None and finite(r["mean"])]
        best_mean = min(float(r["mean"]) for r in present) if present else None

        rows.append(
            {
                "Dataset": dataset,
                "Task": task,
                "Model": model,
                "Metric": metric,
                "n": fmt_int(group["n_units"].max()),
                "Best individual": cell_from_row(
                    candidates["Best individual"],
                    best_mean,
                    digits,
                    include_method_name=True,
                ),
                "Composite-Equal": cell_from_row(candidates["Composite-Equal"], best_mean, digits),
                "Composite-RRF": cell_from_row(candidates["Composite-RRF"], best_mean, digits),
                "LightGBM": cell_from_row(candidates["LightGBM"], best_mean, digits),
                "MLP": cell_from_row(candidates["MLP"], best_mean, digits),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out = out.sort_values(["Task", "_model_order", "Dataset"]).drop(columns="_model_order")

    return out.reset_index(drop=True)


def make_best_method_table(summary: pd.DataFrame, digits: int) -> pd.DataFrame:
    pl = outcome_df(summary, "performance_loss_pct")
    raw = outcome_df(summary, "raw_oracle_gap_points")

    raw_lookup = {
        (r.dataset, r.task, r.model, r.metric, r.method): r
        for r in raw.itertuples(index=False)
    }

    rows = []

    for key, group in pl.groupby(["dataset", "task", "model", "metric"], dropna=False, sort=False):
        dataset, task, model, metric = key
        best = group.sort_values("mean", ascending=True).iloc[0]
        raw_row = raw_lookup.get((dataset, task, model, metric, best["method"]))

        rows.append(
            {
                "Dataset": dataset,
                "Task": task,
                "Model": model,
                "Metric": metric,
                "Best method": f"**{method_label(best['method'])}**",
                "Group": best["method_group"],
                "PL": fmt_ci(best["mean"], best["ci_lower"], best["ci_upper"], digits=digits, bold=True),
                "Raw gap": (
                    fmt_ci(raw_row.mean, raw_row.ci_lower, raw_row.ci_upper, digits=digits, bold=True)
                    if raw_row is not None
                    else "—"
                ),
                "n": fmt_int(best["n_units"]),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out = out.sort_values(["Task", "_model_order", "Dataset"]).drop(columns="_model_order")

    return out.reset_index(drop=True)


def lmm_term_label(term: object) -> str:
    term = str(term)

    mapping = {
        "Intercept": "Baseline: individual, HRL, mT5",
        'C(method_group, Treatment(reference="individual"))[T.composite]': "Composite vs individual, among HRL",
        'C(method_group, Treatment(reference="individual"))[T.trained]': "Trained vs individual, among HRL",
        'C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "LRL vs HRL, for individual methods",
        'C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "MRL vs HRL, for individual methods",
        "C(model)[T.xlm-r]": "XLM-R vs mT5",
        'C(method_group, Treatment(reference="individual"))[T.composite]:C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "Composite × LRL: change in LRL-HRL gap",
        'C(method_group, Treatment(reference="individual"))[T.trained]:C(target_resource_level, Treatment(reference="hrl"))[T.lrl]': "Trained × LRL: change in LRL-HRL gap",
        'C(method_group, Treatment(reference="individual"))[T.composite]:C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "Composite × MRL: change in MRL-HRL gap",
        'C(method_group, Treatment(reference="individual"))[T.trained]:C(target_resource_level, Treatment(reference="hrl"))[T.mrl]': "Trained × MRL: change in MRL-HRL gap",
        "target Var": "Target-language variance component",
    }

    return mapping.get(term, term)


def lmm_direction(row: pd.Series) -> str:
    term = str(row.get("term", ""))

    if term == "target Var":
        return "Target heterogeneity"

    if not finite(row.get("estimate")):
        return "—"

    estimate = float(row["estimate"])
    p = row.get("p_value")

    if finite(p) and float(p) < 0.05:
        if estimate < 0:
            return "Lower loss"
        if estimate > 0:
            return "Higher loss"

    return "No clear effect"


def make_lmm_markdown_table(lmm: pd.DataFrame, outcome: str, digits: int) -> pd.DataFrame:
    df = lmm.loc[lmm["outcome"].eq(outcome)].copy()

    if df.empty:
        return df

    rows = []

    for _, row in df.iterrows():
        term = str(row["term"])
        p_value = row.get("p_value")

        rows.append(
            {
                "Term": lmm_term_label(term),
                "Estimate [95% CI]": (
                    fmt_effect(
                        row.get("estimate"),
                        row.get("ci_lower"),
                        row.get("ci_upper"),
                        p_value,
                        digits=digits,
                    )
                    if term != "target Var"
                    else fmt_num(row.get("estimate"), digits=digits)
                ),
                "p": fmt_p(p_value),
                "Direction": lmm_direction(row),
            }
        )

    return pd.DataFrame(rows)


def make_english_baseline_table(summary: pd.DataFrame, digits: int) -> pd.DataFrame:
    summary = drop_unknown_models(summary)

    pl = outcome_df(summary, "performance_loss_pct")
    raw = outcome_df(summary, "raw_oracle_gap_points")

    raw_lookup = {
        (r.dataset, r.task, r.model, r.metric, r.method): r
        for r in raw.itertuples(index=False)
    }

    rows = []

    for key, group in pl.groupby(["dataset", "task", "model", "metric"], dropna=False, sort=False):
        dataset, task, model, metric = key

        english = row_for_method(group, "always_eng")
        non_base = group.loc[~group["method"].eq("always_eng")].copy()

        if english is None or non_base.empty:
            continue

        best = non_base.sort_values("mean", ascending=True).iloc[0]
        english_raw = raw_lookup.get((dataset, task, model, metric, "always_eng"))
        best_raw = raw_lookup.get((dataset, task, model, metric, best["method"]))

        pl_delta = float(english["mean"]) - float(best["mean"]) if finite(english["mean"]) and finite(best["mean"]) else np.nan

        raw_delta = (
            float(english_raw.mean) - float(best_raw.mean)
            if english_raw is not None and best_raw is not None and finite(english_raw.mean) and finite(best_raw.mean)
            else np.nan
        )

        rows.append(
            {
                "Dataset": dataset,
                "Task": task,
                "Model": model,
                "Metric": metric,
                "n": fmt_int(english["n_units"]),
                "English PL": fmt_ci(
                    english["mean"],
                    english["ci_lower"],
                    english["ci_upper"],
                    digits=digits,
                    bold=finite(pl_delta) and pl_delta <= 0,
                ),
                "Best ranker": f"**{method_label(best['method'])}**",
                "Best ranker PL": fmt_ci(
                    best["mean"],
                    best["ci_lower"],
                    best["ci_upper"],
                    digits=digits,
                    bold=finite(pl_delta) and pl_delta > 0,
                ),
                "PL reduction": fmt_signed(pl_delta, digits=digits),
                "English raw gap": (
                    fmt_ci(
                        english_raw.mean,
                        english_raw.ci_lower,
                        english_raw.ci_upper,
                        digits=digits,
                        bold=finite(raw_delta) and raw_delta <= 0,
                    )
                    if english_raw is not None
                    else "—"
                ),
                "Best raw gap": (
                    fmt_ci(
                        best_raw.mean,
                        best_raw.ci_lower,
                        best_raw.ci_upper,
                        digits=digits,
                        bold=finite(raw_delta) and raw_delta > 0,
                    )
                    if best_raw is not None
                    else "—"
                ),
                "Raw-gap reduction": fmt_signed(raw_delta, digits=digits),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out = out.sort_values(["Task", "_model_order", "Dataset"]).drop(columns="_model_order")

    return out.reset_index(drop=True)


def make_resource_gap_table(resource: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = drop_unknown_models(resource)

    rows = []

    for model, model_df in df.groupby("model", dropna=False, sort=False):
        model_df = model_df.copy()

        if "performance_loss_pct_lrl_minus_hrl" in model_df.columns:
            best_gap = pd.to_numeric(
                model_df["performance_loss_pct_lrl_minus_hrl"],
                errors="coerce",
            ).min()
        else:
            best_gap = np.nan

        for _, row in model_df.iterrows():
            gap = row.get("performance_loss_pct_lrl_minus_hrl", np.nan)
            bold_gap = finite(gap) and finite(best_gap) and np.isclose(
                float(gap),
                float(best_gap),
                rtol=0.0,
                atol=1e-12,
            )

            rows.append(
                {
                    "Model": row.get("model", "—"),
                    "Method": method_label(row.get("method", "—")),
                    "Cells HRL/MRL/LRL": (
                        f"{fmt_int(row.get('n_dataset_cells_hrl'))}/"
                        f"{fmt_int(row.get('n_dataset_cells_mrl'))}/"
                        f"{fmt_int(row.get('n_dataset_cells_lrl'))}"
                    ),
                    "PL HRL": fmt_num(row.get("performance_loss_pct_hrl"), digits=digits),
                    "PL MRL": fmt_num(row.get("performance_loss_pct_mrl"), digits=digits),
                    "PL LRL": fmt_num(row.get("performance_loss_pct_lrl"), digits=digits),
                    "MRL-HRL PL": fmt_signed(
                        row.get("performance_loss_pct_mrl_minus_hrl"),
                        digits=digits,
                    ),
                    "LRL-HRL PL": (
                        f"**{fmt_signed(gap, digits=digits)}**"
                        if bold_gap
                        else fmt_signed(gap, digits=digits)
                    ),
                    "Raw MRL-HRL": fmt_signed(
                        row.get("raw_oracle_gap_points_mrl_minus_hrl"),
                        digits=digits,
                    ),
                    "Raw LRL-HRL": fmt_signed(
                        row.get("raw_oracle_gap_points_lrl_minus_hrl"),
                        digits=digits,
                    ),
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out["_method_order"] = out["Method"].map(lambda x: method_sort_key(str(x)))
    out = out.sort_values(["_model_order", "_method_order", "Method"]).drop(
        columns=["_model_order", "_method_order"]
    )

    return out.reset_index(drop=True)


def make_selection_opportunity_table(selection: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = drop_unknown_models(selection)

    rows = []

    for _, row in df.iterrows():
        resource = str(row.get("target_resource_level", "—"))
        is_lrl = resource == "lrl"

        rows.append(
            {
                "Model": row.get("model", "—"),
                "Resource": resource.upper(),
                "Oracle score": fmt_num(row.get("oracle_score_points"), digits=digits),
                "Source SD": fmt_num(row.get("source_score_sd_points"), digits=digits),
                "Oracle − median": (
                    f"**{fmt_num(row.get('oracle_minus_median_points'), digits=digits)}**"
                    if is_lrl
                    else fmt_num(row.get("oracle_minus_median_points"), digits=digits)
                ),
                "Mean sources": fmt_num(row.get("mean_n_sources"), digits=digits),
                "Dataset cells": fmt_int(row.get("n_dataset_cells")),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out["_resource_order"] = out["Resource"].str.lower().map(resource_sort_key)
    out = out.sort_values(["_model_order", "_resource_order"]).drop(
        columns=["_model_order", "_resource_order"]
    )

    return out.reset_index(drop=True)


def make_target_counts_table(counts: pd.DataFrame) -> pd.DataFrame:
    df = drop_unknown_models(counts)

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "Dataset": row.get("dataset", "—"),
                "Task": row.get("task", "—"),
                "Model": row.get("model", "—"),
                "HRL": fmt_int(row.get("hrl_targets")),
                "MRL": fmt_int(row.get("mrl_targets")),
                "LRL": fmt_int(row.get("lrl_targets")),
                "Unknown": fmt_int(row.get("unknown_targets")),
                "Total": fmt_int(row.get("total_targets")),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out = out.sort_values(["Task", "_model_order", "Dataset"]).drop(columns="_model_order")

    return out.reset_index(drop=True)


def make_ctc_table(ctc: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = drop_unknown_models(ctc)

    rows = []

    for _, row in df.iterrows():
        near = row.get("cnotc_near_oracle_coverage")
        exact = row.get("cnotc_exact_best_coverage")

        rows.append(
            {
                "Model": row.get("model", "—"),
                "Method": method_label(row.get("method", "—")),
                "Group": row.get("method_group", "—"),
                "Resource": str(row.get("target_resource_level", "—")).upper(),
                "Dataset cells": fmt_int(row.get("n_dataset_cells")),
                "Targets": fmt_int(row.get("n_targets")),
                "Trial complexity": fmt_num(row.get("cnotc_trial_complexity"), digits=digits),
                "Pool fraction": fmt_percent_from_fraction(
                    row.get("cnotc_pool_fraction"),
                    digits=1,
                ),
                "Near-oracle coverage": fmt_percent_from_fraction(
                    near,
                    digits=1,
                    bold=finite(near) and float(near) >= 0.90,
                ),
                "Exact-best coverage": fmt_percent_from_fraction(
                    exact,
                    digits=1,
                    bold=finite(exact) and float(exact) >= 0.90,
                ),
                "Best-in-set PL": fmt_maybe_percent_loss(
                    df["cnotc_best_in_set_performance_loss"],
                    row.get("cnotc_best_in_set_performance_loss"),
                    digits=digits,
                ),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_model_order"] = out["Model"].map(model_sort_key)
    out["_resource_order"] = out["Resource"].str.lower().map(resource_sort_key)
    out["_method_order"] = out["Method"].map(lambda x: method_sort_key(str(x)))
    out = out.sort_values(
        ["_model_order", "_resource_order", "_method_order", "Method"]
    ).drop(columns=["_model_order", "_resource_order", "_method_order"])

    return out.reset_index(drop=True)


def make_method_group_average_table(summary: pd.DataFrame, digits: int) -> pd.DataFrame:
    pl = outcome_df(summary, "performance_loss_pct")
    raw = outcome_df(summary, "raw_oracle_gap_points")

    if pl.empty:
        return pd.DataFrame()

    best_by_group = (
        pl.sort_values("mean", ascending=True)
        .groupby(["dataset", "task", "model", "metric", "method_group"], as_index=False)
        .first()
    )

    raw_best_by_group = (
        raw.sort_values("mean", ascending=True)
        .groupby(["dataset", "task", "model", "metric", "method_group"], as_index=False)
        .first()
    )

    pl_group = (
        best_by_group
        .groupby("method_group", as_index=False)
        .agg(
            mean_pl=("mean", "mean"),
            median_pl=("mean", "median"),
            n_dataset_cells=("dataset", "nunique"),
        )
    )

    raw_group = (
        raw_best_by_group
        .groupby("method_group", as_index=False)
        .agg(
            mean_raw_gap=("mean", "mean"),
            median_raw_gap=("mean", "median"),
        )
    )

    out = pl_group.merge(raw_group, on="method_group", how="left")

    if out.empty:
        return out

    best_pl = out["mean_pl"].min()

    rows = []
    for _, row in out.iterrows():
        is_best = finite(row["mean_pl"]) and np.isclose(
            float(row["mean_pl"]),
            float(best_pl),
            rtol=0.0,
            atol=1e-12,
        )

        rows.append(
            {
                "Method group": row["method_group"],
                "Mean PL": (
                    f"**{fmt_num(row['mean_pl'], digits=digits)}**"
                    if is_best
                    else fmt_num(row["mean_pl"], digits=digits)
                ),
                "Median PL": fmt_num(row["median_pl"], digits=digits),
                "Mean raw gap": fmt_num(row["mean_raw_gap"], digits=digits),
                "Median raw gap": fmt_num(row["median_raw_gap"], digits=digits),
                "Dataset cells": fmt_int(row["n_dataset_cells"]),
            }
        )

    return pd.DataFrame(rows)


def write_main_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "main_metric_bootstrap_ci.csv"
    summary = read_csv_if_exists(path)

    if summary is None:
        return written

    summary = drop_unknown_models(summary)

    tables = [
        (
            "main_performance_loss.md",
            "Main performance-loss table",
            make_compact_main_table(summary, "performance_loss_pct", digits),
            (
                "Cells report target-bootstrap means with 95% confidence intervals. "
                "Bold indicates the lowest mean loss within the dataset-model row."
            ),
        ),
        (
            "main_raw_oracle_gap.md",
            "Main raw oracle-gap table",
            make_compact_main_table(summary, "raw_oracle_gap_points", digits),
            (
                "Cells report target-bootstrap means with 95% confidence intervals. "
                "Bold indicates the lowest mean raw oracle gap within the dataset-model row."
            ),
        ),
        (
            "main_best_methods.md",
            "Best method by dataset-model setting",
            make_best_method_table(summary, digits),
            "The best method is selected by mean performance loss.",
        ),
        (
            "method_group_averages.md",
            "Task-balanced method-group averages",
            make_method_group_average_table(summary, digits),
            (
                "Each method group is first reduced to its best method within each dataset-model setting, "
                "then averaged equally over dataset-model settings. Bold indicates the lowest mean PL."
            ),
        ),
    ]

    for filename, title, table, note in tables:
        body = f"{note}\n\n{markdown_table(table)}"
        write_markdown(markdown_dir / filename, title, body)
        written.append(filename)

    return written


def write_lmm_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "resource_lmm_fixed_effects.csv"
    lmm = read_csv_if_exists(path)

    if lmm is None:
        return written

    for outcome in ["performance_loss_pct", "raw_oracle_gap_points"]:
        table = make_lmm_markdown_table(lmm, outcome, digits)

        if table.empty:
            continue

        title = f"LMM fixed effects: {OUTCOME_LABELS.get(outcome, outcome)}"
        filename = f"lmm_{outcome}.md"
        note = (
            "Reference categories are individual-distance methods, HRL targets, and mT5. "
            "Bold coefficients have p < 0.05."
        )
        body = f"{note}\n\n{markdown_table(table)}"
        write_markdown(markdown_dir / filename, title, body)
        written.append(filename)

    return written


def write_english_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "english_baseline_comparison_bootstrap.csv"
    summary = read_csv_if_exists(path)

    if summary is None:
        return written

    table = make_english_baseline_table(summary, digits)

    title = "English baseline comparison"
    filename = "english_baseline_comparison.md"
    note = (
        "The best ranker is selected by mean performance loss on the English-available target subset. "
        "Positive reductions mean the best ranker improves over always-English. Bold marks the lower loss."
    )
    body = f"{note}\n\n{markdown_table(table)}"

    write_markdown(markdown_dir / filename, title, body)
    written.append(filename)

    return written


def write_resource_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "task_balanced_resource_by_method.csv"
    resource = read_csv_if_exists(path)

    if resource is not None:
        table = make_resource_gap_table(resource, digits)
        title = "Task-balanced resource gaps by method"
        filename = "task_balanced_resource_gaps.md"
        note = (
            "Each entry is averaged equally over dataset-model cells, not pooled over target languages. "
            "Bold marks the smallest LRL-HRL performance-loss gap within each model."
        )
        body = f"{note}\n\n{markdown_table(table)}"
        write_markdown(markdown_dir / filename, title, body)
        written.append(filename)

    path = necessary_dir(outdir) / "target_resource_counts.csv"
    counts = read_csv_if_exists(path)

    if counts is not None:
        table = make_target_counts_table(counts)
        title = "Target-resource counts"
        filename = "target_resource_counts.md"
        note = "Counts are target-language counts by dataset-model setting."
        body = f"{note}\n\n{markdown_table(table)}"
        write_markdown(markdown_dir / filename, title, body)
        written.append(filename)

    return written


def write_selection_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "selection_opportunity_task_balanced.csv"
    selection = read_csv_if_exists(path)

    if selection is None:
        return written

    table = make_selection_opportunity_table(selection, digits)
    title = "Source-selection opportunity by resource level"
    filename = "selection_opportunity_task_balanced.md"
    note = (
        "Rows are task-balanced over dataset-model cells. "
        "Bold marks the LRL oracle-minus-median source gap, the key low-resource opportunity measure."
    )
    body = f"{note}\n\n{markdown_table(table)}"

    write_markdown(markdown_dir / filename, title, body)
    written.append(filename)

    return written


def write_ctc_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "ctc_main_resource_by_method.csv"
    ctc = read_csv_if_exists(path)

    if ctc is None:
        return written

    if "error" in ctc.columns:
        table = ctc
    else:
        table = make_ctc_table(ctc, digits)

    title = "CTC resource-conditional summary"
    filename = "ctc_main_resource_by_method.md"
    note = (
        "Rows use only dataset-model settings meeting the minimum-target threshold. "
        "Coverage values at or above 90% are bolded."
    )
    body = f"{note}\n\n{markdown_table(table)}"

    write_markdown(markdown_dir / filename, title, body)
    written.append(filename)

    return written


def write_nnr_outputs(outdir: Path, markdown_dir: Path, digits: int) -> list[str]:
    written: list[str] = []

    path = necessary_dir(outdir) / "nnrank_restricted_metric_bootstrap_ci.csv"
    nnr_summary = read_csv_if_exists(path)

    if nnr_summary is not None:
        nnr_summary = drop_unknown_models(nnr_summary)

        tables = [
            (
                "nnrank_restricted_performance_loss.md",
                "NNRank-restricted performance-loss table",
                make_compact_main_table(nnr_summary, "performance_loss_pct", digits),
                (
                    "Rows are restricted to the NNRank-compatible language subset. "
                    "Bold indicates the lowest mean loss within the dataset-model row."
                ),
            ),
            (
                "nnrank_restricted_raw_oracle_gap.md",
                "NNRank-restricted raw oracle-gap table",
                make_compact_main_table(nnr_summary, "raw_oracle_gap_points", digits),
                (
                    "Rows are restricted to the NNRank-compatible language subset. "
                    "Bold indicates the lowest mean raw oracle gap within the dataset-model row."
                ),
            ),
            (
                "nnrank_restricted_best_methods.md",
                "NNRank-restricted best methods",
                make_best_method_table(nnr_summary, digits),
                "The best method is selected by mean performance loss on the NNRank-compatible subset.",
            ),
        ]

        for filename, title, table, note in tables:
            body = f"{note}\n\n{markdown_table(table)}"
            write_markdown(markdown_dir / filename, title, body)
            written.append(filename)

    path = necessary_dir(outdir) / "nnrank_restricted_task_balanced_resource.csv"
    nnr_resource = read_csv_if_exists(path)

    if nnr_resource is not None:
        table = make_resource_gap_table(nnr_resource, digits)
        title = "NNRank-restricted task-balanced resource gaps"
        filename = "nnrank_restricted_resource_gaps.md"
        note = (
            "Each entry is averaged equally over dataset-model cells in the NNRank-compatible subset. "
            "Bold marks the smallest LRL-HRL performance-loss gap within each model."
        )
        body = f"{note}\n\n{markdown_table(table)}"
        write_markdown(markdown_dir / filename, title, body)
        written.append(filename)

    return written


def write_index(markdown_dir: Path, written: Iterable[str]) -> None:
    written = list(written)

    lines = [
        "# Markdown table index",
        "",
        "Generated markdown tables:",
        "",
    ]

    for filename in sorted(written):
        title = filename.removesuffix(".md").replace("_", " ")
        lines.append(f"- [{title}]({filename})")

    lines.append("")

    (markdown_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    outdir = args.outdir
    markdown_dir = args.markdown_dir or (outdir / "markdown_tables")
    markdown_dir = ensure_dir(markdown_dir)

    written: list[str] = []

    written.extend(write_main_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_lmm_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_english_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_resource_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_selection_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_ctc_outputs(outdir, markdown_dir, args.digits))
    written.extend(write_nnr_outputs(outdir, markdown_dir, args.digits))

    write_index(markdown_dir, written)

    print(f"Wrote {len(written)} markdown tables to: {markdown_dir}")
    print(f"Wrote index to: {markdown_dir / 'README.md'}")


if __name__ == "__main__":
    main()