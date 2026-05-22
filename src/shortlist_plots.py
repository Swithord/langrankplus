from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAX_SCATTER_LABELS = 8
MAX_BAR_LABELS = 18
MIN_LABEL_DISTANCE_PIXELS = 54


def _save(fig, outdir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)

    plt.close(fig)
    return paths


def _method_label(text: str) -> str:
    text = str(text)

    replacements = {
        "lightgbm_lambdarank": "LightGBM",
        "mlp_listnet": "MLP",
        "composite_equal": "Composite",
        "rrf_equal": "RRF",
        "single_new_gen": "Genetic",
        "single_new_typ": "Typological",
        "single_new_geo": "Geographic",
        "single_script": "Script",
        "composite_pairwise_nested": "Composite fitted",
        "rrf_pairwise_nested": "RRF fitted",
        "composite_pairwise_frozen": "Composite frozen",
        "rrf_pairwise_frozen": "RRF frozen",
        "random": "Random",
        "consensus": "Consensus",
        "pareto": "Pareto",
    }

    return replacements.get(text, text.replace("_", " "))


def _metric_label(text: str) -> str:
    text = str(text)
    text = text.replace("relative_0p05_coverage", "near-best coverage, 5% relative")
    text = text.replace("std_0_coverage", "exact-best coverage")
    text = text.replace("std_0p25_coverage", "near-best coverage, 0.25 SD")
    text = text.replace("std_0p5_coverage", "near-best coverage, 0.5 SD")
    text = text.replace("std_1_coverage", "near-best coverage, 1 SD")
    text = text.replace("exact_best_coverage", "exact-best coverage")
    text = text.replace("_", " ")
    return text


def _selector_label(selector: str) -> str:
    selector = str(selector)

    match = re.fullmatch(r"fixed_top_(\d+)", selector)
    if match:
        return f"top-{match.group(1)}"

    match = re.search(r"conformal_(relative_0p05|std_0|std_0p25|std_0p5|std_1)_alpha_([^_]+)_(uncapped|cap_\d+)", selector)
    if match:
        rule, alpha, cap = match.groups()
        rule_label = {
            "relative_0p05": "conf rel 5%",
            "std_0": "conf exact",
            "std_0p25": "conf 0.25 SD",
            "std_0p5": "conf 0.5 SD",
            "std_1": "conf 1 SD",
        }.get(rule, rule)
        cap_label = cap.replace("cap_", "cap ")
        return f"{rule_label}, {cap_label}"

    match = re.search(r"calibrated_min_k_(relative_0p05|std_0|std_0p25|std_0p5|std_1)_target_([^_]+)_max_(\d+)", selector)
    if match:
        rule, target, max_k = match.groups()
        rule_label = {
            "relative_0p05": "cal rel 5%",
            "std_0": "cal exact",
            "std_0p25": "cal 0.25 SD",
            "std_0p5": "cal 0.5 SD",
            "std_1": "cal 1 SD",
        }.get(rule, rule)
        return f"{rule_label}, max {max_k}"

    match = re.fullmatch(r"elbow_max_(\d+)", selector)
    if match:
        return f"elbow max {match.group(1)}"

    match = re.fullmatch(r"score_gap_(relative_0p05|std_0|std_0p25|std_0p5|std_1)_budget_(\d+)", selector)
    if match:
        rule, budget = match.groups()
        rule_label = {
            "relative_0p05": "gap rel 5%",
            "std_0": "gap exact",
            "std_0p25": "gap 0.25 SD",
            "std_0p5": "gap 0.5 SD",
            "std_1": "gap 1 SD",
        }.get(rule, rule)
        return f"{rule_label}, K≤{budget}"

    match = re.fullmatch(r"pareto_front_cap_(\d+)", selector)
    if match:
        return f"Pareto cap {match.group(1)}"

    match = re.fullmatch(r"consensus_top_(\d+)_vote_(\d+)_cap_(\d+)", selector)
    if match:
        k0, vote, cap = match.groups()
        return f"consensus {vote}+ votes, cap {cap}"

    return selector.replace("_", " ")


def _point_label(row: pd.Series) -> str:
    return f"{_method_label(row['method'])} | {_selector_label(row['selector'])}"


def _coverage_column(summary: pd.DataFrame) -> str | None:
    preferred = [
        "relative_0p05_coverage",
        "std_0p5_coverage",
        "std_1_coverage",
        "exact_best_coverage",
    ]

    for col in preferred:
        if col in summary.columns:
            return col

    candidates = [
        col for col in summary.columns
        if col.endswith("_coverage")
        and not col.startswith("calibration_")
    ]

    if not candidates:
        return None

    relative = [col for col in candidates if col.startswith("relative_")]
    if relative:
        return sorted(relative)[0]

    std = [col for col in candidates if col.startswith("std_")]
    if std:
        return sorted(std)[0]

    return sorted(candidates)[0]


def _loss_column(summary: pd.DataFrame) -> str | None:
    if "best_in_set_performance_loss" in summary.columns:
        return "best_in_set_performance_loss"
    return None


def _extract_fixed_k(selector: str) -> int | None:
    match = re.search(r"fixed_top_(\d+)", str(selector))
    if not match:
        return None
    return int(match.group(1))


def _pareto_indices(df: pd.DataFrame,
                    *,
                    x_col: str,
                    y_col: str,
                    y_higher_is_better: bool) -> list[int]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    keep = []

    for i in range(len(df)):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue

        if y_higher_is_better:
            dominated = (
                (x <= x[i])
                & (y >= y[i])
                & ((x < x[i]) | (y > y[i]))
            )
        else:
            dominated = (
                (x <= x[i])
                & (y <= y[i])
                & ((x < x[i]) | (y < y[i]))
            )

        if not np.any(dominated):
            keep.append(i)

    return keep


def _is_interesting(row: pd.Series) -> bool:
    method = str(row.get("method", ""))
    selector = str(row.get("selector", ""))
    family = str(row.get("selector_family", ""))

    if method in {"lightgbm_lambdarank", "mlp_listnet"} and family == "fixed_top_k":
        return True

    if method == "lightgbm_lambdarank" and "cap_10" in selector:
        return True

    if method == "lightgbm_lambdarank" and "uncapped" in selector and (
        "relative_0p05" in selector or "std_0p5" in selector
    ):
        return True

    if method in {"composite_equal", "rrf_equal"} and "uncapped" in selector and "relative_0p05" in selector:
        return True

    if family in {"pareto", "consensus"}:
        return True

    return False


def _label_priority(df: pd.DataFrame,
                    *,
                    x_col: str,
                    y_col: str,
                    y_higher_is_better: bool) -> pd.DataFrame:
    work = df.copy()

    x = pd.to_numeric(work[x_col], errors="coerce")
    y = pd.to_numeric(work[y_col], errors="coerce")

    x_rank = x.rank(method="average", ascending=True, pct=True)

    if y_higher_is_better:
        y_rank = y.rank(method="average", ascending=False, pct=True)
    else:
        y_rank = y.rank(method="average", ascending=True, pct=True)

    work["_label_priority"] = x_rank + y_rank
    work["_interesting"] = work.apply(_is_interesting, axis=1).astype(int)

    work = work.sort_values(
        ["_interesting", "_label_priority", x_col],
        ascending=[False, True, True],
    )

    return work


def _annotate_sparse(ax,
                     df: pd.DataFrame,
                     *,
                     x_col: str,
                     y_col: str,
                     y_higher_is_better: bool,
                     max_labels: int = MAX_SCATTER_LABELS,
                     min_distance_pixels: int = MIN_LABEL_DISTANCE_PIXELS) -> None:
    if df.empty or max_labels <= 0:
        return

    candidates = _label_priority(
        df,
        x_col=x_col,
        y_col=y_col,
        y_higher_is_better=y_higher_is_better,
    )

    placed_pixels: list[np.ndarray] = []
    n_labels = 0

    ax.figure.canvas.draw()

    for _, row in candidates.iterrows():
        if n_labels >= max_labels:
            break

        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        pixel = ax.transData.transform((x, y))

        if any(np.linalg.norm(pixel - old_pixel) < min_distance_pixels for old_pixel in placed_pixels):
            continue

        label = _point_label(row)

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.9,
        )

        placed_pixels.append(pixel)
        n_labels += 1


def _method_marker_map(values: Iterable[str]) -> dict[str, str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    out = {}
    for i, value in enumerate(values):
        out[value] = markers[i % len(markers)]
    return out


def _outside_legend(ax, *, fontsize: int = 8) -> None:
    ax.legend(
        frameon=False,
        fontsize=fontsize,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )


def plot_coverage_size_frontier(summary: pd.DataFrame,
                                outdir: Path,
                                *,
                                formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)
    if coverage_col is None or "average_set_size" not in summary.columns:
        return []

    df = summary.dropna(subset=["average_set_size", coverage_col]).copy()
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(9, 5.8))

    families = list(df["selector_family"].dropna().unique())
    markers = _method_marker_map(families)

    for family, g in df.groupby("selector_family", sort=False):
        ax.scatter(
            g["average_set_size"],
            g[coverage_col],
            label=_method_label(family),
            marker=markers.get(family, "o"),
            alpha=0.65,
            s=34,
        )

    pareto_pos = _pareto_indices(
        df,
        x_col="average_set_size",
        y_col=coverage_col,
        y_higher_is_better=True,
    )
    pareto_df = df.iloc[pareto_pos].copy()

    ax.scatter(
        pareto_df["average_set_size"],
        pareto_df[coverage_col],
        facecolors="none",
        edgecolors="black",
        s=75,
        linewidths=0.8,
        label="Pareto frontier",
    )

    _annotate_sparse(
        ax,
        pareto_df,
        x_col="average_set_size",
        y_col=coverage_col,
        y_higher_is_better=True,
        max_labels=MAX_SCATTER_LABELS,
    )

    ax.set_xlabel("Average shortlist size")
    ax.set_ylabel(_metric_label(coverage_col))
    ax.set_title("Coverage-size frontier")
    ax.grid(True, alpha=0.25)
    _outside_legend(ax)

    return _save(fig, outdir, "fig1_coverage_size_frontier", formats)


def plot_loss_size_frontier(summary: pd.DataFrame,
                            outdir: Path,
                            *,
                            formats: Sequence[str]) -> list[Path]:
    loss_col = _loss_column(summary)
    if loss_col is None or "average_set_size" not in summary.columns:
        return []

    df = summary.dropna(subset=["average_set_size", loss_col]).copy()
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(9, 5.8))

    families = list(df["selector_family"].dropna().unique())
    markers = _method_marker_map(families)

    for family, g in df.groupby("selector_family", sort=False):
        ax.scatter(
            g["average_set_size"],
            g[loss_col],
            label=_method_label(family),
            marker=markers.get(family, "o"),
            alpha=0.65,
            s=34,
        )

    pareto_pos = _pareto_indices(
        df,
        x_col="average_set_size",
        y_col=loss_col,
        y_higher_is_better=False,
    )
    pareto_df = df.iloc[pareto_pos].copy()

    ax.scatter(
        pareto_df["average_set_size"],
        pareto_df[loss_col],
        facecolors="none",
        edgecolors="black",
        s=75,
        linewidths=0.8,
        label="Pareto frontier",
    )

    _annotate_sparse(
        ax,
        pareto_df,
        x_col="average_set_size",
        y_col=loss_col,
        y_higher_is_better=False,
        max_labels=MAX_SCATTER_LABELS,
    )

    ax.set_xlabel("Average shortlist size")
    ax.set_ylabel("Best-in-set performance loss")
    ax.set_title("Best-in-set loss versus shortlist size")
    ax.grid(True, alpha=0.25)
    _outside_legend(ax)

    return _save(fig, outdir, "fig2_loss_size_frontier", formats)


def _best_fixed_topk_method(summary: pd.DataFrame,
                            coverage_col: str) -> str | None:
    fixed = summary[summary["selector_family"] == "fixed_top_k"].copy()
    if fixed.empty:
        return None

    fixed["_k"] = fixed["selector"].map(_extract_fixed_k)
    fixed = fixed.dropna(subset=["_k", coverage_col]).copy()
    if fixed.empty:
        return None

    max_k = fixed["_k"].max()
    candidates = fixed[fixed["_k"] == max_k].copy()

    sort_cols = [coverage_col]
    ascending = [False]

    if "best_in_set_performance_loss" in candidates.columns:
        sort_cols.append("best_in_set_performance_loss")
        ascending.append(True)

    candidates = candidates.sort_values(sort_cols, ascending=ascending)
    return str(candidates.iloc[0]["method"])


def plot_best_method_budget_curve(summary: pd.DataFrame,
                                  outdir: Path,
                                  *,
                                  formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)
    loss_col = _loss_column(summary)

    if coverage_col is None or loss_col is None:
        return []

    method = _best_fixed_topk_method(summary, coverage_col)
    if method is None:
        return []

    fixed = summary[
        (summary["selector_family"] == "fixed_top_k")
        & (summary["method"] == method)
    ].copy()
    fixed["_k"] = fixed["selector"].map(_extract_fixed_k)
    fixed = fixed.dropna(subset=["_k", coverage_col, loss_col]).copy()

    if fixed.empty:
        return []

    fixed = fixed.sort_values("_k")

    fig, ax1 = plt.subplots(figsize=(7, 4.8))

    ax1.plot(fixed["_k"], fixed[coverage_col], marker="o", label=_metric_label(coverage_col))
    ax1.set_xlabel("Shortlist budget K")
    ax1.set_ylabel(_metric_label(coverage_col))
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(fixed["_k"], fixed[loss_col], marker="s", linestyle="--", label="Best-in-set loss")
    ax2.set_ylabel("Best-in-set performance loss")

    ax1.set_title(f"Budget curve for {_method_label(method)}")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, fontsize=8)

    return _save(fig, outdir, "fig3_best_method_budget_curve", formats)


def plot_exact_vs_near(summary: pd.DataFrame,
                       outdir: Path,
                       *,
                       formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)

    if coverage_col is None or coverage_col == "exact_best_coverage":
        return []

    if "exact_best_coverage" not in summary.columns:
        return []

    df = summary.dropna(subset=["exact_best_coverage", coverage_col, "average_set_size"]).copy()
    if df.empty:
        return []

    sizes = pd.to_numeric(df["average_set_size"], errors="coerce").to_numpy(dtype=float)
    size_scaled = 25 + 130 * (sizes - np.nanmin(sizes)) / max(np.nanmax(sizes) - np.nanmin(sizes), 1e-12)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.scatter(
        df["exact_best_coverage"],
        df[coverage_col],
        s=size_scaled,
        alpha=0.55,
    )

    pareto_pos = _pareto_indices(
        df,
        x_col="exact_best_coverage",
        y_col=coverage_col,
        y_higher_is_better=True,
    )
    pareto_df = df.iloc[pareto_pos].copy()

    ax.scatter(
        pareto_df["exact_best_coverage"],
        pareto_df[coverage_col],
        facecolors="none",
        edgecolors="black",
        s=75,
        linewidths=0.8,
    )

    _annotate_sparse(
        ax,
        pareto_df,
        x_col="exact_best_coverage",
        y_col=coverage_col,
        y_higher_is_better=True,
        max_labels=MAX_SCATTER_LABELS,
    )

    ax.set_xlabel("Exact-best coverage")
    ax.set_ylabel(_metric_label(coverage_col))
    ax.set_title("Exact-best coverage versus near-best coverage")
    ax.grid(True, alpha=0.25)

    return _save(fig, outdir, "fig4_exact_vs_near_best", formats)


def plot_conformal_size_pathology(summary: pd.DataFrame,
                                  outdir: Path,
                                  *,
                                  formats: Sequence[str]) -> list[Path]:
    if "selector_family" not in summary.columns or "average_set_size" not in summary.columns:
        return []

    df = summary[
        (summary["selector_family"] == "conformal_rank_top_k")
        & (summary["selector"].astype(str).str.contains("uncapped", case=False, na=False))
    ].copy()

    if df.empty:
        return []

    df = df.sort_values("average_set_size", ascending=False).head(MAX_BAR_LABELS)
    labels = [
        f"{_method_label(row.method)} | {_selector_label(row.selector)}"
        for row in df.itertuples(index=False)
    ]

    height = max(5, min(10, 0.32 * len(df) + 2))
    fig, ax = plt.subplots(figsize=(9, height))

    y = np.arange(len(df))
    ax.barh(y, df["average_set_size"], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Average conformal set size")
    ax.set_title("Largest uncapped conformal set sizes")
    ax.grid(True, axis="x", alpha=0.25)

    return _save(fig, outdir, "fig5_conformal_set_size_pathology", formats)


def plot_fixed_budget_method_comparison(summary: pd.DataFrame,
                                        outdir: Path,
                                        *,
                                        formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)
    if coverage_col is None:
        return []

    fixed = summary[summary["selector_family"] == "fixed_top_k"].copy()
    if fixed.empty:
        return []

    fixed["_k"] = fixed["selector"].map(_extract_fixed_k)
    fixed = fixed.dropna(subset=["_k", coverage_col]).copy()
    if fixed.empty:
        return []

    pivot = fixed.pivot_table(
        index="method",
        columns="_k",
        values=coverage_col,
        aggfunc="mean",
    )

    if pivot.empty:
        return []

    max_k = max(pivot.columns)
    pivot = pivot.sort_values(max_k, ascending=False)

    methods = list(pivot.index)
    ks = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(methods)), 4.8))

    x = np.arange(len(methods))
    width = 0.8 / max(len(ks), 1)

    for i, k in enumerate(ks):
        ax.bar(
            x + (i - (len(ks) - 1) / 2) * width,
            pivot[k].to_numpy(),
            width=width,
            label=f"K={int(k)}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=35, ha="right")
    ax.set_ylabel(_metric_label(coverage_col))
    ax.set_title("Fixed-budget method comparison")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    return _save(fig, outdir, "fig6_fixed_budget_method_comparison", formats)


def plot_calibration_target_vs_achieved(summary: pd.DataFrame,
                                        outdir: Path,
                                        *,
                                        formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)
    if coverage_col is None or "target_coverage" not in summary.columns:
        return []

    df = summary[
        summary["selector_family"].astype(str).eq("calibrated_min_k")
    ].dropna(subset=["target_coverage", coverage_col]).copy()

    if df.empty:
        return []

    x = pd.to_numeric(df["target_coverage"], errors="coerce")
    if x.max() <= 1.0:
        x = x * 100.0

    df["_target_percent"] = x

    fig, ax = plt.subplots(figsize=(7, 5.6))

    methods = list(df["method"].dropna().unique())
    markers = _method_marker_map(methods)

    for method, g in df.groupby("method", sort=False):
        ax.scatter(
            g["_target_percent"],
            g[coverage_col],
            label=_method_label(method),
            marker=markers.get(method, "o"),
            alpha=0.65,
            s=34,
        )

    lo = min(df["_target_percent"].min(), df[coverage_col].min())
    hi = max(df["_target_percent"].max(), df[coverage_col].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_xlabel("Calibration target coverage")
    ax.set_ylabel(f"Held-out {_metric_label(coverage_col)}")
    ax.set_title("Calibration target versus achieved held-out coverage")
    ax.grid(True, alpha=0.25)
    _outside_legend(ax, fontsize=7)

    return _save(fig, outdir, "fig7_calibration_target_vs_achieved", formats)


def plot_set_size_distribution(summary: pd.DataFrame,
                               per_query: pd.DataFrame,
                               outdir: Path,
                               *,
                               formats: Sequence[str]) -> list[Path]:
    coverage_col = _coverage_column(summary)
    if coverage_col is None or per_query.empty:
        return []

    method = _best_fixed_topk_method(summary, coverage_col)
    selected_rows = []

    if method is not None:
        fixed = summary[
            (summary["method"] == method)
            & (summary["selector_family"] == "fixed_top_k")
        ].copy()
        fixed["_k"] = fixed["selector"].map(_extract_fixed_k)
        fixed = fixed.sort_values("_k")
        selected_rows.append(fixed[["method", "selector"]])

        conformal = summary[
            (summary["method"] == method)
            & (summary["selector_family"] == "conformal_rank_top_k")
        ].copy()
        conformal = conformal[
            conformal["selector"].astype(str).str.contains("uncapped|cap_10", case=False, na=False)
        ].sort_values("average_set_size").head(4)
        selected_rows.append(conformal[["method", "selector"]])

    top_extra = summary.sort_values(
        [coverage_col, "average_set_size"],
        ascending=[False, True],
    ).head(5)
    selected_rows.append(top_extra[["method", "selector"]])

    selected = pd.concat(selected_rows, ignore_index=True).drop_duplicates()
    if selected.empty:
        return []

    work = per_query.merge(selected, on=["method", "selector"], how="inner")
    if work.empty:
        return []

    work["_label"] = work["method"].map(_method_label) + " | " + work["selector"].map(_selector_label)

    labels = []
    data = []

    for label, g in work.groupby("_label", sort=False):
        labels.append(label)
        data.append(pd.to_numeric(g["set_size"], errors="coerce").dropna().to_numpy())

    if not data:
        return []

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 4.8))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Set size")
    ax.set_title("Distribution of selected set sizes")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)

    return _save(fig, outdir, "fig8_set_size_distribution", formats)


def make_all_shortlist_plots(summary: pd.DataFrame,
                             per_query: pd.DataFrame,
                             outdir: str | Path,
                             *,
                             performance_col: str,
                             formats: Sequence[str] = ("png", "pdf"),
                             strict: bool = False) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plotters = [
        ("coverage_size_frontier", plot_coverage_size_frontier),
        ("loss_size_frontier", plot_loss_size_frontier),
        ("best_method_budget_curve", plot_best_method_budget_curve),
        ("exact_vs_near_best", plot_exact_vs_near),
        ("conformal_set_size_pathology", plot_conformal_size_pathology),
        ("fixed_budget_method_comparison", plot_fixed_budget_method_comparison),
        ("calibration_target_vs_achieved", plot_calibration_target_vs_achieved),
    ]

    rows = []

    for name, plotter in plotters:
        try:
            paths = plotter(summary, outdir, formats=formats)
            for path in paths:
                rows.append({
                    "performance_col": performance_col,
                    "plot": name,
                    "path": str(path),
                    "status": "written",
                    "message": "",
                })
        except Exception as exc:
            if strict:
                raise
            rows.append({
                "performance_col": performance_col,
                "plot": name,
                "path": "",
                "status": "failed",
                "message": str(exc),
            })

    try:
        paths = plot_set_size_distribution(summary, per_query, outdir, formats=formats)
        for path in paths:
            rows.append({
                "performance_col": performance_col,
                "plot": "set_size_distribution",
                "path": str(path),
                "status": "written",
                "message": "",
            })
    except Exception as exc:
        if strict:
            raise
        rows.append({
            "performance_col": performance_col,
            "plot": "set_size_distribution",
            "path": "",
            "status": "failed",
            "message": str(exc),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(outdir / f"plot_manifest_{performance_col}.csv", index=False)
    return manifest