from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = [
    ("sib200", "xlm-r", "SIB-200", "XLM-R"),
    ("sib200", "mt5", "SIB-200", "mT5"),
    ("taxi1500", "xlm-r", "Taxi1500", "XLM-R"),
    ("taxi1500", "mt5", "Taxi1500", "mT5"),
    ("ud_pos", "xlm-r", "UD-POS", "XLM-R"),
    ("ud_pos", "mt5", "UD-POS", "mT5"),
]

METHODS = [
    ("lightgbm_lambdarank", "LightGBM", "Trained rankers"),
    ("mlp_listnet", "MLP", "Trained rankers"),
    ("composite_equal", "Comp-Eq", "Composite distances"),
    ("rrf_equal", "Comp-RRF", "Composite distances"),
    ("single_new_gen", "Genetic", "Individual distances"),
    ("single_new_typ", "Typological", "Individual distances"),
    ("single_new_geo", "Geographic", "Individual distances"),
    ("single_script", "Script", "Individual distances"),
    ("single_distals_asjp", "ASJP", "Individual distances"),
    ("single_distals_wiki_size", "Wiki-size", "Individual distances"),
]

RESOURCE_LABELS = {
    "hrl": "HRL targets",
    "mrl": "MRL targets",
    "lrl": "LRL targets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Mondrian CTC outputs into paper-style tables."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/mondrian_ctc"),
        help="Directory containing one <task>_<model> subdirectory per run.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/mondrian_ctc/table"),
    )
    parser.add_argument("--performance_col", default="f1_score")
    return parser.parse_args()


def load_results(root: Path, performance_col: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []

    for task_key, model_key, task_label, model_label in SETTINGS:
        path = (
            root
            / f"{task_key}_{model_key}"
            / f"mondrian_ctc_{performance_col}.csv"
        )
        if not path.exists():
            missing.append(path)
            continue

        frame = pd.read_csv(path)
        frame["task_key"] = task_key
        frame["model_key"] = model_key
        frame["task"] = task_label
        frame["model"] = model_label
        frames.append(frame)

    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing Mondrian CTC result files:\n" + joined
        )

    if not frames:
        raise ValueError("No Mondrian CTC files were loaded.")

    out = pd.concat(frames, ignore_index=True)
    expected_methods = {method for method, _, _ in METHODS}
    observed_methods = set(out["method"].astype(str))
    missing_methods = sorted(expected_methods - observed_methods)
    if missing_methods:
        raise ValueError(
            "The following paper-table methods are missing from the evaluation "
            f"outputs: {', '.join(missing_methods)}"
        )

    return out


def _lookup(
    df: pd.DataFrame,
    *,
    method: str,
    resource_level: str,
    task_key: str,
    model_key: str,
) -> pd.Series | None:
    rows = df.loc[
        (df["method"] == method)
        & (df["resource_level"] == resource_level)
        & (df["task_key"] == task_key)
        & (df["model_key"] == model_key)
    ]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(
            "Expected one row for "
            f"{method}/{resource_level}/{task_key}/{model_key}, found {len(rows)}."
        )
    return rows.iloc[0]


def _format_ctc(row: pd.Series | None, *, bold: bool) -> str:
    if row is None or not np.isfinite(row["mean_trial_complexity"]):
        return "--"

    value = str(int(round(float(row["mean_trial_complexity"]))))
    if float(row.get("infinite_threshold_rate_pct", 0.0)) > 0.0:
        value += r"\textsuperscript{\dagger}"
    if bold:
        value = rf"\textbf{{{value}}}"
    return value


def _format_hit_rate(row: pd.Series | None) -> str:
    if row is None or not np.isfinite(row["near_oracle_hit_rate_pct"]):
        return "--"
    rate = float(row["near_oracle_hit_rate_pct"])
    se = float(row["near_oracle_hit_rate_se_pct"])
    if np.isfinite(se):
        return rf"${rate:.1f} \pm {se:.1f}$"
    return rf"${rate:.1f}$"


def make_group_csv(df: pd.DataFrame, resource_level: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for method, method_label, paradigm in METHODS:
        record: dict[str, object] = {
            "paradigm": paradigm,
            "method": method_label,
        }
        for task_key, model_key, task_label, model_label in SETTINGS:
            row = _lookup(
                df,
                method=method,
                resource_level=resource_level,
                task_key=task_key,
                model_key=model_key,
            )
            prefix = f"{task_key}_{model_key}"
            if row is None:
                record[f"{prefix}_ctc"] = np.nan
                record[f"{prefix}_hit_rate_pct"] = np.nan
                record[f"{prefix}_hit_rate_se_pct"] = np.nan
                record[f"{prefix}_n"] = 0
                record[f"{prefix}_mean_calibration_size"] = np.nan
                record[f"{prefix}_infinite_threshold_rate_pct"] = np.nan
            else:
                record[f"{prefix}_ctc"] = row["mean_trial_complexity"]
                record[f"{prefix}_hit_rate_pct"] = row["near_oracle_hit_rate_pct"]
                record[f"{prefix}_hit_rate_se_pct"] = row[
                    "near_oracle_hit_rate_se_pct"
                ]
                record[f"{prefix}_n"] = int(row["n_folds"])
                record[f"{prefix}_mean_calibration_size"] = row[
                    "mean_calibration_size"
                ]
                record[f"{prefix}_infinite_threshold_rate_pct"] = row[
                    "infinite_threshold_rate_pct"
                ]
        records.append(record)

    return pd.DataFrame(records)


def make_latex_table(df: pd.DataFrame, resource_level: str) -> str:
    minima: dict[tuple[str, str], float] = {}
    for task_key, model_key, _, _ in SETTINGS:
        values = []
        for method, _, _ in METHODS:
            row = _lookup(
                df,
                method=method,
                resource_level=resource_level,
                task_key=task_key,
                model_key=model_key,
            )
            if row is not None and np.isfinite(row["mean_trial_complexity"]):
                values.append(float(row["mean_trial_complexity"]))
        minima[(task_key, model_key)] = min(values) if values else float("nan")

    counts = []
    for task_key in ["sib200", "taxi1500", "ud_pos"]:
        rows = df.loc[
            (df["resource_level"] == resource_level)
            & (df["task_key"] == task_key)
        ]
        n = int(rows["n_folds"].iloc[0]) if not rows.empty else 0
        label = next(x[2] for x in SETTINGS if x[0] == task_key)
        counts.append(f"{label}: $n={n}$")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{12}{c}}",
        r"\toprule",
        r"Method",
        r"& \multicolumn{4}{c}{SIB-200}",
        r"& \multicolumn{4}{c}{Taxi1500}",
        r"& \multicolumn{4}{c}{UD-POS} \\",
        r"\cmidrule(lr){2-5}",
        r"\cmidrule(lr){6-9}",
        r"\cmidrule(lr){10-13}",
        r"& \multicolumn{2}{c}{XLM-R}",
        r"& \multicolumn{2}{c}{mT5}",
        r"& \multicolumn{2}{c}{XLM-R}",
        r"& \multicolumn{2}{c}{mT5}",
        r"& \multicolumn{2}{c}{XLM-R}",
        r"& \multicolumn{2}{c}{mT5} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9}",
        r"\cmidrule(lr){10-11}",
        r"\cmidrule(lr){12-13}",
        r"& CTC & HR & CTC & HR & CTC & HR & CTC & HR & CTC & HR & CTC & HR \\",
        r"\midrule",
    ]

    current_paradigm: str | None = None
    for method, method_label, paradigm in METHODS:
        if paradigm != current_paradigm:
            if current_paradigm is not None:
                lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{13}}{{l}}{{\textit{{{paradigm}}}}} \\" 
            )
            current_paradigm = paradigm

        cells = [method_label]
        for task_key, model_key, _, _ in SETTINGS:
            row = _lookup(
                df,
                method=method,
                resource_level=resource_level,
                task_key=task_key,
                model_key=model_key,
            )
            minimum = minima[(task_key, model_key)]
            bold = (
                row is not None
                and np.isfinite(minimum)
                and np.isclose(float(row["mean_trial_complexity"]), minimum)
            )
            cells.extend([
                _format_ctc(row, bold=bold),
                _format_hit_rate(row),
            ])
        lines.append(" & ".join(cells) + r" \\")

    group_title = RESOURCE_LABELS[resource_level]
    count_text = "; ".join(counts)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        (
            rf"\caption{{Resource-level Mondrian conformal trial complexity for "
            rf"{group_title}. CTC is the mean calibrated shortlist size and HR is "
            rf"the near-oracle hit rate reported as mean $\pm$ one standard error. "
            rf"{count_text}. Lower CTC is better. A dagger marks a setting in "
            rf"which at least one held-out fold had an infinite conformal quantile, "
            rf"so the valid shortlist expanded to the full candidate-source pool.}}"
        ),
        rf"\label{{tab:mondrian-ctc-{resource_level}}}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    combined = load_results(args.root, args.performance_col)
    combined.to_csv(args.outdir / "mondrian_ctc_long.csv", index=False)

    latex_tables = []
    for resource_level in ["hrl", "mrl", "lrl"]:
        group_csv = make_group_csv(combined, resource_level)
        group_csv.to_csv(
            args.outdir / f"mondrian_ctc_{resource_level}.csv",
            index=False,
        )
        latex_tables.append(make_latex_table(combined, resource_level))

    tex_path = args.outdir / "mondrian_ctc_tables.tex"
    tex_path.write_text("\n\n".join(latex_tables) + "\n", encoding="utf-8")

    diagnostics_cols = [
        "task",
        "model",
        "method",
        "resource_level",
        "n_folds",
        "mean_trial_complexity",
        "near_oracle_hit_rate_pct",
        "near_oracle_hit_rate_se_pct",
        "mean_calibration_size",
        "minimum_finite_calibration_size",
        "finite_quantile_possible_rate_pct",
        "infinite_threshold_rate_pct",
        "mean_pool_fraction_pct",
    ]
    combined.loc[:, diagnostics_cols].to_csv(
        args.outdir / "mondrian_ctc_diagnostics.csv",
        index=False,
    )

    print(f"Wrote {args.outdir / 'mondrian_ctc_long.csv'}")
    print(f"Wrote {args.outdir / 'mondrian_ctc_hrl.csv'}")
    print(f"Wrote {args.outdir / 'mondrian_ctc_mrl.csv'}")
    print(f"Wrote {args.outdir / 'mondrian_ctc_lrl.csv'}")
    print(f"Wrote {args.outdir / 'mondrian_ctc_diagnostics.csv'}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()