import argparse
from pathlib import Path

import pandas as pd

from src.shortlist_plots import make_all_shortlist_plots
from src.shortlists import (
    PostHocShortlistEvaluator,
    ShortlistConfig,
    load_ranking_parquets,
    prepare_rankings,
    summarize_shortlists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate operational source-language shortlist selectors from saved ranking Parquets.",
    )

    parser.add_argument("--rankings", nargs="+", required=True)
    parser.add_argument("--performance_col", required=True)

    parser.add_argument("--method_col", default="auto")
    parser.add_argument("--dataset_col", default="auto")
    parser.add_argument("--target_col", default="auto")
    parser.add_argument("--source_col", default="auto")
    parser.add_argument("--score_col", default="auto")
    parser.add_argument("--rank_col", default="auto")

    parser.add_argument("--fixed_top_k", nargs="+", type=int, default=[3, 5, 10])

    parser.add_argument("--relative_epsilons", nargs="+", type=float, default=[0.05])
    parser.add_argument("--std_multipliers", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])

    parser.add_argument("--conformal_alphas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--conformal_caps", nargs="+", type=int, default=[3, 5, 10])

    parser.add_argument("--calibrated_targets", nargs="+", type=float, default=[0.75, 0.8, 0.9])
    parser.add_argument("--calibrated_max_k", nargs="+", type=int, default=[3, 5, 10])

    parser.add_argument("--score_gap_budgets", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--score_gap_tolerance", type=float, default=0.01)

    parser.add_argument("--elbow_max_k", nargs="+", type=int, default=[5, 10])

    parser.add_argument("--consensus_top_k0", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--consensus_vote_thresholds", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--consensus_caps", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--consensus_exclude_methods", nargs="*", default=["random"])

    parser.add_argument("--distance_features", nargs="*", default=["new_gen", "new_typ", "new_geo", "script"])
    parser.add_argument("--pareto_caps", nargs="+", type=int, default=[3, 5, 10])

    parser.add_argument("--skip_fixed_topk", action="store_true")
    parser.add_argument("--skip_conformal", action="store_true")
    parser.add_argument("--skip_calibrated_min_k", action="store_true")
    parser.add_argument("--skip_score_gap", action="store_true")
    parser.add_argument("--skip_elbow", action="store_true")
    parser.add_argument("--skip_consensus", action="store_true")
    parser.add_argument("--skip_pareto", action="store_true")

    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--plot_formats", nargs="+", default=["png", "pdf"])

    parser.add_argument("--outdir", default="artifacts/shortlists")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            non_null = out[col].dropna()

            if non_null.empty:
                out[col] = out[col].astype("string")
                continue

            observed_types = {type(x) for x in non_null}

            if len(observed_types) > 1:
                out[col] = out[col].astype("string")
            elif observed_types == {str}:
                out[col] = out[col].astype("string")

    return out


def main() -> None:
    args = parse_args()

    raw = load_ranking_parquets(args.rankings)

    rankings = prepare_rankings(
        raw,
        performance_col=args.performance_col,
        method_col=args.method_col,
        dataset_col=args.dataset_col,
        target_col=args.target_col,
        source_col=args.source_col,
        score_col=args.score_col,
        rank_col=args.rank_col,
    )

    config = ShortlistConfig(
        fixed_top_k=tuple(args.fixed_top_k),
        relative_epsilons=tuple(args.relative_epsilons),
        std_multipliers=tuple(args.std_multipliers),
        conformal_alphas=tuple(args.conformal_alphas),
        conformal_caps=tuple(args.conformal_caps),
        calibrated_targets=tuple(args.calibrated_targets),
        calibrated_max_k=tuple(args.calibrated_max_k),
        score_gap_budgets=tuple(args.score_gap_budgets),
        score_gap_tolerance=args.score_gap_tolerance,
        elbow_max_k=tuple(args.elbow_max_k),
        consensus_top_k0=tuple(args.consensus_top_k0),
        consensus_vote_thresholds=tuple(args.consensus_vote_thresholds),
        consensus_caps=tuple(args.consensus_caps),
        consensus_exclude_methods=tuple(args.consensus_exclude_methods),
        pareto_caps=tuple(args.pareto_caps),
        distance_features=tuple(args.distance_features),
        include_fixed_topk=not args.skip_fixed_topk,
        include_conformal=not args.skip_conformal,
        include_calibrated_min_k=not args.skip_calibrated_min_k,
        include_score_gap=not args.skip_score_gap,
        include_elbow=not args.skip_elbow,
        include_consensus=not args.skip_consensus,
        include_pareto=not args.skip_pareto,
    )

    evaluator = PostHocShortlistEvaluator(
        rankings,
        config,
        verbose=args.verbose,
    )

    per_query = evaluator.run()
    summary = summarize_shortlists(per_query)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_csv = outdir / f"shortlist_summary_{args.performance_col}.csv"
    per_query_csv = outdir / f"shortlist_per_query_{args.performance_col}.csv"
    summary_parquet = outdir / f"shortlist_summary_{args.performance_col}.parquet"
    per_query_parquet = outdir / f"shortlist_per_query_{args.performance_col}.parquet"

    summary.to_csv(summary_csv, index=False)
    per_query.to_csv(per_query_csv, index=False)

    summary_safe = make_parquet_safe(summary)
    per_query_safe = make_parquet_safe(per_query)

    summary_safe.to_parquet(summary_parquet, index=False)
    per_query_safe.to_parquet(per_query_parquet, index=False)

    print("\nShortlist summary")
    if summary.empty:
        print("No shortlist results were produced.")
    else:
        display_cols = [
            col for col in [
                "method",
                "selector_family",
                "selector",
                "n_queries",
                "average_set_size",
                "median_set_size",
                "p90_set_size",
                "relative_0p05_coverage",
                "std_0p5_coverage",
                "std_1_coverage",
                "best_in_set_performance_loss",
            ]
            if col in summary.columns
        ]
        print(summary[display_cols].head(50).to_string(index=False))

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {per_query_csv}")
    print(f"Wrote {summary_parquet}")
    print(f"Wrote {per_query_parquet}")

    if not args.skip_plots:
        plot_dir = outdir / "plots"
        manifest = make_all_shortlist_plots(
            summary,
            per_query,
            plot_dir,
            performance_col=args.performance_col,
            formats=args.plot_formats,
        )

        print("\nPlots")
        if manifest.empty:
            print("No plots were produced.")
        else:
            print(manifest.to_string(index=False))
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()