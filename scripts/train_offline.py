import argparse
from pathlib import Path

import numpy as np

from src.data import (
    add_query_id,
    get_query_cols,
    load_transfer_data,
    normalize_query_features,
)
from src.rankers.composite import CompositeDistanceRanker
from src.rankers.rrf import RRFRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit frozen composite/RRF source-selection rankers.",
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--features", nargs="+",
                        default=["new_gen", "new_typ", "new_geo", "script"])
    parser.add_argument("--performance_col", default="accuracy")
    parser.add_argument("--target_col", default="task_lang")
    parser.add_argument("--source_col", default="transfer_lang")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--methods", nargs="+",
                        default=["composite", "rrf"],
                        choices=["composite", "rrf"])
    parser.add_argument("--normalizer", default="minmax",
                        choices=["none", "minmax"])
    parser.add_argument("--top_k_relevance", type=int, default=10)
    parser.add_argument("--n_opt_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_pairs_per_query", type=int, default=5000)
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument("--rrf_k_grid", nargs="+", type=float,
                        default=[1, 5, 10, 20, 40, 60, 100])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--outdir", default="artifacts/offline_calibration")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def add_relevance_labels(df,
                         *,
                         target_col: str,
                         dataset_col: str,
                         performance_col: str,
                         top_k_relevance: int):
    out = df.copy()
    query_cols = get_query_cols(
        out,
        target_col=target_col,
        dataset_col=dataset_col,
    )

    ranks = out.groupby(query_cols)[performance_col].rank(
        method="min",
        ascending=False,
    )

    out["_relevance"] = np.where(
        ranks <= top_k_relevance,
        top_k_relevance + 1 - ranks,
        0.0,
    )

    out = add_query_id(
        out,
        target_col=target_col,
        dataset_col=dataset_col,
        query_id_col="_query_id",
    )

    return out


def main() -> None:
    args = parse_args()

    dataset_names = args.dataset_names
    if dataset_names == []:
        dataset_names = None

    raw_df = load_transfer_data(
        args.csv,
        dataset_names=dataset_names,
        dataset_col=args.dataset_col,
    )

    df = normalize_query_features(
        raw_df,
        feature_cols=args.features,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        method=args.normalizer,
    )

    df = add_relevance_labels(
        df,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        performance_col=args.performance_col,
        top_k_relevance=args.top_k_relevance,
    )

    X = df[args.features].to_numpy(dtype=float)
    y = df["_relevance"].to_numpy(dtype=float)
    groups = df["_query_id"].to_numpy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "composite" in args.methods:
        ranker = CompositeDistanceRanker(
            trainable=True,
            n_steps=args.n_opt_steps,
            learning_rate=args.learning_rate,
            max_pairs_per_query=args.max_pairs_per_query,
            score_scale=args.score_scale,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        ranker.fit(X, y, groups=groups)
        path = outdir / f"composite_{args.performance_col}.json"
        ranker.save(path)
        print(f"Wrote {path}")

    if "rrf" in args.methods:
        ranker = RRFRanker(
            trainable=True,
            rrf_k_grid=args.rrf_k_grid,
            n_steps=args.n_opt_steps,
            learning_rate=args.learning_rate,
            max_pairs_per_query=args.max_pairs_per_query,
            score_scale=args.score_scale,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        ranker.fit(X, y, groups=groups)
        path = outdir / f"rrf_{args.performance_col}.json"
        ranker.save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()