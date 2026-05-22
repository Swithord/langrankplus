import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import get_query_cols, load_transfer_data, normalize_query_features
from src.rankers.composite import CompositeDistanceRanker
from src.rankers.rrf import RRFRanker
from src.rankers.single import SingleFeatureRanker


def load_fitted_ranker(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    method = payload.get("method", "")

    if method.startswith("composite"):
        return CompositeDistanceRanker.load(path)

    if method.startswith("rrf"):
        return RRFRanker.load(path)

    raise ValueError(f"Unknown fitted ranker method in {path}: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-language ranking on an unlabelled CSV.",
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--features", nargs="+",
                        default=["new_gen", "new_typ", "new_geo", "script"])
    parser.add_argument("--target_col", default="task_lang")
    parser.add_argument("--source_col", default="transfer_lang")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--normalizer", default="minmax",
                        choices=["none", "minmax"])
    parser.add_argument("--method", required=True,
                        choices=["single", "composite_equal", "rrf_equal", "calibrated"])
    parser.add_argument("--single_feature", default=None)
    parser.add_argument("--calibration_json", default=None)
    parser.add_argument("--outdir", default="artifacts/inference")
    return parser.parse_args()


def build_ranker(args: argparse.Namespace):
    if args.method == "single":
        if args.single_feature is None:
            raise ValueError("--single_feature is required when --method single")
        if args.single_feature not in args.features:
            raise ValueError("--single_feature must be one of --features")
        feature_idx = args.features.index(args.single_feature)
        return SingleFeatureRanker(feature_idx=feature_idx, ascending=True), f"single_{args.single_feature}"

    if args.method == "composite_equal":
        return CompositeDistanceRanker(weights=None, trainable=False), "composite_equal"

    if args.method == "rrf_equal":
        return RRFRanker(weights=None, rrf_k=60.0, ascending=True, trainable=False), "rrf_equal"

    if args.method == "calibrated":
        if args.calibration_json is None:
            raise ValueError("--calibration_json is required when --method calibrated")
        ranker = load_fitted_ranker(args.calibration_json)
        return ranker, Path(args.calibration_json).stem

    raise ValueError(f"Unknown method: {args.method}")


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

    ranker, method_name = build_ranker(args)

    query_cols = get_query_cols(
        df,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )

    ranking_frames = []
    selected_records = []

    for query_key, qdf in df.groupby(query_cols, sort=False):
        X = qdf[args.features].to_numpy(dtype=float)
        scores = ranker.predict(X)

        out = qdf.copy()
        out["score"] = scores
        out["rank"] = out["score"].rank(method="first", ascending=False).astype(int)
        out = out.sort_values("rank")

        ranking_frames.append(out)

        best = out.iloc[0]
        record = {
            "method": method_name,
            "target_lang": best[args.target_col],
            "selected_source": best[args.source_col],
            "score": best["score"],
        }

        if args.dataset_col in best.index:
            record["dataset"] = best[args.dataset_col]

        selected_records.append(record)

    rankings = pd.concat(ranking_frames, ignore_index=True)
    selected = pd.DataFrame(selected_records)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rankings_path = outdir / f"rankings_{method_name}.csv"
    selected_path = outdir / f"selected_sources_{method_name}.csv"

    rankings.to_csv(rankings_path, index=False)
    selected.to_csv(selected_path, index=False)

    print(f"Wrote {rankings_path}")
    print(f"Wrote {selected_path}")


if __name__ == "__main__":
    main()