import argparse
from pathlib import Path
import pandas as pd

from src.data import load_transfer_data, normalize_query_features, get_query_cols
from src.validation import validate_dataset
from src.offline.calibration import load_calibration, ranker_from_calibration
from src.rankers.single import SingleFeatureRanker
from src.rankers.composite import CompositeDistanceRanker
from src.rankers.rrf import RRFRanker
from src.rankers.random import RandomRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online source-language selection on new CSV files.",
    )
    parser.add_argument('--csv', nargs='+', required=True)
    parser.add_argument('--dataset_names', nargs='*', default=None)
    parser.add_argument('--features', nargs='+',
                        default=['new_gen', 'new_typ', 'new_geo', 'script'])
    parser.add_argument('--target_col', default='task_lang')
    parser.add_argument('--source_col', default='transfer_lang')
    parser.add_argument('--dataset_col', default='dataset')
    parser.add_argument('--normalizer', default='minmax',
                        choices=['none', 'minmax'])
    parser.add_argument('--method', required=True,
                        choices=[
                            'random',
                            'single',
                            'composite_equal',
                            'rrf_equal',
                            'calibrated',
                        ])
    parser.add_argument('--single_feature', default=None,
                        help="Feature name to use when --method single.")
    parser.add_argument('--calibration_json', default=None,
                        help="Calibration JSON for --method calibrated.")
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--outdir', default='artifacts/inference')
    return parser.parse_args()


def make_ranker(args: argparse.Namespace):
    if args.method == 'random':
        return RandomRanker(random_state=args.random_state)

    if args.method == 'single':
        if args.single_feature is None:
            raise ValueError("--single_feature is required when --method single")
        if args.single_feature not in args.features:
            raise ValueError(f"{args.single_feature} is not in --features")
        idx = args.features.index(args.single_feature)
        return SingleFeatureRanker(feature_idx=idx, ascending=True)

    if args.method == 'composite_equal':
        return CompositeDistanceRanker(weights=None)

    if args.method == 'rrf_equal':
        return RRFRanker(weights=None, rrf_k=60.0, ascending=True)

    if args.method == 'calibrated':
        if args.calibration_json is None:
            raise ValueError("--calibration_json is required when --method calibrated")
        calibration = load_calibration(args.calibration_json)
        return ranker_from_calibration(calibration)

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

    validate_dataset(
        raw_df,
        feature_cols=args.features,
        target_col=args.target_col,
        source_col=args.source_col,
        performance_col='__not_required__',
        dataset_col=args.dataset_col,
        require_performance=False,
    )

    df = normalize_query_features(
        raw_df,
        feature_cols=args.features,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        method=args.normalizer,
    )

    ranker = make_ranker(args)
    query_cols = get_query_cols(df, target_col=args.target_col, dataset_col=args.dataset_col)

    ranking_records = []
    selected_records = []

    for query_key, qdf in df.groupby(query_cols, sort=False):
        X = qdf[args.features].to_numpy(dtype=float)
        ranker.fit(X)
        scores = ranker.predict(X)

        block = qdf.copy()
        block['score'] = scores
        block['rank'] = block['score'].rank(method='first', ascending=False).astype(int)
        block = block.sort_values('rank')

        ranking_records.append(block)

        best = block.iloc[0]
        record = {
            'target_lang': best[args.target_col],
            'selected_source': best[args.source_col],
            'score': best['score'],
        }
        if args.dataset_col in best.index:
            record['dataset'] = best[args.dataset_col]
        selected_records.append(record)

    rankings = pd.concat(ranking_records, ignore_index=True)
    selected = pd.DataFrame(selected_records)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rankings_path = outdir / f'rankings_{args.method}.csv'
    selected_path = outdir / f'selected_sources_{args.method}.csv'

    rankings.to_csv(rankings_path, index=False)
    selected.to_csv(selected_path, index=False)

    print(selected.to_string(index=False))
    print(f"\nWrote {rankings_path}")
    print(f"Wrote {selected_path}")


if __name__ == '__main__':
    main()