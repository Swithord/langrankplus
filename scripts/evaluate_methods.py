import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data import load_transfer_data, normalize_query_features
from src.evaluation import (
    TransferEvaluator,
    results_to_summary,
    results_to_per_fold,
    pairwise_comparisons,
)
from src.offline.calibration import (
    fit_composite_weights,
    fit_rrf_weights,
    load_calibration,
    ranker_from_calibration,
)
from src.rankers.single import SingleFeatureRanker
from src.rankers.composite import CompositeDistanceRanker
from src.rankers.rrf import RRFRanker
from src.rankers.random import RandomRanker
from src.rankers.lightgbm import LightGBMRanker
from src.rankers.mlp import MLPRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare source-language selection methods.",
    )
    parser.add_argument('--csv', nargs='+', required=True,
                        help="One or more transfer-performance CSV files.")
    parser.add_argument('--dataset_names', nargs='*', default=None)
    parser.add_argument('--features', nargs='+',
                        default=['new_gen', 'new_typ', 'new_geo', 'script'])
    parser.add_argument('--performance_col', default='accuracy')
    parser.add_argument('--target_col', default='task_lang')
    parser.add_argument('--source_col', default='transfer_lang')
    parser.add_argument('--dataset_col', default='dataset')
    parser.add_argument('--normalizer', default='minmax',
                        choices=['none', 'minmax'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--top_k_relevance', type=int, default=10)
    parser.add_argument('--val_size', type=float, default=0.0)
    parser.add_argument('--random_state', type=int, default=42)

    parser.add_argument('--skip_random', action='store_true')
    parser.add_argument('--skip_single', action='store_true')
    parser.add_argument('--skip_learned', action='store_true',
                        help="Skip LightGBMRanker and MLPRanker baselines.")

    parser.add_argument('--include_nested_offline', action='store_true',
                        help="Nested evaluation of fitted composite/RRF.")
    parser.add_argument('--n_calibration_samples', type=int, default=5000,
                        help="Number of random simplex samples for nested calibration.")
    parser.add_argument('--rrf_k_grid', nargs='+', type=float,
                        default=[1, 5, 10, 20, 40, 60, 100])

    parser.add_argument('--composite_calibration_json', default=None)
    parser.add_argument('--rrf_calibration_json', default=None)

    parser.add_argument('--include_conformal', action='store_true',
                        help="Add split-conformal source sets to evaluation outputs.")
    parser.add_argument('--conformal_alpha', type=float, default=0.1,
                        help="Miscoverage level for conformal source sets.")
    parser.add_argument('--conformal_cal_size', type=float, default=0.2,
                        help="Fraction of non-held-out queries used for conformal calibration.")

    parser.add_argument('--outdir', default='artifacts/evaluation')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def build_methods(args: argparse.Namespace):
    """
    Build all rankers to compare.
    """
    methods = []

    if not args.skip_random:
        methods.append((
            'random',
            RandomRanker(random_state=args.random_state),
            None,
        ))

    if not args.skip_single:
        for idx, feature in enumerate(args.features):
            methods.append((
                f'single_{feature}',
                SingleFeatureRanker(feature_idx=idx, ascending=True),
                None,
            ))

    methods.append((
        'composite_equal',
        CompositeDistanceRanker(weights=None),
        None,
    ))

    methods.append((
        'rrf_equal',
        RRFRanker(weights=None, rrf_k=60.0, ascending=True),
        None,
    ))

    if args.composite_calibration_json is not None:
        calibration = load_calibration(args.composite_calibration_json)
        methods.append((
            'composite_offline_frozen',
            ranker_from_calibration(calibration),
            None,
        ))

    if args.rrf_calibration_json is not None:
        calibration = load_calibration(args.rrf_calibration_json)
        methods.append((
            'rrf_offline_frozen',
            ranker_from_calibration(calibration),
            None,
        ))

    if args.include_nested_offline:
        def composite_factory(train_df: pd.DataFrame, feature_cols: list[str]):
            calibration = fit_composite_weights(
                train_df,
                feature_cols=feature_cols,
                performance_col=args.performance_col,
                target_col=args.target_col,
                source_col=args.source_col,
                dataset_col=args.dataset_col,
                n_samples=args.n_calibration_samples,
                random_state=args.random_state,
                normalizer='none',
                verbose=args.verbose,
                desc='Nested composite calibration',
            )
            return ranker_from_calibration(calibration)

        def rrf_factory(train_df: pd.DataFrame, feature_cols: list[str]):
            calibration = fit_rrf_weights(
                train_df,
                feature_cols=feature_cols,
                performance_col=args.performance_col,
                target_col=args.target_col,
                source_col=args.source_col,
                dataset_col=args.dataset_col,
                rrf_k_grid=args.rrf_k_grid,
                n_samples=args.n_calibration_samples,
                random_state=args.random_state,
                normalizer='none',
                verbose=args.verbose,
                desc='Nested RRF calibration',
            )
            return ranker_from_calibration(calibration)

        methods.append((
            'composite_offline_nested',
            CompositeDistanceRanker(weights=None),
            composite_factory,
        ))

        methods.append((
            'rrf_offline_nested',
            RRFRanker(weights=None, rrf_k=60.0, ascending=True),
            rrf_factory,
        ))

    if not args.skip_learned:
        methods.append((
            'lightgbm_lambdarank',
            LightGBMRanker(eval_at=args.k, random_state=args.random_state),
            None,
        ))

        methods.append((
            'mlp_listnet',
            MLPRanker(random_state=args.random_state),
            None,
        ))

    return methods


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

    evaluator = TransferEvaluator(
        target_col=args.target_col,
        source_col=args.source_col,
        performance_col=args.performance_col,
        dataset_col=args.dataset_col,
        k=args.k,
        top_k_relevance=args.top_k_relevance,
        val_size=args.val_size,
        random_state=args.random_state,
        verbose=args.verbose,
        include_conformal=args.include_conformal,
        conformal_alpha=args.conformal_alpha,
        conformal_cal_size=args.conformal_cal_size,
    )

    methods = build_methods(args)
    results = []

    method_iterator = tqdm(methods, desc='Methods', disable=not args.verbose)

    for method_name, ranker, factory in method_iterator:
        method_iterator.set_postfix(method=method_name)
        print(f"\nEvaluating {method_name}")

        result = evaluator.evaluate(
            ranker=ranker,
            df=df,
            feature_cols=args.features,
            method_name=method_name,
            fold_ranker_factory=factory,
        )

        print(result)
        results.append(result)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = results_to_summary(results)
    per_fold = results_to_per_fold(results)
    pairwise = pairwise_comparisons(results)

    summary_path = outdir / f'summary_{args.performance_col}.csv'
    per_fold_path = outdir / f'per_fold_{args.performance_col}.csv'
    pairwise_path = outdir / f'pairwise_{args.performance_col}.csv'

    summary.to_csv(summary_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {per_fold_path}")
    print(f"Wrote {pairwise_path}")


if __name__ == '__main__':
    main()