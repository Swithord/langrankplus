import argparse
from pathlib import Path

from src.data import load_transfer_data
from src.offline.calibration import (
    fit_composite_weights,
    fit_rrf_weights,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit offline task-agnostic weights for training-free rankers.",
    )
    parser.add_argument('--csv', nargs='+', required=True,
                        help="One or more transfer-performance CSV files.")
    parser.add_argument('--dataset_names', nargs='*', default=None,
                        help="Optional dataset names matching --csv.")
    parser.add_argument('--features', nargs='+',
                        default=['new_gen', 'new_typ', 'new_geo', 'script'])
    parser.add_argument('--performance_col', default='accuracy')
    parser.add_argument('--target_col', default='task_lang')
    parser.add_argument('--source_col', default='transfer_lang')
    parser.add_argument('--dataset_col', default='dataset')
    parser.add_argument('--methods', nargs='+',
                        default=['composite', 'rrf'],
                        choices=['composite', 'rrf'])
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--rrf_k_grid', nargs='+', type=float,
                        default=[1, 5, 10, 20, 40, 60, 100])
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--outdir', default='artifacts/offline_calibration')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_names = args.dataset_names
    if dataset_names == []:
        dataset_names = None

    df = load_transfer_data(
        args.csv,
        dataset_names=dataset_names,
        dataset_col=args.dataset_col,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if 'composite' in args.methods:
        result = fit_composite_weights(
            df,
            feature_cols=args.features,
            performance_col=args.performance_col,
            target_col=args.target_col,
            source_col=args.source_col,
            dataset_col=args.dataset_col,
            n_samples=args.n_samples,
            random_state=args.random_state,
            normalizer='minmax',
            verbose=args.verbose,
        )
        path = outdir / f'composite_{args.performance_col}.json'
        save_calibration(result, str(path))
        print(f"Wrote {path}")
        print(result)

    if 'rrf' in args.methods:
        result = fit_rrf_weights(
            df,
            feature_cols=args.features,
            performance_col=args.performance_col,
            target_col=args.target_col,
            source_col=args.source_col,
            dataset_col=args.dataset_col,
            rrf_k_grid=args.rrf_k_grid,
            n_samples=args.n_samples,
            random_state=args.random_state,
            normalizer='none',
            verbose=args.verbose,
        )
        path = outdir / f'rrf_{args.performance_col}.json'
        save_calibration(result, str(path))
        print(f"Wrote {path}")
        print(result)


if __name__ == '__main__':
    main()