#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_utils import run_revision_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LangRankPlus reviewer-response post-analysis."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing artifacts/ and data/.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("post-analysis/outputs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=20000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--include-nnrank",
        action="store_true",
        help="Generate NNRank-restricted tables.",
    )
    parser.add_argument(
        "--min-ctc-targets",
        type=int,
        default=50,
        help=(
            "Minimum held-out targets in a dataset-resource cell for the "
            "task-balanced Mondrian CTC summary."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting LangRankPlus post-analysis.", flush=True)
    print(f"Repository root: {args.root.expanduser().resolve()}", flush=True)
    print(f"Output directory: {args.outdir.expanduser().resolve()}", flush=True)
    print(f"Bootstrap replicates: {args.n_bootstrap}", flush=True)
    print(f"Include NNRank: {args.include_nnrank}", flush=True)

    run_revision_analysis(
        root=args.root,
        outdir=args.outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        include_nnrank=args.include_nnrank,
        min_ctc_targets=args.min_ctc_targets,
    )


if __name__ == "__main__":
    main()