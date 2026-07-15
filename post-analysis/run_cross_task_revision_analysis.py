#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from cross_task_analysis_utils import run_cross_task_revision_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reviewer-response tables for leave-one-task-out "
            "cross-task generalisation results, with separate with-English "
            "and without-English variants."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing artifacts/, data/, and post-analysis/.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Cross-task artifact directory. Defaults to "
            "<root>/artifacts/cross_task_generalization."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("post-analysis/cross-task-outputs"),
        help="Output directory for combined data and CSV tables.",
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
        help="Base random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else (args.root / "artifacts" / "cross_task_generalization").resolve()
    )

    print("Starting cross-task generalisation post-analysis.", flush=True)
    print(f"Repository root: {args.root.resolve()}", flush=True)
    print(f"Artifact root: {artifact_root}", flush=True)
    print(f"Output directory: {args.outdir.resolve()}", flush=True)
    print(f"Bootstrap replicates: {args.n_bootstrap}", flush=True)

    run_cross_task_revision_analysis(
        root=args.root,
        artifact_root=args.artifact_root,
        outdir=args.outdir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print("Cross-task generalisation post-analysis completed.", flush=True)


if __name__ == "__main__":
    main()