import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data import load_transfer_data, normalize_query_features
from src.evaluation import (
    TransferEvaluator,
    pairwise_comparisons,
    results_to_mondrian_ctc,
    results_to_per_fold,
    results_to_summary,
    results_to_transfer_type_loss,
)
from src.rankers.composite import CompositeDistanceRanker
from src.rankers.lightgbm import LightGBMRanker
from src.rankers.mlp import MLPRanker
from src.rankers.random import RandomRanker
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
        description="Compare source-language selection methods.",
    )

    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--features", nargs="+",
                        default=["new_gen", "new_typ", "new_geo", "script"])
    parser.add_argument("--performance_col", default="accuracy")
    parser.add_argument("--target_col", default="task_lang")
    parser.add_argument("--source_col", default="transfer_lang")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--normalizer", default="minmax",
                        choices=["none", "minmax"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top_k_relevance", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=0.0)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--skip_random", action="store_true")
    parser.add_argument("--skip_single", action="store_true")
    parser.add_argument("--skip_learned", action="store_true")

    parser.add_argument("--include_nested_offline", action="store_true")
    parser.add_argument("--n_opt_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_pairs_per_query", type=int, default=5000)
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument("--rrf_k_grid", nargs="+", type=float,
                        default=[1, 5, 10, 20, 40, 60, 100])

    parser.add_argument("--composite_calibration_json", default=None)
    parser.add_argument("--rrf_calibration_json", default=None)

    parser.add_argument("--include_cnotc", action="store_true")
    parser.add_argument("--cnotc_alpha", type=float, default=0.1)
    parser.add_argument("--cnotc_epsilon", type=float, default=0.05)
    parser.add_argument("--cnotc_cal_size", type=float, default=0.2)
    parser.add_argument(
        "--cnotc_grouping",
        choices=["none", "resource_level"],
        default="none",
        help=(
            "Use one global CTC quantile ('none') or Mondrian quantiles "
            "conditioned on target-language resource level ('resource_level')."
        ),
    )
    parser.add_argument(
        "--cnotc_unknown_policy",
        choices=["error", "separate"],
        default="error",
        help=(
            "For resource-level Mondrian CTC, either reject unclassified "
            "targets or treat them as a separate Mondrian group."
        ),
    )
    parser.add_argument("--budget_ks", nargs="+", type=int, default=[10])

    parser.add_argument("--include_ir_metrics", action="store_true")
    parser.add_argument("--ir_cutoffs", nargs="+", type=int,
                        default=[1, 3, 5, 10])

    parser.add_argument("--outdir", default="artifacts/evaluation")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def build_methods(args: argparse.Namespace):
    methods = []

    if not args.skip_random:
        methods.append((
            "random",
            RandomRanker(random_state=args.random_state),
        ))

    if not args.skip_single:
        for idx, feature in enumerate(args.features):
            methods.append((
                f"single_{feature}",
                SingleFeatureRanker(feature_idx=idx, ascending=True),
            ))

    methods.append((
        "composite_equal",
        CompositeDistanceRanker(weights=None, trainable=False),
    ))

    methods.append((
        "rrf_equal",
        RRFRanker(weights=None, rrf_k=60.0, ascending=True, trainable=False),
    ))

    if args.composite_calibration_json is not None:
        methods.append((
            "composite_pairwise_frozen",
            load_fitted_ranker(args.composite_calibration_json),
        ))

    if args.rrf_calibration_json is not None:
        methods.append((
            "rrf_pairwise_frozen",
            load_fitted_ranker(args.rrf_calibration_json),
        ))

    if args.include_nested_offline:
        methods.append((
            "composite_pairwise_nested",
            CompositeDistanceRanker(
                weights=None,
                trainable=True,
                n_steps=args.n_opt_steps,
                learning_rate=args.learning_rate,
                max_pairs_per_query=args.max_pairs_per_query,
                score_scale=args.score_scale,
                random_state=args.random_state,
                verbose=args.verbose,
            ),
        ))

        methods.append((
            "rrf_pairwise_nested",
            RRFRanker(
                weights=None,
                ascending=True,
                trainable=True,
                rrf_k_grid=args.rrf_k_grid,
                n_steps=args.n_opt_steps,
                learning_rate=args.learning_rate,
                max_pairs_per_query=args.max_pairs_per_query,
                score_scale=args.score_scale,
                random_state=args.random_state,
                verbose=args.verbose,
            ),
        ))

    if not args.skip_learned:
        methods.append((
            "lightgbm_lambdarank",
            LightGBMRanker(eval_at=args.k, random_state=args.random_state),
        ))

        methods.append((
            "mlp_listnet",
            MLPRanker(random_state=args.random_state),
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
        include_cnotc=args.include_cnotc,
        cnotc_alpha=args.cnotc_alpha,
        cnotc_epsilon=args.cnotc_epsilon,
        cnotc_cal_size=args.cnotc_cal_size,
        cnotc_grouping=args.cnotc_grouping,
        cnotc_unknown_policy=args.cnotc_unknown_policy,
        budget_ks=args.budget_ks,
        include_ir_metrics=args.include_ir_metrics,
        ir_cutoffs=args.ir_cutoffs,
    )

    methods = build_methods(args)
    results = []

    method_iterator = tqdm(methods, desc="Methods", disable=not args.verbose)

    for method_name, ranker in method_iterator:
        method_iterator.set_postfix(method=method_name)
        print(f"\nEvaluating {method_name}")

        result = evaluator.evaluate(
            ranker=ranker,
            df=df,
            feature_cols=args.features,
            method_name=method_name,
            fold_ranker_factory=None,
        )

        print(result)
        results.append(result)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = results_to_summary(results)
    per_fold = results_to_per_fold(results)
    pairwise = pairwise_comparisons(results)
    transfer_type_loss = results_to_transfer_type_loss(results)
    mondrian_ctc = results_to_mondrian_ctc(results)

    summary_path = outdir / f"summary_{args.performance_col}.csv"
    per_fold_path = outdir / f"per_fold_{args.performance_col}.csv"
    pairwise_path = outdir / f"pairwise_{args.performance_col}.csv"
    transfer_type_loss_path = outdir / f"transfer_type_loss_{args.performance_col}.csv"
    mondrian_ctc_path = outdir / f"mondrian_ctc_{args.performance_col}.csv"

    summary.to_csv(summary_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    transfer_type_loss.to_csv(transfer_type_loss_path, index=False)
    if not mondrian_ctc.empty:
        mondrian_ctc.to_csv(mondrian_ctc_path, index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {per_fold_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {transfer_type_loss_path}")
    if not mondrian_ctc.empty:
        print(f"Wrote {mondrian_ctc_path}")


if __name__ == "__main__":
    main()