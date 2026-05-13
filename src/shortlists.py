from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import math

import numpy as np
import pandas as pd
from tqdm import tqdm


def metric_float_tag(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p").replace("-", "m")


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    values = np.sort(np.asarray(scores, dtype=float))
    if values.size == 0:
        raise ValueError("At least one calibration score is required")

    n = values.size
    k = int(math.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return float("inf")

    return float(values[k - 1])


def find_parquet_files(paths: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_file() and path.suffix == ".parquet":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
        else:
            matches = sorted(Path().glob(str(raw_path)))
            files.extend([p for p in matches if p.is_file() and p.suffix == ".parquet"])

    unique_files = []
    seen = set()

    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    if not unique_files:
        raise FileNotFoundError(
            "No parquet files were found. Pass one or more parquet files, directories, or glob patterns."
        )

    return unique_files


def load_ranking_parquets(paths: Sequence[str | Path]) -> pd.DataFrame:
    files = find_parquet_files(paths)
    frames = []

    for path in files:
        frame = pd.read_parquet(path)
        frame["_ranking_file"] = str(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def choose_column(df: pd.DataFrame,
                  preferred: Optional[str],
                  aliases: Sequence[str],
                  *,
                  required: bool = True,
                  label: str = "column") -> Optional[str]:
    candidates = []

    if preferred is not None and preferred != "auto":
        candidates.append(preferred)

    candidates.extend(aliases)

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Could not find {label}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def prepare_rankings(df: pd.DataFrame,
                     *,
                     performance_col: str,
                     method_col: str = "auto",
                     dataset_col: str = "auto",
                     target_col: str = "auto",
                     source_col: str = "auto",
                     score_col: str = "auto",
                     rank_col: str = "auto") -> pd.DataFrame:
    method_source = choose_column(
        df,
        method_col,
        ["method", "ranker", "model", "procedure", "selector"],
        label="method column",
    )
    source_source = choose_column(
        df,
        source_col,
        ["transfer_lang", "source_lang", "source", "candidate_source", "language"],
        label="source column",
    )
    performance_source = choose_column(
        df,
        performance_col,
        [performance_col, "performance", "actual_performance", "score_true", "label"],
        label="performance column",
    )

    dataset_source = choose_column(
        df,
        dataset_col,
        ["dataset", "benchmark", "task_dataset"],
        required=False,
        label="dataset column",
    )
    target_source = choose_column(
        df,
        target_col,
        ["target_lang", "task_lang", "target", "query_target"],
        required=False,
        label="target column",
    )
    query_source = choose_column(
        df,
        "auto",
        ["query_id", "_query_id"],
        required=False,
        label="query id column",
    )
    score_source = choose_column(
        df,
        score_col,
        ["score", "pred_score", "predicted_score", "y_score", "y_pred", "rank_score", "prediction"],
        required=False,
        label="score column",
    )
    rank_source = choose_column(
        df,
        rank_col,
        ["rank", "predicted_rank", "position"],
        required=False,
        label="rank column",
    )

    out = df.copy()

    out["_method"] = out[method_source].astype(str)
    out["_source"] = out[source_source].astype(str)
    out["_performance"] = pd.to_numeric(out[performance_source], errors="coerce")

    if dataset_source is None:
        out["_dataset"] = "default"
    else:
        out["_dataset"] = out[dataset_source].astype(str)

    if target_source is not None:
        out["_target"] = out[target_source].astype(str)
    elif query_source is not None:
        out["_target"] = out[query_source].astype(str)
    else:
        raise ValueError("Could not identify a target/query column.")

    if query_source is not None:
        out["_query_id"] = out["_dataset"].astype(str) + "::" + out[query_source].astype(str)
    else:
        out["_query_id"] = out["_dataset"].astype(str) + "::" + out["_target"].astype(str)

    if score_source is not None:
        out["_score"] = pd.to_numeric(out[score_source], errors="coerce")
    elif rank_source is not None:
        out["_score"] = -pd.to_numeric(out[rank_source], errors="coerce")
    else:
        raise ValueError(
            "Could not identify a score or rank column. "
            "Expected one of score/pred_score/predicted_score/y_score/y_pred/rank_score/prediction "
            "or rank/predicted_rank/position."
        )

    if rank_source is not None:
        out["_rank"] = pd.to_numeric(out[rank_source], errors="coerce")
    else:
        out["_rank"] = (
            out.groupby(["_method", "_query_id"])["_score"]
            .rank(method="first", ascending=False)
            .astype(float)
        )

    out = out.dropna(subset=["_score", "_rank", "_performance"]).copy()
    out = out.sort_values(["_method", "_query_id", "_rank", "_source"])
    out = out.drop_duplicates(["_method", "_query_id", "_source"], keep="first")

    return out


def near_best_mask(performances: np.ndarray,
                   *,
                   rule: str,
                   value: float) -> np.ndarray:
    performances = np.asarray(performances, dtype=float)
    best = float(np.max(performances))

    if rule == "relative":
        epsilon = float(value)
        if epsilon < 0:
            raise ValueError("relative epsilon must be nonnegative")
        if best <= 0:
            return performances >= best
        return (best - performances) / best <= epsilon

    if rule == "std":
        multiplier = float(value)
        if multiplier < 0:
            raise ValueError("std multiplier must be nonnegative")
        sd = float(np.std(performances, ddof=0))
        return performances >= best - multiplier * sd

    raise ValueError(f"Unknown near-best rule: {rule}")


def selected_source_string(sources: Sequence[str]) -> str:
    return ",".join(str(x) for x in sources)


def top_k_sources(qdf: pd.DataFrame, k: int) -> list[str]:
    k = int(max(0, min(k, len(qdf))))
    if k == 0:
        return []
    return qdf.sort_values(["_rank", "_source"])["_source"].head(k).tolist()


def score_gap_sources(qdf: pd.DataFrame,
                      tau: float,
                      *,
                      cap: Optional[int]) -> list[str]:
    ordered = qdf.sort_values(["_rank", "_source"]).copy()
    scores = ordered["_score"].to_numpy(dtype=float)

    top = float(np.max(scores))
    bottom = float(np.min(scores))
    denom = top - bottom

    if denom <= 0:
        gaps = np.zeros_like(scores)
    else:
        gaps = (top - scores) / denom

    selected = ordered.loc[gaps <= tau, "_source"].tolist()

    if not selected:
        selected = ordered["_source"].head(1).tolist()

    if cap is not None and len(selected) > cap:
        selected = ordered[ordered["_source"].isin(selected)]["_source"].head(cap).tolist()

    return selected


def elbow_sources(qdf: pd.DataFrame,
                  *,
                  max_k: int) -> list[str]:
    ordered = qdf.sort_values(["_rank", "_source"]).copy()
    max_k = int(max(1, min(max_k, len(ordered))))

    if len(ordered) <= 1:
        return ordered["_source"].head(1).tolist()

    scores = ordered["_score"].to_numpy(dtype=float)
    candidate_scores = scores[:max_k]

    if candidate_scores.size <= 1:
        k = 1
    else:
        drops = candidate_scores[:-1] - candidate_scores[1:]
        k = int(np.argmax(drops) + 1)

    k = max(1, min(k, max_k))
    return ordered["_source"].head(k).tolist()


def evaluate_selection(qdf: pd.DataFrame,
                       selected_sources: Sequence[str],
                       *,
                       selector: str,
                       selector_family: str,
                       method: str,
                       relative_epsilons: Sequence[float],
                       std_multipliers: Sequence[float],
                       extra: Optional[dict] = None) -> dict:
    selected_set = set(selected_sources)
    performances = qdf["_performance"].to_numpy(dtype=float)
    sources = qdf["_source"].astype(str).to_numpy()

    selected_mask = np.array([source in selected_set for source in sources], dtype=bool)

    best_perf = float(np.max(performances))
    best_sources = sources[performances == best_perf].tolist()

    if selected_mask.any():
        best_in_set = float(np.max(performances[selected_mask]))
        if best_perf <= 0:
            best_in_set_loss = float("nan")
        else:
            best_in_set_loss = float((best_perf - best_in_set) / best_perf)
    else:
        best_in_set = float("nan")
        best_in_set_loss = float("nan")

    record = {
        "method": method,
        "selector_family": selector_family,
        "selector": selector,
        "dataset": qdf["_dataset"].iloc[0],
        "target_lang": qdf["_target"].iloc[0],
        "query_id": qdf["_query_id"].iloc[0],
        "n_candidates": int(len(qdf)),
        "set_size": int(len(selected_set)),
        "set_size_fraction": float(len(selected_set) / len(qdf)) if len(qdf) else float("nan"),
        "empty": float(len(selected_set) == 0),
        "selected_sources": selected_source_string(selected_sources),
        "best_sources": selected_source_string(best_sources),
        "best_performance": best_perf,
        "best_in_set_performance": best_in_set,
        "best_in_set_performance_loss": best_in_set_loss,
        "exact_best_coverage": float(bool(selected_set.intersection(best_sources))),
    }

    for epsilon in relative_epsilons:
        tag = metric_float_tag(epsilon)
        good = near_best_mask(performances, rule="relative", value=epsilon)
        good_sources = set(sources[good].tolist())
        record[f"relative_{tag}_coverage"] = float(bool(selected_set.intersection(good_sources)))

    for multiplier in std_multipliers:
        tag = metric_float_tag(multiplier)
        good = near_best_mask(performances, rule="std", value=multiplier)
        good_sources = set(sources[good].tolist())
        record[f"std_{tag}_coverage"] = float(bool(selected_set.intersection(good_sources)))

    if extra:
        record.update(extra)

    return record


def first_near_best_rank(qdf: pd.DataFrame,
                         *,
                         rule: str,
                         value: float) -> float:
    ordered = qdf.sort_values(["_rank", "_source"]).copy()
    performances = ordered["_performance"].to_numpy(dtype=float)
    good = near_best_mask(performances, rule=rule, value=value)

    if not np.any(good):
        return float(len(ordered))

    positions = np.arange(1, len(ordered) + 1)
    return float(np.min(positions[good]))


def normalized_gap_grid(groups: Sequence[pd.DataFrame],
                        *,
                        max_grid_size: int = 101) -> np.ndarray:
    values = [0.0, 1.0]

    for qdf in groups:
        ordered = qdf.sort_values(["_rank", "_source"])
        scores = ordered["_score"].to_numpy(dtype=float)
        top = float(np.max(scores))
        bottom = float(np.min(scores))
        denom = top - bottom

        if denom <= 0:
            values.append(0.0)
        else:
            gaps = (top - scores) / denom
            values.extend(gaps.tolist())

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.clip(values, 0.0, 1.0)

    if values.size > max_grid_size:
        probs = np.linspace(0.0, 1.0, max_grid_size)
        values = np.quantile(values, probs)

    return np.unique(values)


def tune_score_gap_tau(groups: Sequence[pd.DataFrame],
                       *,
                       rule: str,
                       value: float,
                       cap: int,
                       tolerance: float = 0.01) -> tuple[float, float, float]:
    tau_grid = normalized_gap_grid(groups)
    rows = []

    prepared = []
    for qdf in groups:
        ordered = qdf.sort_values(["_rank", "_source"]).copy()
        scores = ordered["_score"].to_numpy(dtype=float)
        performances = ordered["_performance"].to_numpy(dtype=float)
        sources = ordered["_source"].astype(str).to_numpy()

        top = float(np.max(scores))
        bottom = float(np.min(scores))
        denom = top - bottom

        if denom <= 0:
            gaps = np.zeros_like(scores)
        else:
            gaps = (top - scores) / denom

        good = near_best_mask(performances, rule=rule, value=value)
        good_sources = set(sources[good].tolist())

        prepared.append((ordered, gaps, good_sources))

    for tau in tau_grid:
        coverages = []
        sizes = []

        for ordered, gaps, good_sources in prepared:
            selected = ordered.loc[gaps <= tau, "_source"].tolist()

            if not selected:
                selected = ordered["_source"].head(1).tolist()

            if len(selected) > cap:
                selected = selected[:cap]

            selected_set = set(selected)
            coverages.append(float(bool(selected_set.intersection(good_sources))))
            sizes.append(len(selected_set))

        rows.append({
            "tau": float(tau),
            "coverage": float(np.mean(coverages)),
            "average_size": float(np.mean(sizes)),
        })

    table = pd.DataFrame(rows)
    best_coverage = float(table["coverage"].max())
    eligible = table[table["coverage"] >= best_coverage - tolerance].copy()
    eligible = eligible.sort_values(["average_size", "tau"], ascending=[True, True])

    chosen = eligible.iloc[0]
    return float(chosen["tau"]), float(chosen["coverage"]), float(chosen["average_size"])


def pareto_front_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[0]

    keep = np.ones(n, dtype=bool)

    for i in range(n):
        dominated_by_someone = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if np.any(dominated_by_someone):
            keep[i] = False

    return keep


def pareto_sources(qdf: pd.DataFrame,
                   *,
                   distance_features: Sequence[str],
                   cap: int) -> list[str]:
    available = [col for col in distance_features if col in qdf.columns]
    if not available:
        return []

    base = qdf.drop_duplicates("_source").copy()
    X = base[available].to_numpy(dtype=float)

    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    X_norm = (X - mins) / denom

    front = pareto_front_mask(X_norm)
    front_df = base.loc[front].copy()
    front_df["_mean_distance"] = X_norm[front].mean(axis=1)

    front_df = front_df.sort_values(["_mean_distance", "_source"])
    return front_df["_source"].head(cap).tolist()


@dataclass
class ShortlistConfig:
    fixed_top_k: tuple[int, ...] = (3, 5, 10)
    relative_epsilons: tuple[float, ...] = (0.05,)
    std_multipliers: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    conformal_alphas: tuple[float, ...] = (0.1,)
    conformal_caps: tuple[int, ...] = (3, 5, 10)
    calibrated_targets: tuple[float, ...] = (0.75, 0.8, 0.9)
    calibrated_max_k: tuple[int, ...] = (3, 5, 10)
    score_gap_budgets: tuple[int, ...] = (3, 5, 10)
    score_gap_tolerance: float = 0.01
    elbow_max_k: tuple[int, ...] = (5, 10)
    consensus_top_k0: tuple[int, ...] = (5, 10)
    consensus_vote_thresholds: tuple[int, ...] = (2, 3, 4)
    consensus_caps: tuple[int, ...] = (3, 5, 10)
    consensus_exclude_methods: tuple[str, ...] = ("random",)
    pareto_caps: tuple[int, ...] = (3, 5, 10)
    distance_features: tuple[str, ...] = ("new_gen", "new_typ", "new_geo", "script")
    include_fixed_topk: bool = True
    include_conformal: bool = True
    include_calibrated_min_k: bool = True
    include_score_gap: bool = True
    include_elbow: bool = True
    include_consensus: bool = True
    include_pareto: bool = True


class PostHocShortlistEvaluator:
    def __init__(self,
                 rankings: pd.DataFrame,
                 config: ShortlistConfig,
                 *,
                 verbose: bool = False):
        self.rankings = rankings.copy()
        self.config = config
        self.verbose = verbose

    def near_best_configs(self) -> list[tuple[str, float, str]]:
        configs = []

        for epsilon in self.config.relative_epsilons:
            tag = f"relative_{metric_float_tag(epsilon)}"
            configs.append(("relative", float(epsilon), tag))

        for multiplier in self.config.std_multipliers:
            tag = f"std_{metric_float_tag(multiplier)}"
            configs.append(("std", float(multiplier), tag))

        return configs

    def method_query_groups(self, method_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        groups = {}

        for query_id, qdf in method_df.groupby("_query_id", sort=False):
            groups[str(query_id)] = qdf.sort_values(["_rank", "_source"]).copy()

        return groups

    def first_rank_cache(self,
                         groups: dict[str, pd.DataFrame],
                         near_configs: list[tuple[str, float, str]]) -> dict[tuple[str, str], float]:
        cache: dict[tuple[str, str], float] = {}

        for rule, value, near_tag in near_configs:
            for query_id, qdf in groups.items():
                cache[(near_tag, query_id)] = first_near_best_rank(
                    qdf,
                    rule=rule,
                    value=value,
                )

        return cache

    def evaluate_fixed_topk(self) -> list[dict]:
        records = []

        for method, method_df in tqdm(
            self.rankings.groupby("_method", sort=False),
            desc="fixed top-K",
            disable=not self.verbose,
        ):
            for _, qdf in method_df.groupby("_query_id", sort=False):
                qdf = qdf.sort_values(["_rank", "_source"]).copy()

                for k in self.config.fixed_top_k:
                    selected = top_k_sources(qdf, k)
                    selector = f"fixed_top_{k}"

                    records.append(
                        evaluate_selection(
                            qdf,
                            selected,
                            selector=selector,
                            selector_family="fixed_top_k",
                            method=method,
                            relative_epsilons=self.config.relative_epsilons,
                            std_multipliers=self.config.std_multipliers,
                            extra={"chosen_k": int(k)},
                        )
                    )

        return records

    def evaluate_conformal(self) -> list[dict]:
        records = []
        near_configs = self.near_best_configs()
        caps: list[Optional[int]] = [None] + [int(x) for x in self.config.conformal_caps]

        for method, method_df in tqdm(
            self.rankings.groupby("_method", sort=False),
            desc="conformal rank top-K",
            disable=not self.verbose,
        ):
            groups = self.method_query_groups(method_df)
            query_ids = list(groups.keys())

            if len(query_ids) < 3:
                continue

            rank_cache = self.first_rank_cache(groups, near_configs)

            for rule, value, near_tag in near_configs:
                all_ranks = np.asarray(
                    [rank_cache[(near_tag, query_id)] for query_id in query_ids],
                    dtype=float,
                )

                for heldout_pos, heldout_query in enumerate(query_ids):
                    test_qdf = groups[heldout_query]
                    calibration_ranks = np.delete(all_ranks, heldout_pos)

                    for alpha in self.config.conformal_alphas:
                        threshold = conformal_quantile(calibration_ranks, alpha=alpha)

                        if np.isinf(threshold):
                            uncapped_k = len(test_qdf)
                        else:
                            uncapped_k = int(math.ceil(threshold))

                        uncapped_k = max(1, min(uncapped_k, len(test_qdf)))

                        for cap in caps:
                            if cap is None:
                                k = uncapped_k
                                cap_tag = "uncapped"
                                capped = 0.0
                            else:
                                k = min(uncapped_k, int(cap))
                                cap_tag = f"cap_{int(cap)}"
                                capped = float(uncapped_k > int(cap))

                            selected = top_k_sources(test_qdf, k)
                            selector = (
                                f"conformal_{near_tag}_alpha_{metric_float_tag(alpha)}_{cap_tag}"
                            )

                            records.append(
                                evaluate_selection(
                                    test_qdf,
                                    selected,
                                    selector=selector,
                                    selector_family="conformal_rank_top_k",
                                    method=method,
                                    relative_epsilons=self.config.relative_epsilons,
                                    std_multipliers=self.config.std_multipliers,
                                    extra={
                                        "conformal_rule": rule,
                                        "conformal_value": float(value),
                                        "conformal_alpha": float(alpha),
                                        "conformal_threshold": threshold,
                                        "uncapped_k": int(uncapped_k),
                                        "chosen_k": int(k),
                                        "cap": "" if cap is None else int(cap),
                                        "capped": capped,
                                    },
                                )
                            )

        return records

    def evaluate_calibrated_min_k(self) -> list[dict]:
        records = []
        near_configs = self.near_best_configs()

        for method, method_df in tqdm(
            self.rankings.groupby("_method", sort=False),
            desc="calibrated smallest K",
            disable=not self.verbose,
        ):
            groups = self.method_query_groups(method_df)
            query_ids = list(groups.keys())

            if len(query_ids) < 3:
                continue

            rank_cache = self.first_rank_cache(groups, near_configs)

            for rule, value, near_tag in near_configs:
                all_ranks = np.asarray(
                    [rank_cache[(near_tag, query_id)] for query_id in query_ids],
                    dtype=float,
                )

                for heldout_pos, heldout_query in enumerate(query_ids):
                    test_qdf = groups[heldout_query]
                    calibration_ranks = np.delete(all_ranks, heldout_pos)

                    for target in self.config.calibrated_targets:
                        for max_k in self.config.calibrated_max_k:
                            max_k = int(max_k)

                            chosen_k = max_k
                            calibration_coverage = float(np.mean(calibration_ranks <= max_k))

                            for candidate_k in range(1, max_k + 1):
                                candidate_coverage = float(
                                    np.mean(calibration_ranks <= candidate_k)
                                )
                                if candidate_coverage >= float(target):
                                    chosen_k = candidate_k
                                    calibration_coverage = candidate_coverage
                                    break

                            selected = top_k_sources(test_qdf, chosen_k)
                            selector = (
                                f"calibrated_min_k_{near_tag}_target_"
                                f"{metric_float_tag(target)}_max_{max_k}"
                            )

                            records.append(
                                evaluate_selection(
                                    test_qdf,
                                    selected,
                                    selector=selector,
                                    selector_family="calibrated_min_k",
                                    method=method,
                                    relative_epsilons=self.config.relative_epsilons,
                                    std_multipliers=self.config.std_multipliers,
                                    extra={
                                        "calibration_rule": rule,
                                        "calibration_value": float(value),
                                        "target_coverage": float(target),
                                        "max_k": int(max_k),
                                        "chosen_k": int(chosen_k),
                                        "calibration_coverage": float(calibration_coverage),
                                    },
                                )
                            )

        return records

    def evaluate_score_gap(self) -> list[dict]:
        records = []
        near_configs = self.near_best_configs()

        for method, method_df in tqdm(
            self.rankings.groupby("_method", sort=False),
            desc="score-gap sets",
            disable=not self.verbose,
        ):
            groups = self.method_query_groups(method_df)
            query_ids = list(groups.keys())

            if len(query_ids) < 3:
                continue

            for heldout_query in query_ids:
                test_qdf = groups[heldout_query]
                calibration_groups = [
                    groups[qid] for qid in query_ids if qid != heldout_query
                ]

                for rule, value, near_tag in near_configs:
                    for budget in self.config.score_gap_budgets:
                        tau, cal_cov, cal_size = tune_score_gap_tau(
                            calibration_groups,
                            rule=rule,
                            value=value,
                            cap=int(budget),
                            tolerance=self.config.score_gap_tolerance,
                        )

                        selected = score_gap_sources(test_qdf, tau=tau, cap=int(budget))
                        selector = f"score_gap_{near_tag}_budget_{int(budget)}"

                        records.append(
                            evaluate_selection(
                                test_qdf,
                                selected,
                                selector=selector,
                                selector_family="score_gap",
                                method=method,
                                relative_epsilons=self.config.relative_epsilons,
                                std_multipliers=self.config.std_multipliers,
                                extra={
                                    "score_gap_rule": rule,
                                    "score_gap_value": float(value),
                                    "budget": int(budget),
                                    "chosen_tau": float(tau),
                                    "calibration_coverage": cal_cov,
                                    "calibration_average_size": cal_size,
                                },
                            )
                        )

        return records

    def evaluate_elbow(self) -> list[dict]:
        records = []

        for method, method_df in tqdm(
            self.rankings.groupby("_method", sort=False),
            desc="elbow sets",
            disable=not self.verbose,
        ):
            for _, qdf in method_df.groupby("_query_id", sort=False):
                qdf = qdf.sort_values(["_rank", "_source"]).copy()

                for max_k in self.config.elbow_max_k:
                    selected = elbow_sources(qdf, max_k=int(max_k))
                    selector = f"elbow_max_{int(max_k)}"

                    records.append(
                        evaluate_selection(
                            qdf,
                            selected,
                            selector=selector,
                            selector_family="elbow",
                            method=method,
                            relative_epsilons=self.config.relative_epsilons,
                            std_multipliers=self.config.std_multipliers,
                            extra={"max_k": int(max_k), "chosen_k": int(len(selected))},
                        )
                    )

        return records

    def evaluate_consensus(self) -> list[dict]:
        records = []
        excluded = set(self.config.consensus_exclude_methods)
        work = self.rankings[~self.rankings["_method"].isin(excluded)].copy()

        if work["_method"].nunique() < 2:
            return records

        for query_id, qdf_all in tqdm(
            work.groupby("_query_id", sort=False),
            desc="consensus sets",
            disable=not self.verbose,
        ):
            base = (
                qdf_all.sort_values(["_method", "_rank", "_source"])
                .drop_duplicates(["_source"], keep="first")
                .copy()
            )

            for k0 in self.config.consensus_top_k0:
                top = qdf_all[qdf_all["_rank"] <= int(k0)].copy()
                votes = (
                    top.groupby("_source")
                    .agg(
                        vote_count=("_method", "nunique"),
                        mean_rank=("_rank", "mean"),
                        mean_score=("_score", "mean"),
                    )
                    .reset_index()
                )

                all_sources = (
                    qdf_all.groupby("_source")
                    .agg(mean_rank=("_rank", "mean"), mean_score=("_score", "mean"))
                    .reset_index()
                )

                votes = all_sources.merge(votes, on="_source", how="left", suffixes=("", "_vote"))
                votes["vote_count"] = votes["vote_count"].fillna(0.0)
                votes["mean_rank"] = votes["mean_rank_vote"].fillna(votes["mean_rank"])
                votes["mean_score"] = votes["mean_score_vote"].fillna(votes["mean_score"])

                for vote_threshold in self.config.consensus_vote_thresholds:
                    eligible = votes[votes["vote_count"] >= int(vote_threshold)].copy()

                    if eligible.empty:
                        eligible = votes.copy()

                    eligible = eligible.sort_values(
                        ["vote_count", "mean_rank", "mean_score", "_source"],
                        ascending=[False, True, False, True],
                    )

                    for cap in self.config.consensus_caps:
                        selected = eligible["_source"].head(int(cap)).tolist()

                        selector = (
                            f"consensus_top_{int(k0)}_vote_{int(vote_threshold)}_cap_{int(cap)}"
                        )

                        records.append(
                            evaluate_selection(
                                base,
                                selected,
                                selector=selector,
                                selector_family="consensus",
                                method="consensus",
                                relative_epsilons=self.config.relative_epsilons,
                                std_multipliers=self.config.std_multipliers,
                                extra={
                                    "consensus_top_k0": int(k0),
                                    "vote_threshold": int(vote_threshold),
                                    "cap": int(cap),
                                    "n_rankers": int(work["_method"].nunique()),
                                },
                            )
                        )

        return records

    def evaluate_pareto(self) -> list[dict]:
        records = []
        available = [col for col in self.config.distance_features if col in self.rankings.columns]

        if not available:
            return records

        base = (
            self.rankings.sort_values(["_query_id", "_source"])
            .drop_duplicates(["_query_id", "_source"], keep="first")
            .copy()
        )

        for _, qdf in tqdm(
            base.groupby("_query_id", sort=False),
            desc="pareto sets",
            disable=not self.verbose,
        ):
            for cap in self.config.pareto_caps:
                selected = pareto_sources(
                    qdf,
                    distance_features=available,
                    cap=int(cap),
                )
                selector = f"pareto_front_cap_{int(cap)}"

                records.append(
                    evaluate_selection(
                        qdf,
                        selected,
                        selector=selector,
                        selector_family="pareto",
                        method="pareto",
                        relative_epsilons=self.config.relative_epsilons,
                        std_multipliers=self.config.std_multipliers,
                        extra={
                            "cap": int(cap),
                            "distance_features": ",".join(available),
                        },
                    )
                )

        return records

    def run(self) -> pd.DataFrame:
        records = []

        if self.config.include_fixed_topk:
            records.extend(self.evaluate_fixed_topk())

        if self.config.include_conformal:
            records.extend(self.evaluate_conformal())

        if self.config.include_calibrated_min_k:
            records.extend(self.evaluate_calibrated_min_k())

        if self.config.include_score_gap:
            records.extend(self.evaluate_score_gap())

        if self.config.include_elbow:
            records.extend(self.evaluate_elbow())

        if self.config.include_consensus:
            records.extend(self.evaluate_consensus())

        if self.config.include_pareto:
            records.extend(self.evaluate_pareto())

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)


def summarize_shortlists(per_query: pd.DataFrame) -> pd.DataFrame:
    if per_query.empty:
        return pd.DataFrame()

    group_cols = ["method", "selector_family", "selector"]
    rows = []

    metric_cols = [
        col for col in per_query.columns
        if (
            col.endswith("_coverage")
            or col.endswith("_performance_loss")
            or col in {
                "set_size",
                "set_size_fraction",
                "empty",
                "chosen_k",
                "uncapped_k",
                "capped",
                "calibration_coverage",
                "calibration_average_size",
            }
        )
    ]

    for keys, g in per_query.groupby(group_cols, sort=False):
        method, selector_family, selector = keys

        row = {
            "method": method,
            "selector_family": selector_family,
            "selector": selector,
            "n_queries": int(g["query_id"].nunique()),
            "average_set_size": float(g["set_size"].mean()),
            "median_set_size": float(g["set_size"].median()),
            "p90_set_size": float(g["set_size"].quantile(0.9)),
            "empty_rate": float(g["empty"].mean() * 100),
        }

        for col in metric_cols:
            if col in {"set_size", "empty"}:
                continue

            values = pd.to_numeric(g[col], errors="coerce")
            if values.notna().sum() == 0:
                continue

            if col.endswith("_coverage"):
                row[col] = float(values.mean() * 100)
            elif col.endswith("_performance_loss"):
                row[col] = float(values.mean() * 100)
            elif col.endswith("_fraction"):
                row[col] = float(values.mean() * 100)
            elif col.endswith("_rate"):
                row[col] = float(values.mean() * 100)
            else:
                row[col] = float(values.mean())

        rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = []
    ascending = []

    if "relative_0p05_coverage" in out.columns:
        sort_cols.append("relative_0p05_coverage")
        ascending.append(False)

    if "std_0p5_coverage" in out.columns:
        sort_cols.append("std_0p5_coverage")
        ascending.append(False)

    sort_cols.append("average_set_size")
    ascending.append(True)

    return out.sort_values(sort_cols, ascending=ascending)