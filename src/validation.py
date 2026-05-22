import pandas as pd


def validate_dataset(df: pd.DataFrame,
                     feature_cols: list[str],
                     target_col: str = 'task_lang',
                     source_col: str = 'transfer_lang',
                     performance_col: str = 'performance',
                     dataset_col: str | None = None,
                     require_performance: bool = True,
                     require_complete_matrix: bool = False) -> None:
    """
    Validate that the dataset is well-formed for transfer language ranking.

    We assume:
    - Each row corresponds to a (dataset, target language, source language) pair,
      or just a (target language, source language) pair if dataset_col is absent.
    - Each query has at least 2 source languages.
    - Feature columns have no missing values.
    - There are no duplicate source candidates within a query.

    :param df: dataset
    :param feature_cols: list of feature column names
    :param target_col: target language column
    :param source_col: source language column
    :param performance_col: performance column
    :param dataset_col: optional dataset column
    :param require_performance: whether performance_col must exist and be non-missing
    :param require_complete_matrix: whether to enforce a complete target-source matrix
    """
    query_cols = [target_col]
    if dataset_col is not None and dataset_col in df.columns:
        query_cols = [dataset_col, target_col]

    required_cols = query_cols + [source_col] + list(feature_cols)
    if require_performance:
        required_cols.append(performance_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    nan_features = df[feature_cols].isna().any()
    if nan_features.any():
        bad = nan_features[nan_features].index.tolist()
        raise ValueError(f"NaN values found in feature columns: {bad}")

    if require_performance and df[performance_col].isna().any():
        n_nan = df[performance_col].isna().sum()
        raise ValueError(f"NaN values found in performance column '{performance_col}' "
                         f"({n_nan} rows)")

    counts = df.groupby(query_cols).size()
    too_few = counts[counts < 2]
    if not too_few.empty:
        raise ValueError("Some ranking queries have fewer than 2 source languages")

    duplicate_cols = query_cols + [source_col]
    duplicates = df.duplicated(subset=duplicate_cols)
    if duplicates.any():
        n_dup = int(duplicates.sum())
        raise ValueError(f"Found {n_dup} duplicate query-source pairs")

    if require_complete_matrix:
        if dataset_col is not None and dataset_col in df.columns:
            for dataset_name, ddf in df.groupby(dataset_col, sort=False):
                n_targets = ddf[target_col].nunique()
                n_sources = ddf[source_col].nunique()
                expected_rows = n_targets * n_sources
                if len(ddf) != expected_rows:
                    raise ValueError(f"Dataset '{dataset_name}' expected {expected_rows} "
                                     f"rows for {n_targets} targets and {n_sources} "
                                     f"sources, but found {len(ddf)}")
        else:
            n_targets = df[target_col].nunique()
            n_sources = df[source_col].nunique()
            expected_rows = n_targets * n_sources
            if len(df) != expected_rows:
                raise ValueError(f"Expected {expected_rows} rows for {n_targets} targets "
                                 f"and {n_sources} sources, but found {len(df)}")