import pandas as pd


def validate_dataset(df: pd.DataFrame,
                     feature_cols: list[str],
                     target_col: str = 'task_lang',
                     source_col: str = 'transfer_lang',
                     performance_col: str = 'performance') -> None:
    """
    Validate that the dataset is well-formed for transfer language ranking.
    We assume the following about datasets:
    - Each row corresponds to a (target language, source language) pair.
    - Each target language has at least 2 source languages (so we can rank them).
    - No missing values.
    - With m target languages and n source languages, there are exactly m*n rows (no duplicates, no missing pairs).

    :param df: the dataset
    :param feature_cols: list of feature column names
    :param target_col: name of the target language column
    :param source_col: name of the source language column
    :param performance_col: name of the performance column
    :raises ValueError: if validation fails
    """
    required_cols = [target_col, source_col, performance_col] + list(feature_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    nan_features = df[feature_cols].isna().any()
    if nan_features.any():
        bad = nan_features[nan_features].index.tolist()
        raise ValueError(f"NaN values found in feature columns: {bad}")

    if df[performance_col].isna().any():
        n_nan = df[performance_col].isna().sum()
        raise ValueError(f"NaN values found in performance column '{performance_col}' "
                         f"({n_nan} rows)")

    counts = df.groupby(target_col).size()
    too_few = counts[counts < 2]
    if not too_few.empty:
        raise ValueError(f"Target languages with fewer than 2 sources "
                         f"(cannot rank): {too_few.index.tolist()}")

    duplicates = df.duplicated(subset=[target_col, source_col])
    if duplicates.any():
        n_dup = duplicates.sum()
        raise ValueError(f"Found {n_dup} duplicate (target, source) pairs")

    n_targets = df[target_col].nunique()
    n_sources = df[source_col].nunique()
    expected_rows = n_targets * n_sources
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows for {n_targets} targets and "
                         f"{n_sources} sources, but found {len(df)}")
