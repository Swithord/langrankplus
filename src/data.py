from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd


def drop_unnamed_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop index columns
    """
    keep = [c for c in df.columns if not str(c).startswith('Unnamed:')]
    return df.loc[:, keep].copy()


def load_transfer_data(csv_paths: Sequence[str],
                       dataset_names: Optional[Sequence[str]] = None,
                       dataset_col: str = 'dataset') -> pd.DataFrame:
    """
    Load one or more transfer-performance CSVs and add a dataset column.

    Each CSV is assumed to contain rows of the form
        (task_lang, transfer_lang, performance, features...).

    :param csv_paths: paths to CSV files
    :param dataset_names: optional names to assign to each CSV. If None, file stems are used.
    :param dataset_col: column to identify the dataset/task collection
    :return: concatenated dataframe
    """
    if dataset_names is not None and len(dataset_names) != len(csv_paths):
        raise ValueError("dataset_names must have the same length as csv_paths")

    frames = []
    for idx, path in enumerate(csv_paths):
        path_obj = Path(path)
        name = dataset_names[idx] if dataset_names is not None else path_obj.stem
        df = pd.read_csv(path_obj)
        df = drop_unnamed_index_columns(df)
        df[dataset_col] = name
        frames.append(df)

    if not frames:
        raise ValueError("At least one CSV path is required")

    return pd.concat(frames, ignore_index=True)


def get_query_cols(df: pd.DataFrame,
                   target_col: str = 'task_lang',
                   dataset_col: Optional[str] = 'dataset') -> list[str]:
    """
    Return the columns defining one ranking query.

    For a single dataset with no dataset column, queries are target languages.
    For multiple datasets, queries are (dataset, target language) pairs.
    """
    if dataset_col is not None and dataset_col in df.columns:
        return [dataset_col, target_col]
    return [target_col]


def add_query_id(df: pd.DataFrame,
                 target_col: str = 'task_lang',
                 dataset_col: Optional[str] = 'dataset',
                 query_id_col: str = '_query_id') -> pd.DataFrame:
    """
    Add a string-valued query id column.
    """
    df = df.copy()
    query_cols = get_query_cols(df, target_col=target_col, dataset_col=dataset_col)
    df[query_id_col] = df[query_cols].astype(str).agg('::'.join, axis=1)
    return df


def normalize_query_features(df: pd.DataFrame,
                             feature_cols: list[str],
                             target_col: str = 'task_lang',
                             dataset_col: Optional[str] = 'dataset',
                             method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize feature columns within each query. This is appropriate for online
    training-free source selection because each ranking decision is made within a
    fixed target-language candidate set.

    :param df: input dataframe
    :param feature_cols: feature columns to normalize
    :param target_col: target language column
    :param dataset_col: optional dataset column
    :param method: currently supports 'none' or 'minmax'
    :return: dataframe with normalized feature columns
    """
    if method == 'none':
        return df.copy()
    if method != 'minmax':
        raise ValueError(f"Unknown normalization method: {method}")

    out = df.copy()
    query_cols = get_query_cols(out, target_col=target_col, dataset_col=dataset_col)

    for _, idx in out.groupby(query_cols, sort=False).groups.items():
        block = out.loc[idx, feature_cols].astype(float)
        mins = block.min(axis=0)
        maxs = block.max(axis=0)
        denom = (maxs - mins).replace(0.0, np.nan)
        normalized = (block - mins) / denom
        normalized = normalized.fillna(0.0)
        out.loc[idx, feature_cols] = normalized.to_numpy()

    return out