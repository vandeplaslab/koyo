"""Statistics functions."""

from __future__ import annotations

import sys
import typing as ty

import numpy as np
import scipy.stats as ss

if ty.TYPE_CHECKING:
    import pandas as pd


def _is_dataframe(obj: ty.Any) -> bool:
    """Check whether `obj` is a pandas DataFrame without importing pandas (pandas is an optional dependency)."""
    pd_ = sys.modules.get("pandas")
    return pd_ is not None and isinstance(obj, pd_.DataFrame)


def rank_features(df: pd.DataFrame) -> np.ndarray:
    """Rank features (each column independently, ties get the average rank)."""
    return ss.rankdata(np.asarray(df), axis=0)


def pairwise_spearmanr(
    array_one: np.ndarray | pd.DataFrame,
    array_two: np.ndarray | pd.DataFrame | None = None,
    ranked: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Compute pairwise Spearman rank correlation between columns of two arrays."""
    if array_two is None:
        array_two = array_one

    columns_one = None
    if _is_dataframe(array_one):
        columns_one = array_one.columns
        array_one = array_one.values
    columns_two = None
    if _is_dataframe(array_two):
        columns_two = array_two.columns
        array_two = array_two.values
    if not ranked:
        array_one = ss.rankdata(array_one, axis=0)
        array_two = ss.rankdata(array_two, axis=0)
    am = array_one - np.mean(array_one, axis=0, keepdims=True)
    bm = array_two - np.mean(array_two, axis=0, keepdims=True)

    correlation_matrix = (
        am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))
    )
    if columns_one is not None and columns_two is not None:
        import pandas as pd

        correlation_matrix = pd.DataFrame(correlation_matrix, index=columns_one, columns=columns_two)
    return correlation_matrix


def pairwise_pearsonr(array_one: np.ndarray, array_two: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
    """Compute pairwise Pearson correlation between columns of two arrays.

    Parameters
    ----------
    - array_one: numpy array of shape (N, M1)
    - array_two: numpy array of shape (N, M2)

    Returns
    -------
    - correlation_matrix: numpy array of shape (M1, M2) with Pearson correlation coefficients
    """

    def _get_stats(array):
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std[std == 0] = 1
        return (array - mean) / std

    if array_two is None:
        array_two = array_one

    columns_one = None
    if _is_dataframe(array_one):
        columns_one = array_one.columns
        array_one = array_one.values
    columns_two = None
    if _is_dataframe(array_two):
        columns_two = array_two.columns
        array_two = array_two.values

    array1_normalized = _get_stats(array_one)
    array2_normalized = _get_stats(array_two)

    # Compute the correlation matrix as dot product
    correlation_matrix = np.dot(array1_normalized.T, array2_normalized) / array_one.shape[0]
    if columns_one is not None and columns_two is not None:
        import pandas as pd

        correlation_matrix = pd.DataFrame(correlation_matrix, index=columns_one, columns=columns_two)
    return correlation_matrix


def feature_redundancy(W, similarity_threshold: float = 0.9) -> tuple[list[tuple[int, int, float]], float, np.ndarray]:
    """Calculate spatial redundancy of NMF components based on cosine similarity of the weight matrix W."""
    from sklearn.metrics.pairwise import cosine_similarity

    W_norm = W / (np.linalg.norm(W, axis=0) + 1e-12)
    sim = cosine_similarity(W_norm.T)
    k = W.shape[1]
    rows, cols = np.triu_indices(k, k=1)
    mask = sim[rows, cols] > similarity_threshold
    redundant_pairs = [
        (int(i), int(j), sim[i, j]) for i, j in zip(rows[mask], cols[mask], strict=True)
    ]
    total_possible = k * (k - 1) / 2
    redundant_percent = 100 * (len(redundant_pairs) / total_possible) if total_possible else 0.0
    return redundant_pairs, redundant_percent, sim


def get_significant_correlations(
    correlation_matrix: np.ndarray | pd.DataFrame,
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Get significant correlations."""
    if _is_dataframe(correlation_matrix):
        correlation_matrix = correlation_matrix.values

    correlation_matrix = np.asarray(correlation_matrix)
    rows, cols = np.nonzero(np.abs(correlation_matrix) > threshold)
    pairs = [(int(i), int(j), float(correlation_matrix[i, j])) for i, j in zip(rows, cols, strict=True)]
    return sorted(pairs, key=lambda x: x[2])


def get_n_significant_correlations(
    correlation_matrix: np.ndarray | pd.DataFrame,
    n: int,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Get the n highest and lowest correlations."""
    if _is_dataframe(correlation_matrix):
        correlation_matrix = correlation_matrix.values

    correlation_matrix = np.asarray(correlation_matrix)
    rows, cols = np.triu_indices(correlation_matrix.shape[0], k=1, m=correlation_matrix.shape[1])
    pairs = [(int(i), int(j), float(correlation_matrix[i, j])) for i, j in zip(rows, cols, strict=True)]
    pairs = sorted(pairs, key=lambda x: x[2])
    anti_correlated = pairs[:n]
    correlated = pairs[::-1][:n]
    return anti_correlated, correlated
