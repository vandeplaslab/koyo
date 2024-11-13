"""Statistics functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as ss


def rank_features(df: pd.DataFrame) -> np.ndarray:
    """Rank features."""
    return np.apply_along_axis(ss.rankdata, 0, df)


def pairwise_spearmanr(
    array_one: np.ndarray | pd.DataFrame, array_two: np.ndarray | pd.DataFrame | None = None, ranked: bool = False
) -> np.ndarray | pd.DataFrame:
    """Compute pairwise Spearman rank correlation between columns of two arrays."""
    if array_two is None:
        array_two = array_one

    columns_one = None
    if isinstance(array_one, pd.DataFrame):
        columns_one = array_one.columns
        array_one = array_one.values
    columns_two = None
    if isinstance(array_two, pd.DataFrame):
        columns_two = array_two.columns
        array_two = array_two.values
    if not ranked:
        array_one = np.apply_along_axis(ss.rankdata, 0, array_one)
        array_two = np.apply_along_axis(ss.rankdata, 0, array_two)
    am = array_one - np.mean(array_one, axis=0, keepdims=True)
    bm = array_two - np.mean(array_two, axis=0, keepdims=True)

    correlation_matrix = (
        am.T
        @ bm
        / (np.sqrt(np.sum(am**2, axis=0, keepdims=True) + 1).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)) + 1)
    )
    if columns_one is not None and columns_two is not None:
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
    if isinstance(array_one, pd.DataFrame):
        columns_one = array_one.columns
        array_one = array_one.values
    columns_two = None
    if isinstance(array_two, pd.DataFrame):
        columns_two = array_two.columns
        array_two = array_two.values

    array1_normalized = _get_stats(array_one)
    array2_normalized = _get_stats(array_two)

    # Compute the correlation matrix as dot product
    correlation_matrix = np.dot(array1_normalized.T, array2_normalized) / array_one.shape[0]
    if columns_one is not None and columns_two is not None:
        correlation_matrix = pd.DataFrame(correlation_matrix, index=columns_one, columns=columns_two)
    return correlation_matrix


def get_significant_correlations(
    correlation_matrix: np.ndarray | pd.DataFrame, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Get significant correlations."""
    if isinstance(correlation_matrix, pd.DataFrame):
        correlation_matrix = correlation_matrix.values

    pairs = [
        (i, j, float(correlation_matrix[i, j]))
        for i in range(correlation_matrix.shape[0])
        for j in range(correlation_matrix.shape[1])
        if abs(correlation_matrix[i, j]) > threshold
    ]
    pairs = sorted(pairs, key=lambda x: x[2])
    return pairs


def get_n_significant_correlations(
    correlation_matrix: np.ndarray | pd.DataFrame, n: int
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Get the n highest and lowest correlations"""
    if isinstance(correlation_matrix, pd.DataFrame):
        correlation_matrix = correlation_matrix.values

    pairs = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            pairs.append((i, j, float(correlation_matrix[i, j])))
    pairs = sorted(pairs, key=lambda x: x[2])
    anti_correlated = pairs[:n]
    correlated = pairs[-n:][::-1]
    return anti_correlated, correlated
