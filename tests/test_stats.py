"""Tests for koyo.stats."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from koyo.stats import (
    get_n_significant_correlations,
    get_significant_correlations,
    pairwise_pearsonr,
    pairwise_spearmanr,
    rank_features,
)

# ---------------------------------------------------------------------------
# rank_features
# ---------------------------------------------------------------------------


def test_rank_features_shape():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [10, 30, 20]})
    result = rank_features(df)
    assert result.shape == df.shape


def test_rank_features_values():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = rank_features(df)
    # Sorted ascending → ranks 1, 2, 3
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])


# ---------------------------------------------------------------------------
# pairwise_spearmanr
# ---------------------------------------------------------------------------


def test_pairwise_spearmanr_self_correlation_shape():
    arr = np.random.default_rng(0).random((20, 4))
    result = pairwise_spearmanr(arr)
    assert result.shape == (4, 4)


def test_pairwise_spearmanr_dataframe_output():
    df1 = pd.DataFrame(np.arange(12).reshape(4, 3), columns=["a", "b", "c"])
    df2 = pd.DataFrame(np.arange(8).reshape(4, 2), columns=["x", "y"])
    result = pairwise_spearmanr(df1, df2)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["a", "b", "c"]
    assert list(result.columns) == ["x", "y"]


def test_pairwise_spearmanr_ranked_path():
    arr = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    result_ranked = pairwise_spearmanr(arr, ranked=True)
    result_unranked = pairwise_spearmanr(arr, ranked=False)
    # Both should return a (2, 2) matrix
    assert result_ranked.shape == (2, 2)
    assert result_unranked.shape == (2, 2)


def test_pairwise_spearmanr_one_array():
    arr = np.random.default_rng(1).random((10, 3))
    result = pairwise_spearmanr(arr)
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# pairwise_pearsonr
# ---------------------------------------------------------------------------


def test_pairwise_pearsonr_shape():
    arr = np.random.default_rng(2).random((20, 5))
    result = pairwise_pearsonr(arr)
    assert result.shape == (5, 5)


def test_pairwise_pearsonr_self_diagonal():
    arr = np.random.default_rng(3).random((50, 4))
    result = pairwise_pearsonr(arr)
    # Diagonal should be ~1 (each column correlated with itself)
    np.testing.assert_allclose(np.diag(result), np.ones(4), atol=1e-10)


def test_pairwise_pearsonr_dataframe_output():
    df1 = pd.DataFrame(np.random.default_rng(4).random((10, 2)), columns=["a", "b"])
    df2 = pd.DataFrame(np.random.default_rng(5).random((10, 3)), columns=["x", "y", "z"])
    result = pairwise_pearsonr(df1, df2)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# get_significant_correlations
# ---------------------------------------------------------------------------


def test_get_significant_correlations_filters():
    mat = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]])
    pairs = get_significant_correlations(mat, threshold=0.5)
    # Only values > 0.5 should appear (diagonal = 1.0, off-diag 0.8)
    for _, _, val in pairs:
        assert abs(val) > 0.5


def test_get_significant_correlations_sorted():
    mat = np.array([[1.0, 0.9, 0.6], [0.9, 1.0, 0.7], [0.6, 0.7, 1.0]])
    pairs = get_significant_correlations(mat, threshold=0.0)
    values = [v for _, _, v in pairs]
    assert values == sorted(values)


def test_get_significant_correlations_dataframe():
    df = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]])
    pairs = get_significant_correlations(df, threshold=0.5)
    assert len(pairs) > 0


# ---------------------------------------------------------------------------
# get_n_significant_correlations
# ---------------------------------------------------------------------------


def test_get_n_significant_correlations_counts():
    mat = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.7], [0.3, 0.7, 1.0]])
    anti, corr = get_n_significant_correlations(mat, n=1)
    assert len(anti) == 1
    assert len(corr) == 1


def test_get_n_significant_correlations_ordering():
    mat = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.7], [0.3, 0.7, 1.0]])
    anti, corr = get_n_significant_correlations(mat, n=2)
    # anti-correlated should have lower values than correlated
    assert anti[-1][2] <= corr[-1][2]


def test_get_n_significant_correlations_dataframe():
    df = pd.DataFrame([[1.0, 0.9, 0.3], [0.9, 1.0, 0.7], [0.3, 0.7, 1.0]])
    anti, corr = get_n_significant_correlations(df, n=1)
    assert len(anti) == 1
    assert len(corr) == 1
