"""Tests for koyo.compat."""

import collections
import typing as ty

from koyo.compat import enable_compat


def test_enable_compat_is_noop():
    assert enable_compat() is None


def test_collections_callable_is_available():
    assert hasattr(collections, "Callable")
    assert collections.Callable is ty.Callable
