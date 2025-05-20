"""Fix some compatibility issues between Python versions."""

import collections
import typing as ty

if not hasattr(collections, "Callable"):
    collections.Callable = ty.Callable


def enable_compat():
    """No-op compat."""
