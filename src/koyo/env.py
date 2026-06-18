"""Environment utilities."""

from __future__ import annotations
import os


def check_environment_variable(var_name: str) -> bool:
    """Check if an environment variable is set."""
    return var_name in os.environ


def get_environment_variable(var_name: str, default: str | None = None) -> str | None:
    """Get the value of an environment variable."""
    return os.environ.get(var_name, default)


def get_environment_variable_bool(var_name: str, default: bool = False) -> bool:
    """Get the value of an environment variable as a boolean."""
    value = os.environ.get(var_name, None)
    if value is None:
        return default
    return value.lower() in ["true", "1", "yes", "y"]
