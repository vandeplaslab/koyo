"""Regex module."""

from __future__ import annotations

import re

from pathlib import Path


def remove_date_from_filename(filename: str | Path) -> str:
    """Remove a YYYY_MM_DD date (and one adjacent underscore) from the basename of `filename`.

    Examples
    --------
    >>> remove_date_from_filename("2024_01_15_run.d")
    'run.d'
    >>> remove_date_from_filename("/data/run_2024_01_15.d")
    'run.d'
    """
    name = Path(filename).name

    # `_` is a word character, so `\b` can't be used to detect a date preceded by an underscore
    return re.sub(r"_\d{4}_\d{2}_\d{2}(?![0-9])|(?<![0-9])\d{4}_\d{2}_\d{2}_?", "", name)
