"""Regex module."""

from __future__ import annotations

import re

from path import Path


def remove_date_from_filename(filename: str | Path) -> str:
    """Remove a YYYY_MM_DD date from a filename."""
    name = Path(filename).name

    return re.sub(r"\b\d{4}_\d{2}_\d{2}_?", "", name)
