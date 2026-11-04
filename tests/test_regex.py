"""Tests for koyo.regex."""

import pytest

from koyo.regex import remove_date_from_filename


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("2024_01_15_run.d", "run.d"),
        ("run_2024_01_15.d", "run.d"),
        ("run_2024_01_15_x.d", "run_x.d"),
        ("/data/run_2024_01_15.d", "run.d"),
        ("no_date.d", "no_date.d"),
    ],
)
def test_remove_date_from_filename(filename, expected):
    assert remove_date_from_filename(filename) == expected
