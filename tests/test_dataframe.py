"""Tests for koyo.dataframe."""

import pytest

pd = pytest.importorskip("pandas")

from koyo.dataframe import read_csv_with_comments


def test_read_csv_with_comments_header_after_hash_row(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\n# comment,skip\ncity,score\nlondon,10\nparis,20\n")
    result = read_csv_with_comments(path)
    assert list(result.columns) == ["city", "score"]
    assert result.to_dict("records") == [{"city": "london", "score": "10"}, {"city": "paris", "score": "20"}]


def test_read_csv_with_comments_parser_error_fallback(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("# comment\n# another\nname,value\nalpha,1\nbeta,2\n\n")
    result = read_csv_with_comments(path)
    assert list(result.columns) == ["name", "value"]
    assert result.to_dict("records") == [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}]


def test_read_csv_with_comments_handles_bad_input_without_crashing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\x00\x00")
    result = read_csv_with_comments(path)
    assert list(result.columns) == ["Unnamed: 0"]
    assert result.empty
