"""Test that click produces the expected output."""

import pytest
from koyo.click import (
    parse_arg,
    parse_extra_args,
    parse_int_with_range,
    parse_paths,
    repr_filelist,
    repr_iterable,
    filter_kwargs,
)


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("n=5", ("n", 5)),
        ("n=5.0", ("n", 5.0)),
        ("n=[10,20,30]", ("n", [10, 20, 30])),
        ("n=value=with=equal=sign", ("n", "value=with=equal=sign")),
    ],
)
def test_parse_arg_no_key(arg, expected):
    res_key, res_value = parse_arg(arg, "")
    assert res_key == expected[0], f"Expected {expected[0]}, got {res_key}"
    assert res_value == expected[1], f"Expected {expected[1]}, got {res_value}"


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("--un:n=5", ("n", 5)),
        ("--un:n=5.0", ("n", 5.0)),
        ("--un:n=[10,20,30]", ("n", [10, 20, 30])),
    ],
)
def test_parse_arg_with_key(arg, expected):
    res_key, res_value = parse_arg(arg, "--un:")
    assert res_key == expected[0], f"Expected {expected[0]}, got {res_key}"
    assert res_value == expected[1], f"Expected {expected[1]}, got {res_value}"


@pytest.mark.parametrize(
    "int_range, expected",
    [
        ("1", [1]),
        ("1,2,3", [1, 2, 3]),
        ("1:3", [1, 2, 3]),
        ("1-3", [1, 2, 3]),
        ("0:10:2", [0, 2, 4, 6, 8, 10]),
        ("0-10", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ("1,3,5:7", [1, 3, 5, 6, 7]),
    ],
)
def test_parse_int_with_range(int_range, expected):
    assert parse_int_with_range(int_range) == expected


def test_parse_int_with_range_none():
    assert parse_int_with_range(None) == []


def test_parse_paths_basic(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    result = parse_paths([str(d1), str(d2)])
    assert str(d1) in result
    assert str(d2) in result


def test_parse_paths_glob(tmp_path):
    for name in ("x", "y", "z"):
        (tmp_path / name).mkdir()
    result = parse_paths([str(tmp_path / "*")])
    assert len(result) == 3


def test_parse_paths_trailing_slash(tmp_path):
    d = tmp_path / "mydir"
    d.mkdir()
    result = parse_paths([str(d) + "/"])
    assert result[0] == str(d)


def test_parse_paths_sort(tmp_path):
    for name in ("c", "a", "b"):
        (tmp_path / name).mkdir()
    result = parse_paths([str(tmp_path / "*")], sort=True)
    assert result == sorted(result)


def test_parse_extra_args_basic():
    result = parse_extra_args(("a=1", "b=hello"))
    assert result["a"] == 1
    assert result["b"] == "hello"


def test_parse_extra_args_repeated_key():
    result = parse_extra_args(("a=1", "a=2"))
    assert result["a"] == [1, 2]


def test_parse_extra_args_none():
    assert parse_extra_args(None) == {}


def test_parse_extra_args_skips_no_equals():
    result = parse_extra_args(("no_equals",))
    assert result == {}


def test_repr_filelist(tmp_path):
    paths = [tmp_path / "foo.txt", tmp_path / "bar.txt"]
    result = repr_filelist(paths)
    assert "foo.txt" in result
    assert "bar.txt" in result
    assert "; " in result


def test_repr_iterable_short():
    items = list(range(5))
    assert repr_iterable(items) == items


def test_repr_iterable_long():
    items = list(range(11))
    result = repr_iterable(items)
    assert "11" in str(result)
    assert "items" in str(result)


def test_repr_iterable_none():
    assert repr_iterable(None) is None


def test_filter_kwargs():
    result = filter_kwargs("a", "b", a=1, b=2, c=3)
    assert result == {"a": 1, "b": 2}
    assert "c" not in result
