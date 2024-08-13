"""Test that click produces the expected output."""

import pytest
from koyo.click import parse_arg


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("n=5", ("n", 5)),
        ("n=5.0", ("n", 5.0)),
        ("n=[10,20,30]", ("n", [10, 20, 30])),
        ("n=value=with=equal=sign", ("n", "value=with=equal=sign")),
    ],
)
def test_parse_arg(arg, expected):
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
def test_parse_arg(arg, expected):
    res_key, res_value = parse_arg(arg, "--un:")
    assert res_key == expected[0], f"Expected {expected[0]}, got {res_key}"
    assert res_value == expected[1], f"Expected {expected[1]}, got {res_value}"
