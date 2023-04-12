import time

import pytest
from koyo.timer import format_time, time_average, time_loop


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.01, "us"),
        (0.1, "ms"),
        (0.5, "s"),
        (1, "s"),
        (60, "s"),
        (75, "min"),
        (3654, "hr"),
    ],
)
def test_format_time(value, expected):
    """Test 'format_time'"""
    result = format_time(value)
    assert expected in result


def test_time_average():
    """Test 'time_average'"""
    t_start = time.time()
    time.sleep(0.01)
    result = time_average(t_start, 1)
    assert "Avg:" in result
    assert "Tot:" in result
    assert result.startswith("[")
    assert result.endswith("]")


def test_time_loop():
    """Test 'time_loop'"""
    t_start = time.time()
    time.sleep(0.01)
    result = time_loop(t_start, 0, 0)
    assert "Avg:" in result
    assert "Rem:" in result
    assert "Tot:" in result
    assert "%" in result

    result = time_loop(t_start, 0, 0, as_percentage=False)
    assert "%" not in result
    assert result.startswith("[")
    assert result.endswith("]")
