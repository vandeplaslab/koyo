import time

import pytest
from koyo.timer import MeasureTimer, format_datetime_ago, format_human_time, format_time, time_average, time_loop


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


# ---------------------------------------------------------------------------
# format_datetime_ago
# ---------------------------------------------------------------------------


def test_format_datetime_ago_empty_string():
    assert format_datetime_ago("") == ""


def test_format_datetime_ago_invalid_format():
    result = format_datetime_ago("not-a-date")
    assert result == "Invalid datetime format"


def test_format_datetime_ago_just_now():
    from datetime import datetime, timedelta

    recent = (datetime.utcnow() - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = format_datetime_ago(recent)
    assert result == "just now"


def test_format_datetime_ago_minutes():
    from datetime import datetime, timedelta

    past = (datetime.utcnow() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = format_datetime_ago(past)
    assert "minute" in result


def test_format_datetime_ago_hours():
    from datetime import datetime, timedelta

    past = (datetime.utcnow() - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = format_datetime_ago(past)
    assert "hour" in result


def test_format_datetime_ago_days():
    from datetime import datetime, timedelta

    past = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = format_datetime_ago(past)
    assert "day" in result


# ---------------------------------------------------------------------------
# MeasureTimer
# ---------------------------------------------------------------------------


def test_measure_timer_elapsed_positive():
    timer = MeasureTimer()
    time.sleep(0.005)
    assert timer.elapsed() > 0


def test_measure_timer_call_returns_string():
    timer = MeasureTimer()
    time.sleep(0.005)
    result = timer()
    assert isinstance(result, str)


def test_measure_timer_context_manager():
    with MeasureTimer() as timer:
        time.sleep(0.005)
    assert timer.end is not None


def test_measure_timer_stopwatch():
    timer = MeasureTimer()
    time.sleep(0.005)
    timer.stopwatch("step1")
    assert len(timer.steps) == 1


def test_format_human_time_nanoseconds():
    result = format_human_time(500)
    assert "ns" in result


def test_format_human_time_milliseconds():
    result = format_human_time(5 * 1e6)
    assert "ms" in result
