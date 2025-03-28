"""Timer utilities."""

from __future__ import annotations

import time
import typing as ty
from contextlib import contextmanager
from datetime import datetime, timedelta

PERIODS = [
    ("year", 60 * 60 * 24 * 365 * 1e9),
    ("month", 60 * 60 * 24 * 30 * 1e9),
    ("day", 60 * 60 * 24 * 1e9),
    ("h", 60 * 60 * 1e9),
    ("min", 60 * 1e9),
    ("s", 1 * 1e9),
    ("ms", 1 * 1e6),
    ("μs", 1 * 1e3),
    ("ns", 1),
]


def format_datetime_ago(datetime_str: str) -> str:
    """
    Format a datetime string into a human-readable 'time ago' format.

    Parameters
    ----------
    datetime_str : str
        Datetime string in the format 'YYYY-MM-DDTHH:MM:SSZ'.

    Returns
    -------
    str
        Human-readable 'time ago' format.
    """
    if not datetime_str:
        return ""

    try:
        # Parse the input datetime string
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")

        # Calculate the time difference from the current time
        now = datetime.utcnow()
        time_difference = now - datetime_obj

        # Define time units
        units = [
            ("year", timedelta(days=365)),
            ("month", timedelta(days=30)),
            ("day", timedelta(days=1)),
            ("hour", timedelta(hours=1)),
            ("minute", timedelta(minutes=1)),
        ]

        # Find the largest time unit that fits
        for unit, delta in units:
            if time_difference >= delta:
                time_ago = time_difference // delta
                if time_ago > 1:
                    unit += "s"
                return f"{time_ago} {unit} ago"

        # If less than a minute, show 'just now'
        return "just now"

    except ValueError:
        return "Invalid datetime format"


def format_human_time_s(seconds: float, n_max: int = 2) -> str:
    """Format time so its human-readable."""
    return format_human_time(seconds * 1e9, n_max)


def format_human_time(nanoseconds: float, n_max: int = 2) -> str:
    """Format time so its human-readable."""
    values = []
    for period_name, period_seconds in PERIODS:
        if nanoseconds > period_seconds:
            period_value, nanoseconds = divmod(nanoseconds, period_seconds)
            values.append(f"{int(period_value)}{period_name}")
    if len(values) > n_max:
        values = values[0:n_max]
    return " ".join(values)


def format_time(seconds: float) -> str:
    """Convert time to nicer format. Value is provided in seconds."""
    if seconds <= 0.01:
        return f"{seconds * 1000000:.0f}us"
    elif seconds <= 0.1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds > 86400:
        return f"{seconds / 86400:.2f}day"
    elif seconds > 1800:
        return f"{seconds / 3600:.2f}hr"
    elif seconds > 60:
        return f"{seconds / 60:.2f}min"
    return f"{seconds:.2f}s"


class MeasureTimer:
    """Timer class."""

    def __init__(self, func: ty.Callable | None = None, msg: str | None = None, human: bool = True):
        self.func = func
        self.msg = msg
        self.human = human
        if self.func and not self.msg:
            self.msg = "Execution took: {}"
        if self.msg and "{}" not in self.msg:
            self.msg += " in {}"

        self.start: float = float(time.perf_counter_ns() if human else time.perf_counter())
        self.end: float | None = None
        self.last: float | None = None
        self.steps: list[str] = []

    def stopwatch(self, name: str = "") -> None:
        """Add a step to the timer."""
        self.steps.append(self(since_last=True, name=name, with_steps=False))

    def finish(self) -> MeasureTimer:
        """Finish the timer."""
        self.end = self.current()
        return self

    def current(self) -> float:
        """Return current time."""
        return time.perf_counter_ns() if self.human else time.perf_counter()

    def elapsed(self, n: int = 1, start: float | None = None) -> float:
        """Return amount of time that elapsed."""
        end = self.end or self.current()
        start = start or self.start
        elapsed = end - start
        elapsed = elapsed / n
        self.last = end
        return elapsed

    def elapsed_since_last(self) -> float:
        """Elapsed since last time."""
        return self.elapsed(start=self.last)

    def format(self, elapsed: int) -> str:
        """Format time."""
        return format_human_time(elapsed) if self.human else format_time(elapsed)

    def __call__(
        self,
        n: int = 1,
        current: int = 1,
        start: float | None = None,
        since_last: bool = False,
        with_steps: bool = True,
        name: str = "",
    ) -> str:
        if since_last:
            start = self.last
        elapsed = self.elapsed(n, start)
        formatted = format_human_time(elapsed) if self.human else format_time(elapsed)
        if n > 1:
            formatted = f"{formatted} [{current}/{n}]"
        if name:
            formatted = f"{formatted} ({name})"
        if with_steps and self.steps:
            formatted += f" | {', '.join(self.steps)}"
        return formatted

    def __repr__(self) -> str:
        """Return nicely formatted execution time."""
        return self()

    def __str__(self) -> str:
        """Return nicely formatted execution time."""
        return self()

    def __enter__(self) -> MeasureTimer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        if self.func and self.msg:
            self.func(self.msg.format(self()))


@contextmanager
def measure_time():
    """Function that measures how long a function took place.

    Examples
    --------
    func = lambda: time.sleep(1)
    with measure_time() as t:
        func()
    t()
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


@contextmanager
def report_measure_time(human: bool = True) -> ty.Generator[ty.Callable, None, None]:
    """Report measured time and print it."""
    if human:
        start = time.perf_counter_ns()
        yield lambda: format_human_time(time.perf_counter_ns() - start)
    else:
        start = time.perf_counter()
        yield lambda: format_time(time.perf_counter() - start)


def report_time(t_start: float):
    """Returns nicely formatted execution time."""
    return format_time(time.time() - t_start)


def time_loop(t_start: float, n_item: int, n_total: int, as_percentage: bool = True) -> str:
    """Calculate average, remaining and total times.

    Parameters
    ----------
    t_start : float
        starting time of the for loop
    n_item : int
        index of the current item - assumes index starts at 0
    n_total : int
        total number of items in the for loop - assumes index starts at 0
    as_percentage : bool, optional
        if 'True', progress will be displayed as percentage rather than the raw value

    Returns
    -------
    timed : str
        loop timing information
    """
    t_tot = time.time() - t_start
    t_avg = t_tot / (n_item + 1)
    t_rem = t_avg * (n_total - n_item + 1)

    # calculate progress
    progress = f"{n_item}/{n_total + 1}"
    if as_percentage:
        progress = f"{(n_item / (n_total + 1)) * 100:.1f}% ({progress})"

    return f"[Avg: {format_time(t_avg)} | Rem: {format_time(t_rem)} | Tot: {format_time(t_tot)} || {progress}]"


def time_average(t_start: float, n_total: int) -> str:
    """Calculate average and total time of a task.

    Parameters
    ----------
    t_start : float
        starting time of the task
    n_total : int
        total number of items

    Returns
    -------
    value : str
        formatted text
    """
    t_tot = time.time() - t_start
    t_avg = t_tot / (n_total + 1)

    return f"[Avg: {format_time(t_avg)} | Tot: {format_time(t_tot)}]"


class Timer:
    """Timer class."""

    def __init__(self, value=None, init=False):
        """Initialize timer."""
        self.timers = {}
        self._last_key = None
        if value:
            self.append(value)

        if init:
            self.qappend()

    def reset(self):
        """Reset timer."""
        self.timers = {}

    def append(self, value, label=""):
        """Adds new timed object to the dict."""
        _n = len(self.timers)
        if label == "":
            label = f"timer {_n + 1}"
        self.timers[_n] = [value, label]
        self._last_key = _n

    def qappend(self, label=""):
        self.append(time.time(), label)

    def get(self, add_total=True, sep="\n"):
        """Return computed timings."""
        keys = list(self.timers.keys())
        n_keys = len(keys)

        if n_keys < 2:
            return

        # get total
        total = time.time() - self.timers[0][0]

        timings = []
        for key in reversed(keys):
            value_curr, label = self.timers[key]
            if key >= 1:
                value_prev, __ = self.timers[key - 1]
                value = value_curr - value_prev
                try:
                    value_percent = (value / total) * 100
                except ZeroDivisionError:
                    value_percent = 0
                timings.append(f"{label}: {format_time(value)} ({value_percent:.2f}%)")

        # ensures nicer formatting
        timings.append("")

        # add total
        if add_total:
            timings.insert(0, f"TOTAL: {total:.4f}s")

        return sep.join(reversed(timings))

    def show(self, add_total=True, sep="\n"):
        """Returns computed timings."""
        self.get(add_total, sep)

    def last(self):
        """Retrieve last time."""
        return self.timers[self._last_key][0]

    def first(self):
        """Retrieve first time."""
        return self.timers[0][0]
