import time
import warnings

import pytest

from koyo.decorators import classproperty, deprecated, renamed_parameter, retry


class Dummy:
    @classproperty
    def label(cls) -> str:
        return cls.__name__


def test_classproperty_accessible_from_class_and_instance():
    assert Dummy.label == "Dummy"
    assert Dummy().label == "Dummy"


def test_deprecated_emits_warning_and_returns_value():
    @deprecated
    def add_one(value: int) -> int:
        return value + 1

    with pytest.warns(DeprecationWarning):
        assert add_one(2) == 3


def test_retry_retries_until_success(monkeypatch):
    calls: list[int] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda delay: sleep_calls.append(delay))

    @retry(exception=ValueError, n_tries=3, delay=0, backoff=1)
    def sometimes_fails() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("fail")
        return "ok"

    assert sometimes_fails() == "ok"
    assert len(calls) == 3
    assert sleep_calls == [0, 0]


def test_retry_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    @retry(exception=RuntimeError, n_tries=2, delay=0)
    def always_fails() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        always_fails()


def test_renamed_parameter_translates_and_warns():
    @renamed_parameter({"old": "new"})
    def target(*, new: int) -> int:
        return new * 2

    with pytest.warns(UserWarning, match="renamed"):
        assert target(old=5) == 10

    with pytest.raises(TypeError):
        target(old=1, new=2)


def test_renamed_parameter_no_second_warning_when_warn_once():
    @renamed_parameter({"old": "new"}, warn_once=True)
    def echo(*, new: int) -> int:
        return new

    with pytest.warns(UserWarning):
        assert echo(old=1) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert echo(old=2) == 2


def test_renamed_parameter_validate_new_names():
    def build_bad_decorator():
        @renamed_parameter({"old": "missing"})
        def func(new: int) -> int:
            return new

        return func

    with pytest.raises(ValueError):
        build_bad_decorator()
