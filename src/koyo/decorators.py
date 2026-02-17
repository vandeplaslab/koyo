"""Decorators."""

from __future__ import annotations

import functools
import inspect
import logging
import time
import typing as ty
import warnings
from functools import partial, wraps


def retry(
    func: ty.Callable | None = None,
    exception: Exception = Exception,
    n_tries: int = 5,
    delay: float = 5,
    backoff: int = 1,
    logger: bool = False,
) -> ty.Callable:
    """Retry decorator with exponential backoff.

    Parameters
    ----------
    func : typing.Callable, optional
        Callable on which the decorator is applied, by default None
    exception : Exception or tuple of Exceptions, optional
        Exception(s) that invoke retry, by default Exception
    n_tries : int, optional
        Number of tries before giving up, by default 5
    delay : int, optional
        Initial delay between retries in seconds, by default 5
    backoff : int, optional
        Backoff multiplier e.g. value of 2 will double the delay, by default 1
    logger : bool, optional
        Option to log or print, by default False

    Returns
    -------
    typing.Callable
        Decorated callable that calls itself when exception(s) occur.

    Notes
    -----
    Taken from https://stackoverflow.com/a/61093779/4364202

    Examples
    --------
    >>> import random
    >>> @retry(exception=Exception, n_tries=4)
    ... def test_random(text):
    ...    x = random.random()
    ...    if x < 0.5:
    ...        raise Exception("Fail")
    ...    else:
    ...        print("Success: ", text)
    >>> test_random("It works!")
    """
    if func is None:
        return partial(
            retry,
            exception=exception,
            n_tries=n_tries,
            delay=delay,
            backoff=backoff,
            logger=logger,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        ntries, ndelay = n_tries, delay

        while ntries > 1:
            try:
                return func(*args, **kwargs)
            except exception as e:
                msg = f"{e!s}, Retrying in {ndelay} seconds..."
                if logger:
                    logging.warning(msg)
                else:
                    print(msg)
                time.sleep(ndelay)
                ntries -= 1
                ndelay *= backoff

        return func(*args, **kwargs)

    return wrapper


def deprecated(func: ty.Callable, context: str = "") -> ty.Callable:
    """A decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            f"Call to deprecated function {func.__name__}. {context}",
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class classproperty:
    """
    Decorator that converts a method with a single cls argument into a property
    that can be accessed directly from the class.
    """

    def __init__(self, method=None):
        self.fget = method

    def __get__(self, instance, cls=None):
        return self.fget(cls)

    def getter(self, method):
        self.fget = method
        return self


def renamed_parameter(
    renames: ty.Mapping[str, str],
    *,
    error_on_both: bool = True,
    warn_once: bool = True,
    validate_new_names: bool = True,
) -> ty.Callable[[ty.Callable[..., ty.Any]], ty.Callable[..., ty.Any]]:
    """Decorator to mark parameters as renamed."""
    warned: set[str] = set()

    def decorator(fn: ty.Callable[..., ty.Any]) -> ty.Callable[..., ty.Any]:
        """Parameters"""
        sig = inspect.signature(fn)

        if validate_new_names:
            params = sig.parameters
            missing = [new for new in renames.values() if new not in params]
            if missing:
                raise ValueError(
                    f"{fn.__qualname__}: rename target(s) not in signature: {missing}. Signature is: {sig}",
                )

        @functools.wraps(fn)
        def wrapper(*args: ty.Any, **kwargs: ty.Any) -> ty.Any:
            """Wrapper."""
            # 1) Rewrite kwargs BEFORE binding (binding would reject deprecated names)
            if kwargs:
                for old, new in renames.items():
                    if old not in kwargs:
                        continue

                    if new in kwargs:
                        msg = f"{fn.__qualname__}: received both deprecated '{old}' and '{new}'. Use only '{new}'."
                        if error_on_both:
                            raise TypeError(msg)

                        # Keep explicit new, ignore old
                        kwargs.pop(old, None)
                        if (not warn_once) or (old not in warned):
                            warnings.warn(msg + f" Ignoring '{old}'.", stacklevel=2, category=UserWarning)
                            warned.add(old)
                        continue

                    # Move old -> new
                    kwargs[new] = kwargs.pop(old)
                    if (not warn_once) or (old not in warned):
                        warnings.warn(
                            f"{fn.__qualname__}: parameter '{old}' was renamed to '{new}' (please update your call)",
                            stacklevel=2,
                            category=UserWarning,
                        )
                        warned.add(old)

            # 2) Now it's safe to bind (and it still validates *other* mistakes)
            bound = sig.bind_partial(*args, **kwargs)

            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
