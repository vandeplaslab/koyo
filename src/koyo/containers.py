"""Container classes."""

from __future__ import annotations

import time
import typing as ty
from abc import abstractmethod
from keyword import iskeyword
from typing import overload

from koyo.utilities import is_valid_python_name

_T = ty.TypeVar("_T")
_K = ty.TypeVar("_K")


def is_valid_python_name(name):
    """Check whether the key is valid identifier."""
    return name.isidentifier() and not iskeyword(name)


class AttributeDict:
    """Simple wrapper around schema dictionary."""

    def __init__(self, data: dict):
        self._data = data

    def __repr__(self):
        return repr(self._data)

    def __getitem__(self, key: str | int) -> ty.Any:
        """Get item id."""
        return self._data[key]

    def __getattr__(self, item: str | int) -> ty.Any:
        # allow access to group members via dot notation
        try:
            return self.__getitem__(item)
        except KeyError as err:
            raise AttributeError from err

    def __dir__(self):
        # noinspection PyUnresolvedReferences
        base = super().__dir__()
        keys = sorted(set(base + list(self._data.keys())))
        keys = [k for k in keys if is_valid_python_name(k)]
        return keys

    def _ipython_key_completions_(self):
        return sorted(self)


class MutableSequence(ty.MutableSequence[_T]):
    """List object that holds objects with optional check if it exists."""

    def __init__(self, data: ty.Iterable[_T] = ()):
        self._list: list[_T] = []
        if data:
            self.extend(data)

    def insert(self, index: int, value: _T) -> None:
        """Insert formula."""
        self._list.insert(index, self._check(value))

    @ty.overload
    def __getitem__(self, i: int) -> _T: ...

    @ty.overload
    def __getitem__(self, s: slice) -> ty.MutableSequence[_T]: ...

    def __getitem__(self, i: int) -> _T:
        return self._list[i]

    @ty.overload
    def __setitem__(self, i: int, o: _T) -> None: ...

    @ty.overload
    def __setitem__(self, s: slice, o: ty.Iterable[_T]) -> None: ...

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            if not isinstance(value, ty.Iterable):
                raise TypeError("Can only assign an iterable to slice.")
            self._list[key] = [self._check(v) for v in value]
        else:
            self._list[key] = self._check(value)

    @ty.overload
    def __delitem__(self, i: int) -> None: ...

    @ty.overload
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i) -> None:
        del self._list[i]

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self):
        return repr(self._list)

    def __eq__(self, other: ty.Any):
        return self._list == other

    def __hash__(self) -> int:
        # it's important to add this to allow this object to be hashable
        # given that we've also reimplemented __eq__
        return id(self)

    def _check(self, value: _T) -> _T:
        """Check whether duplicate items exist in the list."""
        return value

    def iter(self):
        """Iterator."""
        yield from self._list


class MutableMapping(ty.MutableMapping[_K, _T]):
    """Mutable mapping instance."""

    def __init__(self, data: dict[_K, _T] | None = None):
        self._dict: dict[_K, _T] = {}
        if data is not None:
            self.update(data)

    def __dir__(self) -> list[str]:
        # noinspection PyUnresolvedReferences
        base = super().__dir__()
        keys = sorted(set(base + list(self) + list(self._dict.keys())))  # type: ignore[operator]
        keys = [k for k in keys if is_valid_python_name(k)]
        return keys

    def _ipython_key_completions_(self) -> list[str]:
        return sorted(self)

    def __setitem__(self, k: _K, v: _T) -> None:
        self._dict[k] = v

    def __delitem__(self, v: _K) -> None:
        del self._dict[v]

    def __getitem__(self, k: _K) -> _T:
        return self._dict[k]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> ty.Iterator[_K]:
        yield from self._dict.keys()


class SizedDict(ty.OrderedDict):
    """Sized dictionary."""

    def __init__(self, *args: dict[_K, _T], maxsize: int = -1, **kwargs: ty.Any):
        self._maxsize = maxsize
        super().__init__(*args, **kwargs)

    def _check_size(self) -> None:
        if self._maxsize > 0:
            while len(self) > self._maxsize:
                self.popitem(last=False)

    def __setitem__(self, key: _K, value: _T) -> None:
        super().__setitem__(key, value)
        self._check_size()

    def update(self, m: dict[_K, _T], **kwargs: ty.Any) -> None:
        """Override update."""
        super().update(m, **kwargs)
        self._check_size()


class ShortLivedDict(dict):
    """Short-lived dictionary that expires after a given time-to-live (TTL)."""

    def __init__(self, ttl: float = 5.0):
        super().__init__()
        self.ttl = ttl
        self._created = time.time()

    def __getitem__(self, key: _K) -> _T:
        self._check_expired()
        return super().__getitem__(key)

    def __setitem__(self, key: _K, value: _T) -> None:
        self._check_expired()
        return super().__setitem__(key, value)

    def _expired(self) -> bool:
        return time.time() - self._created > self.ttl

    def _check_expired(self):
        if self._expired():
            super().clear()
            self._created = time.time()

    def get(self, key: _K, default: _T | None = None) -> _T | None:
        """Get item with default value."""
        self._check_expired()
        return super().get(key, default)

    def keys(self) -> ty.KeysView[_K]:
        """Get items."""
        self._check_expired()
        return super().keys()

    def items(self) -> ty.ItemsView[_K, _T]:
        """Get items."""
        self._check_expired()
        return super().items()

    def clear(self) -> None:
        """Clear the dictionary."""
        self._created = time.time()
        return super().clear()


class SizedList(ty.MutableSequence[_T]):
    """Sized list."""

    def __init__(self, *args: ty.Iterable[_T], maxsize: int = -1, **kwargs: ty.Any):
        self._maxsize = maxsize
        self._list: list[_T] = []
        super().__init__(*args, **kwargs)

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> _T: ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> MutableSequence[_T]: ...

    def __getitem__(self, index: int | slice) -> _T | MutableSequence[_T]:
        return self._list[index]

    @overload
    @abstractmethod
    def __delitem__(self, index: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: int | slice):
        del self._list[index]

    def __len__(self):
        return len(self._list)

    def _check_size(self) -> None:
        if self._maxsize > 0:
            while len(self) > self._maxsize:
                self.pop(0)

    def __setitem__(self, key: int, value: _T) -> None:
        super().__setitem__(key, value)
        self._check_size()

    def insert(self, index: int, value: _T):
        """Insert items."""
        self._list.insert(index, value)
        self._check_size()
