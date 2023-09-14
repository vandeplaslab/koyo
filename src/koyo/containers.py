import typing as ty

_T = ty.TypeVar("_T")
_K = ty.TypeVar("_K")


class MutableSequence(ty.MutableSequence[_T]):
    """List object that holds objects with optional check if it exists."""

    def __init__(self, data: ty.Iterable[_T] = ()):
        self._list: ty.List[_T] = []
        if data:
            self.extend(data)

    def insert(self, index: int, value: _T) -> None:
        """Insert formula."""
        self._list.insert(index, self._check(value))

    @ty.overload
    def __getitem__(self, i: int) -> _T:
        ...

    @ty.overload
    def __getitem__(self, s: slice) -> ty.MutableSequence[_T]:
        ...

    def __getitem__(self, i):
        return self._list[i]

    @ty.overload
    def __setitem__(self, i: int, o: _T) -> None:
        ...

    @ty.overload
    def __setitem__(self, s: slice, o: ty.Iterable[_T]) -> None:
        ...

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            if not isinstance(value, ty.Iterable):
                raise TypeError("Can only assign an iterable to slice.")
            self._list[key] = [self._check(v) for v in value]
        else:
            self._list[key] = self._check(value)

    @ty.overload
    def __delitem__(self, i: int) -> None:
        ...

    @ty.overload
    def __delitem__(self, i: slice) -> None:
        ...

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

    def __init__(self, data: ty.Optional[ty.Dict[_K, _T]] = None):
        self._dict: ty.Dict[_K, _T] = {}
        if data is not None:
            self.update(data)

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
