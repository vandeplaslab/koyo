import pytest
from koyo.containers import MutableMapping, MutableSequence, SizedDict, SizedList


def test_mapping():
    class T(MutableMapping[str, int]):
        """Test class."""

    t = T()
    t["a"] = 1
    t.update({"b": 2})
    assert t["a"] == 1
    assert t["b"] == 2
    del t["a"]
    assert "a" not in t


def test_sequence():
    class T(MutableSequence[int]):
        """Test class."""

        def _check(self, value: int) -> int:
            if value in self._list:
                raise ValueError("Duplicate value")
            return value

    t = T()
    t.append(1)
    with pytest.raises(ValueError):
        t.append(1)
    del t[0]
    assert len(t) == 0
    t.append(1)


def test_sized_dict():
    d = SizedDict(maxsize=2)
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3
    assert len(d) == 2


def test_sized_list():
    l = SizedList(maxsize=2)
    l.append(1)
    l.append(2)
    l.append(3)
    assert len(l) == 2
