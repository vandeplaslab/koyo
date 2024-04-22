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
    data = SizedDict(maxsize=2)
    data["a"] = 1
    data["b"] = 2
    data["c"] = 3
    assert len(data) == 2


def test_sized_list():
    data = SizedList(maxsize=2)
    data.append(1)
    data.append(2)
    data.append(3)
    assert len(data) == 2
