import time

import pytest
from koyo.containers import AttributeDict, MutableMapping, MutableSequence, ShortLivedDict, SizedDict, SizedList


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


# ---------------------------------------------------------------------------
# AttributeDict
# ---------------------------------------------------------------------------


def test_attribute_dict_getitem():
    d = AttributeDict({"a": 1, "b": 2})
    assert d["a"] == 1


def test_attribute_dict_getattr():
    d = AttributeDict({"x": 42})
    assert d.x == 42


def test_attribute_dict_attr_missing():
    d = AttributeDict({"x": 42})
    with pytest.raises(AttributeError):
        _ = d.nonexistent


def test_attribute_dict_dir_contains_keys():
    d = AttributeDict({"alpha": 1, "beta": 2})
    names = dir(d)
    assert "alpha" in names
    assert "beta" in names


# ---------------------------------------------------------------------------
# MutableMapping extras
# ---------------------------------------------------------------------------


def test_mapping_len():
    class T(MutableMapping[str, int]):
        pass

    t = T()
    t["a"] = 1
    t["b"] = 2
    assert len(t) == 2


def test_mapping_iter():
    class T(MutableMapping[str, int]):
        pass

    t = T()
    t["x"] = 10
    t["y"] = 20
    assert set(t) == {"x", "y"}


# ---------------------------------------------------------------------------
# SizedDict eviction behavior
# ---------------------------------------------------------------------------


def test_sized_dict_evicts_oldest():
    data = SizedDict(maxsize=2)
    data["a"] = 1
    data["b"] = 2
    data["c"] = 3
    assert "a" not in data
    assert "b" in data
    assert "c" in data


def test_sized_dict_update_also_trims():
    data = SizedDict(maxsize=2)
    data["a"] = 1
    data.update({"b": 2, "c": 3})
    assert len(data) == 2


def test_sized_dict_unlimited():
    data = SizedDict(maxsize=-1)
    for i in range(100):
        data[i] = i
    assert len(data) == 100


# ---------------------------------------------------------------------------
# ShortLivedDict
# ---------------------------------------------------------------------------


def test_short_lived_dict_within_ttl():
    d = ShortLivedDict(ttl=60.0)
    d["k"] = "v"
    assert d["k"] == "v"


def test_short_lived_dict_expires(monkeypatch):
    d = ShortLivedDict(ttl=1.0)
    d["k"] = "v"
    # Simulate expiry by advancing time
    monkeypatch.setattr(time, "time", lambda: d._created + 2.0)
    # After expiry, clear is triggered on next access
    keys = list(d.keys())
    assert "k" not in keys


def test_short_lived_dict_get_default():
    d = ShortLivedDict(ttl=60.0)
    assert d.get("missing", "default") == "default"


def test_short_lived_dict_clear_resets_timer(monkeypatch):
    d = ShortLivedDict(ttl=1.0)
    d["a"] = 1
    original_created = d._created
    d.clear()
    # _created should be refreshed
    assert d._created >= original_created
