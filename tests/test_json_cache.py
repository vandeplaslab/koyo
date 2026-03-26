"""Tests for koyo.json_cache.JSONCache."""

import pytest

from koyo.json_cache import JSONCache


class MyCache(JSONCache):
    FILENAME = "cache.json"


@pytest.fixture
def cache(tmp_path):
    return MyCache(tmp_path)


# ---------------------------------------------------------------------------
# Basic read / write
# ---------------------------------------------------------------------------


def test_setitem_getitem_roundtrip(cache):
    cache["key"] = "value"
    assert cache["key"] == "value"


def test_setitem_int(cache):
    cache["count"] = 42
    assert cache["count"] == 42


def test_setitem_bool(cache):
    cache["flag"] = True
    assert cache["flag"] is True


def test_setitem_float(cache):
    cache["pi"] = 3.14
    assert cache["pi"] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


def test_contains_present(cache):
    cache["x"] = 1
    assert "x" in cache


def test_contains_absent(cache):
    assert "missing" not in cache


# ---------------------------------------------------------------------------
# __delitem__ / remove / remove_multiple
# ---------------------------------------------------------------------------


def test_delitem(cache):
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache


def test_remove(cache):
    cache["b"] = 2
    cache.remove("b")
    assert "b" not in cache


def test_remove_nonexistent(cache):
    # Should not raise
    cache.remove("nonexistent")


def test_remove_multiple(cache):
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache.remove_multiple("a", "b")
    assert "a" not in cache
    assert "b" not in cache
    assert "c" in cache


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_cache(cache):
    cache["x"] = 1
    cache["y"] = 2
    cache.clear()
    assert list(cache.keys()) == []


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_merges_keys(cache):
    cache["a"] = 1
    cache.update(b=2, c=3)
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3


# ---------------------------------------------------------------------------
# get with default
# ---------------------------------------------------------------------------


def test_get_existing(cache):
    cache["k"] = 99
    assert cache.get("k") == 99


def test_get_default(cache):
    assert cache.get("missing", "default") == "default"


def test_get_default_none(cache):
    assert cache.get("missing") is None


# ---------------------------------------------------------------------------
# as_dict / as_str
# ---------------------------------------------------------------------------


def test_as_dict(cache):
    cache["a"] = 1
    cache["b"] = 2
    d = cache.as_dict()
    assert d == {"a": 1, "b": 2}


def test_as_dict_exclude(cache):
    cache["a"] = 1
    cache["b"] = 2
    d = cache.as_dict(exclude=("b",))
    assert "b" not in d
    assert "a" in d


def test_as_str(cache):
    cache["x"] = 10
    s = cache.as_str()
    assert "x" in s
    assert "10" in s


def test_as_str_empty(cache):
    assert cache.as_str() == ""


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("42", 42),
        ("3.14", 3.14),
        ("hello", "hello"),
        (123, 123),
        (True, True),
    ],
)
def test_format_value(raw, expected):
    result = MyCache.format_value(raw)
    assert result == expected


# ---------------------------------------------------------------------------
# keys / items
# ---------------------------------------------------------------------------


def test_keys(cache):
    cache["a"] = 1
    cache["b"] = 2
    assert set(cache.keys()) == {"a", "b"}


def test_items(cache):
    cache["a"] = 1
    pairs = dict(cache.items())
    assert pairs["a"] == 1


# ---------------------------------------------------------------------------
# exists / name / repr
# ---------------------------------------------------------------------------


def test_exists_false_initially(tmp_path):
    c = MyCache(tmp_path / "new_dir")
    assert not c.exists()


def test_exists_true_after_write(cache):
    cache["k"] = 1
    assert cache.exists()


def test_name(tmp_path):
    c = MyCache(tmp_path / "my_project")
    assert c.name == "my_project"


def test_repr(cache):
    assert "MyCache" in repr(cache)
