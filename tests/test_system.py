"""Tests for koyo.system utilities."""

import os

import pytest

from koyo.system import (
    get_version,
    is_above_version,
    is_envvar,
    is_envvar_set,
    is_installed,
    is_network_path,
    running_as_pyinstaller_app,
)


# ---------------------------------------------------------------------------
# is_envvar / is_envvar_set
# ---------------------------------------------------------------------------


def test_is_envvar_true(monkeypatch):
    monkeypatch.setenv("_KOYO_TEST_VAR", "hello")
    assert is_envvar("_KOYO_TEST_VAR", "hello")


def test_is_envvar_wrong_value(monkeypatch):
    monkeypatch.setenv("_KOYO_TEST_VAR", "hello")
    assert not is_envvar("_KOYO_TEST_VAR", "world")


def test_is_envvar_missing():
    assert not is_envvar("_KOYO_NONEXISTENT_VAR", "value")


def test_is_envvar_set_true(monkeypatch):
    monkeypatch.setenv("_KOYO_TEST_VAR", "1")
    assert is_envvar_set("_KOYO_TEST_VAR")


def test_is_envvar_set_empty(monkeypatch):
    monkeypatch.setenv("_KOYO_TEST_VAR", "")
    assert not is_envvar_set("_KOYO_TEST_VAR")


def test_is_envvar_set_missing():
    assert not is_envvar_set("_KOYO_NONEXISTENT_VAR")


# ---------------------------------------------------------------------------
# is_installed
# ---------------------------------------------------------------------------


def test_is_installed_present():
    assert is_installed("numpy")


def test_is_installed_submodule():
    assert is_installed("numpy.linalg")


def test_is_installed_missing():
    assert not is_installed("_koyo_definitely_not_installed")


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------


def test_get_version_installed():
    version = get_version("numpy")
    assert version != "N/A"
    # should look like a semver-ish string
    assert "." in version


def test_get_version_missing():
    assert get_version("_koyo_definitely_not_installed") == "N/A"


# ---------------------------------------------------------------------------
# is_above_version
# ---------------------------------------------------------------------------


def test_is_above_version_true():
    # numpy is definitely > 0.1.0
    assert is_above_version("numpy", "0.1.0")


def test_is_above_version_false():
    # numpy is definitely < 9999.0.0
    assert not is_above_version("numpy", "9999.0.0")


def test_is_above_version_missing():
    assert not is_above_version("_koyo_definitely_not_installed", "1.0.0")


# ---------------------------------------------------------------------------
# running_as_pyinstaller_app
# ---------------------------------------------------------------------------


def test_running_as_pyinstaller_app_false():
    # In a normal test environment this should always be False
    assert not running_as_pyinstaller_app()


# ---------------------------------------------------------------------------
# is_network_path
# ---------------------------------------------------------------------------


def test_is_network_path_local(tmp_path):
    # A local temp directory is never a network path
    assert not is_network_path(str(tmp_path))
