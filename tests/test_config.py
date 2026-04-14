"""Tests for koyo.config.BaseConfig."""

import typing as ty
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from koyo.config import BaseConfig

# ---------------------------------------------------------------------------
# Minimal concrete config for testing
# ---------------------------------------------------------------------------


class _TmpConfig(BaseConfig):
    model_config = {"arbitrary_types_allowed": True}

    name: str = "default"
    count: int = 0
    enabled: bool = True

    # Class variable — set by each fixture
    USER_CONFIG_DIR: ty.ClassVar[Path]
    USER_CONFIG_FILENAME: ty.ClassVar[str] = "config.json"


@pytest.fixture
def cfg(tmp_path):
    _TmpConfig.USER_CONFIG_DIR = tmp_path
    return _TmpConfig()


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_file(cfg, tmp_path):
    cfg.name = "saved"
    cfg.save()
    assert (tmp_path / "config.json").exists()


def test_load_restores_values(cfg, tmp_path):
    cfg.name = "restored"
    cfg.count = 7
    cfg.save()

    cfg2 = _TmpConfig()
    cfg2.load()
    assert cfg2.name == "restored"
    assert cfg2.count == 7


def test_load_nonexistent_file(cfg):
    # Should not raise even if the file doesn't exist yet
    cfg.load()


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_changes_value(cfg, tmp_path):
    cfg.update(name="updated", save=False)
    assert cfg.name == "updated"


def test_update_saves_when_changed(cfg, tmp_path):
    cfg.update(name="auto_saved")
    assert (tmp_path / "config.json").exists()


def test_update_ignores_unknown_key(cfg):
    # Should not raise; unknown keys are silently warned
    cfg.update(nonexistent_key="value", save=False)


# ---------------------------------------------------------------------------
# temporary_overwrite
# ---------------------------------------------------------------------------


def test_temporary_overwrite_reverts(cfg):
    original_name = cfg.name
    with cfg.temporary_overwrite(name="temp"):
        assert cfg.name == "temp"
    assert cfg.name == original_name


def test_temporary_overwrite_multiple_fields(cfg):
    with cfg.temporary_overwrite(name="t", count=99):
        assert cfg.name == "t"
        assert cfg.count == 99
    assert cfg.name == "default"
    assert cfg.count == 0


def test_temporary_overwrite_reverts_after_exception(cfg):
    with pytest.raises(RuntimeError, match="boom"), cfg.temporary_overwrite(name="temp", count=5):
        raise RuntimeError("boom")
    assert cfg.name == "default"
    assert cfg.count == 0


# ---------------------------------------------------------------------------
# output_path
# ---------------------------------------------------------------------------


def test_output_path(cfg, tmp_path):
    assert cfg.output_path == tmp_path / "config.json"


def test_auto_load_reads_existing_file(tmp_path):
    _TmpConfig.USER_CONFIG_DIR = tmp_path
    stored = _TmpConfig(name="stored", count=3)
    stored.save()

    cfg = _TmpConfig(_auto_load=True)
    assert cfg.name == "stored"
    assert cfg.count == 3
