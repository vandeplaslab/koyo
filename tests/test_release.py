"""Tests for release helpers."""

from __future__ import annotations

import pytest

from koyo.release import get_target


def test_get_target_returns_windows_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    assert get_target() == "win_amd64"


def test_get_target_returns_windows_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    assert get_target(installer=True) == "win_amd64_exe"


def test_get_target_ignores_installer_for_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.processor", lambda: "arm")

    assert get_target(installer=True) == "macosx_arm64"
