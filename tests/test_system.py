"""Tests for koyo.system utilities."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from koyo.system import (
    _is_network_linux,
    _linux_sys_name,
    _linux_sys_name_lsb_release,
    check_available_space,
    get_cli_path,
    get_module_path,
    get_version,
    is_above_version,
    is_envvar,
    is_envvar_set,
    is_installed,
    is_network_path,
    reraise_exception_if_debug,
    running_as_pyinstaller_app,
    who_called_me,
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


def test_check_available_space_no_drive_returns_true():
    assert check_available_space(str(Path.cwd()), 10**18)


def test_get_cli_path_from_env(monkeypatch, tmp_path):
    script = tmp_path / "tool"
    script.write_text("")
    monkeypatch.setenv("APP_TOOL_PATH", str(script))
    assert get_cli_path("tool", env_key="APP") == str(script)


def test_get_cli_path_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("_MISSING_TOOL_PATH", raising=False)
    assert get_cli_path("tool", env_key="_MISSING", default="fallback") == "fallback"


def test_get_module_path_points_to_source_file():
    result = get_module_path("koyo", "json")
    assert result.endswith("json.py")
    assert Path(result).exists()


def test_reraise_exception_if_debug_logs(monkeypatch):
    messages = []
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.setattr("koyo.system.logger.exception", messages.append)
    reraise_exception_if_debug(ValueError("boom"), "logged")
    assert messages == ["logged"]


def test_reraise_exception_if_debug_raises(monkeypatch):
    monkeypatch.setenv("DEV_MODE", "1")
    with pytest.raises(ValueError, match="boom"):
        reraise_exception_if_debug(ValueError("boom"))


def test_who_called_me(capsys):
    def caller():
        return who_called_me()

    file_name, function_name, line_number = caller()
    assert function_name == "caller"
    assert file_name.endswith("test_system.py")
    assert isinstance(line_number, int)
    assert "Called from file:" in capsys.readouterr().out


def test_linux_sys_name_from_os_release(monkeypatch):
    content = 'NAME="Ubuntu"\nVERSION_ID="22.04"\n'

    def fake_open(*args, **kwargs):
        from io import StringIO

        return StringIO(content)

    monkeypatch.setattr("os.path.exists", lambda path: path == "/etc/os-release")
    monkeypatch.setattr("builtins.open", fake_open)
    assert _linux_sys_name() == "Ubuntu 22.04"


def test_linux_sys_name_lsb_release(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"Description:\tUbuntu 22.04.5 LTS\nRelease:\t22.04\n"),
    )
    assert _linux_sys_name_lsb_release() == "Ubuntu 22.04.5 LTS 22.04"


def test_is_network_linux_matches_longest_mount(monkeypatch):
    mounts = (
        "/dev/disk1 / ext4 rw 0 0\n"
        "server:/share /mnt/share nfs rw 0 0\n"
        "server:/share/sub /mnt/share/sub ext4 rw 0 0\n"
    )

    def fake_open(*args, **kwargs):
        from io import StringIO

        return StringIO(mounts)

    monkeypatch.setattr("builtins.open", fake_open)
    assert _is_network_linux("/mnt/share/file.txt")
    assert not _is_network_linux("/mnt/share/sub/file.txt")


# ---------------------------------------------------------------------------
# is_network_path
# ---------------------------------------------------------------------------


def test_is_network_path_local(tmp_path):
    # A local temp directory is never a network path
    assert not is_network_path(str(tmp_path))
