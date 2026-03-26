"""Tests for koyo.path."""

import os
import sys
from pathlib import Path

import pytest

from koyo.path import (
    apply_drive_mapping,
    copy_file,
    create_directory,
    create_link_txtlnk,
    create_symlink,
    dir_iter,
    empty_directory,
    get_copy_path,
    get_subdirectories,
    mglob,
    remove_file,
    resolve_links,
)


# ---------------------------------------------------------------------------
# copy_file
# ---------------------------------------------------------------------------


def test_copy_file_content(tmp_path, monkeypatch):
    # Remove os.sendfile so the shutil fallback is exercised
    monkeypatch.delattr(os, "sendfile", raising=False)
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello world")
    copy_file(str(src), str(dst))
    assert dst.exists()
    assert dst.read_text() == "hello world"


def test_copy_file_binary(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "sendfile", raising=False)
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    data = bytes(range(256))
    src.write_bytes(data)
    copy_file(str(src), str(dst))
    assert dst.read_bytes() == data


# ---------------------------------------------------------------------------
# mglob
# ---------------------------------------------------------------------------


def test_mglob_finds_matching_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.csv").write_text("c")
    results = list(mglob(tmp_path, "*.txt"))
    assert len(results) == 2


def test_mglob_nonexistent_path(tmp_path):
    results = list(mglob(tmp_path / "nonexistent", "*.txt"))
    assert results == []


def test_mglob_multiple_patterns(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.csv").write_text("b")
    results = list(mglob(tmp_path, "*.txt", "*.csv"))
    assert len(results) == 2


def test_mglob_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("x")
    results = list(mglob(tmp_path, "*.txt", recursive=True))
    assert len(results) == 1


# ---------------------------------------------------------------------------
# dir_iter
# ---------------------------------------------------------------------------


def test_dir_iter_yields_only_directories(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    dirs = list(dir_iter(tmp_path))
    assert len(dirs) == 1
    assert dirs[0].is_dir()


# ---------------------------------------------------------------------------
# get_copy_path
# ---------------------------------------------------------------------------


def test_get_copy_path_nonexistent(tmp_path):
    p = tmp_path / "new_file.txt"
    result = get_copy_path(p)
    assert result == p


def test_get_copy_path_existing(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("x")
    result = get_copy_path(p)
    assert result != p
    assert "copy" in result.name


def test_get_copy_path_multiple_existing(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("x")
    copy1 = tmp_path / "file_copy1.txt"
    copy1.write_text("x")
    result = get_copy_path(p)
    assert result == tmp_path / "file_copy2.txt"


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


def test_remove_file_existing(tmp_path):
    f = tmp_path / "to_delete.txt"
    f.write_text("x")
    remove_file(f)
    assert not f.exists()


def test_remove_file_nonexistent(tmp_path):
    # Should not raise
    remove_file(tmp_path / "ghost.txt")


# ---------------------------------------------------------------------------
# empty_directory
# ---------------------------------------------------------------------------


def test_empty_directory_removes_files(tmp_path):
    sub = tmp_path / "mydir"
    sub.mkdir()
    (sub / "file.txt").write_text("x")
    empty_directory(sub, remove_parent=False)
    assert not (sub / "file.txt").exists()
    assert sub.exists()  # parent kept


def test_empty_directory_removes_parent(tmp_path):
    sub = tmp_path / "mydir"
    sub.mkdir()
    (sub / "file.txt").write_text("x")
    empty_directory(sub, remove_parent=True)
    assert not sub.exists()


def test_empty_directory_nonexistent(tmp_path):
    # Should not raise
    empty_directory(tmp_path / "ghost", remove_parent=True)


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------


def test_create_directory_creates_nested(tmp_path):
    new_dir = tmp_path / "a" / "b" / "c"
    result = create_directory(str(new_dir))
    assert new_dir.exists()
    assert result == new_dir


def test_create_directory_already_exists(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    result = create_directory(str(existing))
    assert result is None


# ---------------------------------------------------------------------------
# get_subdirectories
# ---------------------------------------------------------------------------


def test_get_subdirectories_lists_dirs(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "file.txt").write_text("x")
    subs = get_subdirectories(tmp_path)
    assert len(subs) == 2
    assert all(s.is_dir() for s in subs)


def test_get_subdirectories_nonexistent(tmp_path):
    result = get_subdirectories(tmp_path / "ghost")
    assert result == []


# ---------------------------------------------------------------------------
# apply_drive_mapping
# ---------------------------------------------------------------------------


def test_apply_drive_mapping_replaces():
    result = apply_drive_mapping(Path("/old/path/file.txt"), (("/old", "/new"),))
    assert result == "/new/path/file.txt"


def test_apply_drive_mapping_no_match():
    result = apply_drive_mapping(Path("/some/path"), (("/other", "/alt"),))
    assert result == "/some/path"


def test_apply_drive_mapping_empty():
    result = apply_drive_mapping(Path("/some/path"), ())
    assert result == "/some/path"


# ---------------------------------------------------------------------------
# create_symlink (POSIX only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_create_symlink_creates_link(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("content")
    out_dir = tmp_path / "links"
    link = create_symlink(target, out_dir)
    assert link.exists() or link.is_symlink()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_create_symlink_overwrites_existing(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("content")
    out_dir = tmp_path / "links"
    link1 = create_symlink(target, out_dir)
    # Second call should overwrite without error
    link2 = create_symlink(target, out_dir)
    assert link1 == link2


# ---------------------------------------------------------------------------
# create_link_txtlnk
# ---------------------------------------------------------------------------


def test_create_link_txtlnk_writes_target(tmp_path):
    target = tmp_path / "target.h5"
    target.write_text("data")
    out_dir = tmp_path / "links"
    link = create_link_txtlnk(target, out_dir)
    assert link.suffix == ".txtlnk"
    assert link.read_text().strip() == str(target)


def test_create_link_txtlnk_with_prefix(tmp_path):
    target = tmp_path / "file.h5"
    target.write_text("data")
    out_dir = tmp_path / "links"
    link = create_link_txtlnk(target, out_dir, prefix="exp01")
    assert "exp01" in link.name


# ---------------------------------------------------------------------------
# resolve_links
# ---------------------------------------------------------------------------


def test_resolve_links_finds_existing_targets(tmp_path):
    target = tmp_path / "data.h5"
    target.write_text("data")
    link_file = tmp_path / "data.h5.txtlnk"
    link_file.write_text(str(target))
    result = resolve_links(tmp_path, (".h5",))
    assert target in result


def test_resolve_links_skips_missing_targets(tmp_path):
    link_file = tmp_path / "missing.h5.txtlnk"
    link_file.write_text(str(tmp_path / "ghost.h5"))
    result = resolve_links(tmp_path, (".h5",))
    assert result == []
