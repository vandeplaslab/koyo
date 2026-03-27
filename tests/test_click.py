"""Test that click produces the expected output."""

from pathlib import Path

import click
import pytest
from koyo.click import (
    BrokenCommand,
    Parameter,
    arg_parse_env,
    arg_parse_int_with_range_multi,
    arg_parse_int_with_range_multi_merge,
    arg_parse_path,
    arg_split_float,
    arg_split_int,
    arg_split_str,
    cli_parse_paths_sort_auto_glob,
    expand_data_dirs,
    expand_dirs,
    filter_kwargs,
    format_value,
    get_args_from_option,
    parse_arg,
    parse_args_with_keys,
    parse_env_args,
    parse_extra_args,
    parse_int_with_range,
    parse_paths,
    parse_values,
    repr_filelist,
    repr_iterable,
    select_from_list,
    set_env_args,
    set_env_args_from_tuples,
    timed_iterator,
    with_plugins,
)


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("n=5", ("n", 5)),
        ("n=5.0", ("n", 5.0)),
        ("n=[10,20,30]", ("n", [10, 20, 30])),
        ("n=value=with=equal=sign", ("n", "value=with=equal=sign")),
    ],
)
def test_parse_arg_no_key(arg, expected):
    res_key, res_value = parse_arg(arg, "")
    assert res_key == expected[0], f"Expected {expected[0]}, got {res_key}"
    assert res_value == expected[1], f"Expected {expected[1]}, got {res_value}"


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("--un:n=5", ("n", 5)),
        ("--un:n=5.0", ("n", 5.0)),
        ("--un:n=[10,20,30]", ("n", [10, 20, 30])),
    ],
)
def test_parse_arg_with_key(arg, expected):
    res_key, res_value = parse_arg(arg, "--un:")
    assert res_key == expected[0], f"Expected {expected[0]}, got {res_key}"
    assert res_value == expected[1], f"Expected {expected[1]}, got {res_value}"


@pytest.mark.parametrize(
    "int_range, expected",
    [
        ("1", [1]),
        ("1,2,3", [1, 2, 3]),
        ("1:3", [1, 2, 3]),
        ("1-3", [1, 2, 3]),
        ("0:10:2", [0, 2, 4, 6, 8, 10]),
        ("0-10", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ("1,3,5:7", [1, 3, 5, 6, 7]),
    ],
)
def test_parse_int_with_range(int_range, expected):
    assert parse_int_with_range(int_range) == expected


def test_parse_int_with_range_none():
    assert parse_int_with_range(None) == []


def test_parse_paths_basic(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    result = parse_paths([str(d1), str(d2)])
    assert str(d1) in result
    assert str(d2) in result


def test_parse_paths_glob(tmp_path):
    for name in ("x", "y", "z"):
        (tmp_path / name).mkdir()
    result = parse_paths([str(tmp_path / "*")])
    assert len(result) == 3


def test_parse_paths_trailing_slash(tmp_path):
    d = tmp_path / "mydir"
    d.mkdir()
    result = parse_paths([str(d) + "/"])
    assert result[0] == str(d)


def test_parse_paths_sort(tmp_path):
    for name in ("c", "a", "b"):
        (tmp_path / name).mkdir()
    result = parse_paths([str(tmp_path / "*")], sort=True)
    assert result == sorted(result)


def test_parse_paths_absolute(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    result = parse_paths([str(child)], absolute=True)
    assert result == [str(child.resolve())]


def test_expand_data_dirs_without_glob():
    assert expand_data_dirs("plain/path") == ["plain/path"]


def test_cli_parse_paths_sort_auto_glob_on_directory(tmp_path):
    for name in ("10", "2"):
        (tmp_path / name).mkdir()
    result = cli_parse_paths_sort_auto_glob(None, None, [str(tmp_path)])
    assert result == [str(tmp_path / "2"), str(tmp_path / "10")]


def test_parse_extra_args_basic():
    result = parse_extra_args(("a=1", "b=hello"))
    assert result["a"] == 1
    assert result["b"] == "hello"


def test_parse_extra_args_repeated_key():
    result = parse_extra_args(("a=1", "a=2"))
    assert result["a"] == [1, 2]


def test_parse_extra_args_none():
    assert parse_extra_args(None) == {}


def test_parse_extra_args_skips_no_equals():
    result = parse_extra_args(("no_equals",))
    assert result == {}


def test_parse_args_with_keys_clean():
    parsed, remaining = parse_args_with_keys(("--fig:dpi=150", "keep=1"), ("--fig:",), clean=True)
    assert parsed == {"dpi": 150}
    assert remaining == ["keep=1"]


def test_parse_env_args_clean():
    parsed, remaining = parse_env_args(("--env:DEBUG=1", "keep=1"), clean=True)
    assert parsed == {"DEBUG": 1}
    assert remaining == ["keep=1"]


def test_repr_filelist(tmp_path):
    paths = [tmp_path / "foo.txt", tmp_path / "bar.txt"]
    result = repr_filelist(paths)
    assert "foo.txt" in result
    assert "bar.txt" in result
    assert "; " in result


def test_repr_iterable_short():
    items = list(range(5))
    assert repr_iterable(items) == items


def test_repr_iterable_long():
    items = list(range(11))
    result = repr_iterable(items)
    assert "11" in str(result)
    assert "items" in str(result)


def test_repr_iterable_none():
    assert repr_iterable(None) is None


def test_format_value_dict_and_empty_list():
    dict_rows = format_value("desc", "--arg", {"a": 1, "b": 2})
    assert dict_rows[0] == ("desc", "--arg", "a=1")
    assert dict_rows[1] == ("", "", "b=2")
    assert format_value("desc", "--arg", []) == [("desc", "--arg", "<no value>")]


def test_get_args_from_option():
    option = click.option("--name", "-n")
    assert get_args_from_option(option) == "-n/--name"


def test_parameter_to_list_with_option():
    param = Parameter("Name", click.option("--name", "-n"), "value")
    assert param.to_list() == [("Name", "-n/--name", "value")]


def test_expand_dirs_sequence():
    result = expand_dirs([Path("a"), Path("b")])
    assert result == ["a", "b"]


def test_arg_parse_path_absolute(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    result = arg_parse_path(None, None, (f"'{child}'",))
    assert result == [child.resolve()]


def test_arg_split_helpers():
    assert arg_split_str(None, None, "a, b") == ["a", "b"]
    assert arg_split_float(None, None, "1, 2.5") == [1.0, 2.5]
    assert arg_split_int(None, None, "1, 2") == [1, 2]


def test_arg_parse_env():
    assert arg_parse_env(None, None, ("A=1", "B=two")) == [["A", "1"], ["B", "two"]]


def test_parse_int_with_range_multi_helpers():
    assert arg_parse_int_with_range_multi(None, None, ("1:2", "4")) == [[1, 2], [4]]
    assert arg_parse_int_with_range_multi_merge(None, None, ("1:2", "4")) == [1, 2, 4]


def test_parse_values_handles_string_and_iterable():
    assert parse_values(None, None, "1,'a'") == [1, "a"]
    assert parse_values(None, None, ("1", "'a'")) == [1, "a"]


def test_timed_iterator_calls_logger():
    seen = []
    messages = []
    for item in timed_iterator([1, 2], text="Processed", func=messages.append):
        seen.append(item)
    assert seen == [1, 2]
    assert len(messages) == 2
    assert all("Processed" in msg for msg in messages)


def test_set_env_args(monkeypatch):
    monkeypatch.delenv("KOYO_TMP_A", raising=False)
    set_env_args(KOYO_TMP_A=123)
    assert "KOYO_TMP_A" in __import__("os").environ
    assert __import__("os").environ["KOYO_TMP_A"] == "123"


def test_set_env_args_from_tuples(monkeypatch):
    monkeypatch.delenv("KOYO_TMP_B", raising=False)
    set_env_args_from_tuples((("KOYO_TMP_B", "456"), (None, "skip")))
    assert __import__("os").environ["KOYO_TMP_B"] == "456"


def test_select_from_list_auto_modes():
    items = ["a", "b", "c"]
    assert select_from_list(items, auto_select="newest") == 2
    assert select_from_list(items, auto_select="oldest") == 0
    assert select_from_list([], default=7) == 7


def test_select_from_list_prompt(monkeypatch):
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: 5)
    assert select_from_list(["a", "b"], auto_select="off", default=1) == 5


def test_with_plugins_registers_broken_command():
    class BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    group = click.Group()
    wrapped = with_plugins([BrokenEntryPoint()])(group)
    assert isinstance(wrapped.commands["broken"], BrokenCommand)


def test_with_plugins_rejects_non_group():
    with pytest.raises(TypeError):
        with_plugins([])(object())


def test_broken_command_parse_args_passthrough():
    command = BrokenCommand("broken")
    assert command.parse_args(None, ["--x"]) == ["--x"]


def test_broken_command_invoke_exits(capsys):
    command = BrokenCommand("broken")

    class Ctx:
        color = False

        @staticmethod
        def exit(code):
            raise SystemExit(code)

    with pytest.raises(SystemExit) as exc:
        command.invoke(Ctx())
    assert exc.value.code == 1
    assert "Warning: entry point could not be loaded" in capsys.readouterr().out


def test_filter_kwargs():
    result = filter_kwargs("a", "b", a=1, b=2, c=3)
    assert result == {"a": 1, "b": 2}
    assert "c" not in result
