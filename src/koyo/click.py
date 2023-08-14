"""Override click group to enable ordering."""
import typing as ty
from pathlib import Path

import click

from koyo.typing import PathLike


def repr_filelist(filelist: ty.Sequence[PathLike]) -> str:
    """Pretty iterable..."""
    res = [Path(file).name for file in filelist]
    return "; ".join(res)


def repr_iterable(iterable: ty.Sequence[ty.Any]):
    """Pretty iterable..."""
    if iterable is None:
        return iterable
    if isinstance(iterable, ty.Sequence):
        n = len(iterable)
        if n > 10:
            return f"{n} items..."
        return iterable


def append(table: ty.List[ty.Tuple[str, str, str]], name="", param="", value=""):
    """Append to table."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and len(value) > 80:
        value = "..." + str(value)[-80:]
    elif isinstance(value, (list, tuple)):
        value = repr_iterable(value)
    table.append((name, param, value))


def format_value(description: str, args: str, value: ty.Any) -> ty.List[ty.Tuple[str, str, str]]:
    """Format value."""
    res = []
    # lists should be printed as multiple rows
    if isinstance(value, ty.List):
        if len(value) > 0:
            append(res, description, args, value[0])
            for v in value[1::]:
                append(res, "", "", v)
        else:
            append(res, description, args, "<no value>")
    # dicts should be printed as multiple rows
    elif isinstance(value, ty.Dict):
        if len(value) > 0:
            for i, (k, v) in enumerate(value.items()):
                if i == 0:
                    append(res, description, args, f"{k}={v}")
                else:
                    append(res, "", "", f"{k}={v}")
        else:
            append(res, description, args, "<no value>")
    else:
        append(res, description, args, value)
    return res


class Parameter:
    """Parameter object."""

    __slots__ = ["description", "args", "value"]

    def __init__(self, description: str, args: str, value: ty.Optional[ty.Any] = None):
        self.description = description
        self.args = args
        self.value = value

    def with_value(
        self, value: ty.Any, description: ty.Optional[str] = None, args: ty.Optional[str] = None
    ) -> "Parameter":
        """Set the value of the parameter and return self."""
        return Parameter(description or self.description, args or self.args, value)

    def to_list(self) -> ty.List[ty.Tuple[str, str, str]]:
        """Return formatted version of the value."""
        if self.value:
            return format_value(self.description, self.args, self.value)
        return [(self.description, self.args, "")]


class OrderedGroup(click.Group):
    """Override click group to enable ordering.

    See https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    """

    def __init__(self, *args, **kwargs):
        self.priorities = {}
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit
    def get_help(self, ctx):
        """Get help."""
        self.list_commands = self.list_commands_for_help
        return super().get_help(ctx)

    def list_commands_for_help(self, ctx):
        """Reorder the list of commands when listing the help."""
        commands = super().list_commands(ctx)
        return (c[1] for c in sorted((self.priorities.get(command, 1), command) for command in commands))

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        priority = kwargs.pop("priority", 1)
        priorities = self.priorities

        def decorator(f):
            cmd = super(OrderedGroup, self).command(*args, **kwargs)(f)
            priorities[cmd.name] = priority
            return cmd

        return decorator


def print_parameters(*parameters: Parameter):
    """Print parameters as table."""
    from tabulate import tabulate

    table = []
    for param in parameters:
        table.extend(param.to_list())
    print(tabulate(table, headers=["Name", "Parameter", "Value"], tablefmt="github"))
