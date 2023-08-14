"""Override click group to enable ordering."""
import os
import typing as ty
from ast import literal_eval
from pathlib import Path

import click

from koyo.typing import PathLike


_verbosity_options = [
    click.option(
        "--verbose",
        "-v",
        "verbosity",
        default=1,
        count=True,
        help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
        " information.",
    ),
    click.option("--quiet", "-q", "verbosity", flag_value=0, help="Minimal output"),
    click.option("--debug", "verbosity", flag_value=5, help="Maximum output"),
]


def verbosity_options(func):
    """Verbosity options"""
    for option in reversed(_verbosity_options):
        func = option(func)
    return func


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


def error_msg(msg):
    """Display error message."""
    click.echo(f"ERROR: {msg}")


def warning_msg(msg):
    """Display warning message."""
    click.echo(f"WARNING: {msg}")


def info_msg(msg):
    """Display info message."""
    click.echo(f"INFO: {msg}")


def success_msg(msg):
    """Display success message."""
    click.echo(f"SUCCESS: {msg}")

def expand_dirs(input_dir: str) -> ty.Sequence[str]:
    """Expand data directory"""
    if "*" in str(input_dir):
        return glob.glob(input_dir)
    elif isinstance(input_dir, (str, Path)):
        return [str(input_dir)]
    elif isinstance(input_dir, ty.Sequence):
        return [str(path) for path in input_dir]
    return input_dir


# noinspection PyUnusedLocal
def arg_parse_path(ctx, param, value) -> ty.List[Path]:
    """Split arguments."""
    if value is None:
        return []
    res: ty.List[Path] = []
    if isinstance(value, (str, Path)):
        value = [value]
    for path in value:
        path = str(path)
        if path.endswith("/"):
            path = path[:-1]
        res.extend(expand_dirs(path))
    return [Path(path).absolute() for path in res]

# noinspection PyUnusedLocal
def arg_split_str(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [arg.strip() for arg in value.split(",")]
    return args


# noinspection PyUnusedLocal
def arg_split_float(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [float(arg.strip()) for arg in value.split(",")]
    return args


# noinspection PyUnusedLocal
def arg_split_int(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [int(arg.strip()) for arg in value.split(",")]
    return args


# noinspection PyUnusedLocal
def arg_parse_framelist(ctx, param, value: str):
    """Parse framelist."""
    from imimspy.utils.utilities import parse_str_framelist

    if value is None:
        return None
    return parse_str_framelist(value)


# noinspection PyUnusedLocal
def parse_values(ctx, param, value: str):
    """Parse values and return them as list of potential options."""
    from ast import literal_eval

    if isinstance(value, str):
        return [literal_eval(val) for val in value.split(",")]
    return [literal_eval(val) for val in value]


def timed_iterator(
    iterable: ty.Iterable,
    text: str = "Task executed in",
    func: ty.Callable = info_msg,
    silent: bool = False,
    is_filename: bool = False,
):
    """Timed iterable that yields value and prints amount of time spent on said iterable."""
    from koyo.timer import format_human_time_s, measure_time

    n_tasks = len(list(iterable))
    total = 0
    for i, item in enumerate(iterable, start=1):
        with measure_time() as timer:
            yield item
        if not silent:
            # update timing
            execution_time = timer()
            total += execution_time
            avg = total / i
            remaining = avg * (n_tasks - i)
            _text = text if not is_filename else text + f" ({os.path.basename(item)})"
            func(
                f"[{i}/{n_tasks}] {_text} {format_human_time_s(execution_time)}"
                f" [avg={format_human_time_s(avg)}; rem={format_human_time_s(remaining)};"
                f" tot={format_human_time_s(total)}]"
            )


def parse_arg(arg: str, key: str):
    """Parse argument."""
    try:
        if key:
            arg = arg.split(key)[1]
        name, value = arg.split("=")
        # try parsing value - it will fail if string value was provided
        try:
            value = literal_eval(value)
        # try wrapping value, so it can be evaluated as string literal
        except ValueError:
            value = f'"{value}"'
            value = literal_eval(value)
        except SyntaxError:
            pass
        return name, value
    except ValueError:
        raise ValueError(f"Could not parse argument {arg}")


def parse_extra_args(extra_args: ty.Optional[ty.Tuple[str]]):
    """Arguments"""
    kwargs = {}
    if extra_args is None:
        return kwargs
    for arg in extra_args:
        if "=" not in arg:
            info_msg(f"Skipping argument {arg} as it does not contain '='")
            continue
        name, value = parse_arg(arg, "")
        kwargs[name] = value
    return kwargs


def parse_fig_args(extra_args: ty.Optional[ty.Tuple[str]], clean: bool = False):
    """Parse extra parameters."""
    extra_kwargs = {}
    if extra_args is None:
        if clean:
            return extra_kwargs, extra_args
        return extra_kwargs
    extra_args_ = []
    for arg in extra_args:
        if arg.startswith("--fig:"):
            name, value = parse_arg(arg, "--fig:")
            extra_kwargs[name] = value
        elif arg.startswith("--f:"):
            name, value = parse_arg(arg, "--f:")
            extra_kwargs[name] = value
        else:
            extra_args_.append(arg)
    if clean:
        return extra_kwargs, extra_args_
    return extra_kwargs


def parse_env_args(extra_args: ty.Optional[ty.Tuple[str]], clean: bool = False):
    """Parse extra environment variables."""
    env_kwargs = {}
    if extra_args is None:
        if clean:
            return env_kwargs, extra_args
        return env_kwargs
    extra_args_ = []
    for arg in extra_args:
        if arg.startswith("--env:"):
            name, value = parse_arg(arg, "--env:")
            env_kwargs[name] = value
        else:
            extra_args_.append(arg)
    if clean:
        return env_kwargs, extra_args_
    return env_kwargs


def set_env_args(**kwargs):
    """Set environment variables."""
    from loguru import logger

    for name, value in kwargs.items():
        os.environ[name] = str(value)
        logger.trace(f"Set environment variable: {name}={value}")
