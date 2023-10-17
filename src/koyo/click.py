"""Override click group to enable ordering."""
import glob
import os
import sys
import traceback
import typing as ty
from ast import literal_eval
from pathlib import Path

import click
from loguru import logger

from koyo.typing import PathLike


def with_plugins(plugins, **kwargs):
    """
    A decorator to register external CLI commands to an instance of
    `click.Group()`.

    Parameters
    ----------
    plugins : iter
        An iterable producing one `pkg_resources.EntryPoint()` per iteration.
    kwargs : **kwargs, optional
        Additional keyword arguments for instantiating `click.Group()`.

    Returns
    -------
    click.Group()
    """

    def decorator(group):
        if not isinstance(group, click.Group):
            raise TypeError("Plugins can only be attached to an instance of click.Group()")

        for entry_point in plugins or ():
            try:
                group.add_command(entry_point.load(), **kwargs)
            except Exception:
                # Catch this so a busted plugin doesn't take down the CLI.
                # Handled by registering a dummy command that does nothing
                # other than explain the error.
                group.add_command(BrokenCommand(entry_point.name))

        return group

    return decorator


class BrokenCommand(click.Command):
    """
    Rather than completely crash the CLI when a broken plugin is loaded, this
    class provides a modified help message informing the user that the plugin is
    broken and they should contact the owner.  If the user executes the plugin
    or specifies `--help` a traceback is reported showing the exception the
    plugin loader encountered.
    """

    def __init__(self, name):
        """Define the special help messages after instantiating a `click.Command()`."""
        click.Command.__init__(self, name)

        util_name = os.path.basename(sys.argv and sys.argv[0] or __file__)

        if os.environ.get("CLICK_PLUGINS_HONESTLY"):  # pragma no cover
            icon = "\U0001F4A9"
        else:
            icon = "\u2020"

        self.help = (
            "\nWarning: entry point could not be loaded. Contact "
            "its author for help.\n\n\b\n" + traceback.format_exc()
        )
        self.short_help = icon + f" Warning: could not load plugin. See `{util_name} {self.name} --help`."

    def invoke(self, ctx):
        """Print the traceback instead of doing nothing."""
        click.echo(self.help, color=ctx.color)
        ctx.exit(1)

    def parse_args(self, ctx, args):
        return args


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
    """Verbosity options."""
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
    if isinstance(value, str) and len(value) > 120:
        value = "..." + str(value)[-120:]
    elif isinstance(value, (list, tuple)):
        value = repr_iterable(value)
    table.append((name, param, value))


def format_value(description: str, args: str, value: ty.Any) -> ty.List[ty.Tuple[str, str, str]]:
    """Format value."""
    res = []
    # lists should be printed as multiple rows
    if isinstance(value, (ty.List, ty.Tuple)):
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


def get_args_from_option(option: ty.Callable):
    """Return argument information from option."""
    if not hasattr(option, "__closure__"):
        raise ValueError(f"Option {option} does not have closure.")
    closure = option.__closure__
    for cell in closure:
        if isinstance(cell.cell_contents, tuple):
            break
    ret = ""
    for value in cell.cell_contents:
        if isinstance(value, str) and value.startswith("-"):
            ret += value
    return ret


class Parameter:
    """Parameter object."""

    __slots__ = ["description", "args", "value"]

    def __init__(self, description: str, args: ty.Union[str, ty.Callable], value: ty.Optional[ty.Any] = None):
        self.description = description
        self.args = args if isinstance(args, str) else get_args_from_option(args)
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


class Parameters:
    """Parameters class."""

    @classmethod
    def p(cls, description: str, args: str, value: ty.Any):
        """Return Parameter instance."""
        return Parameter(description, args, value)


def print_parameters(*parameters: Parameter):
    """Print parameters as table."""
    from tabulate import tabulate

    table = []
    for param in parameters:
        table.extend(param.to_list())
    print(tabulate(table, headers=["Name", "Parameter", "Value"], tablefmt="github"))


def error_msg(msg: str):
    """Display error message."""
    logger.error(msg)


def warning_msg(msg: str):
    """Display warning message."""
    logger.warning(msg)


def info_msg(msg: str):
    """Display info message."""
    logger.info(msg)


def success_msg(msg: str):
    """Display success message."""
    logger.info(msg)


def expand_dirs(input_dir: str) -> ty.Sequence[str]:
    """Expand data directory."""
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
    res: ty.List[str] = []
    if isinstance(value, (str, Path)):
        value = [value]
    for path in value:
        path = str(path).replace("'", "").replace('"', "")
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


def parse_str_framelist(framelist: str) -> ty.List[int]:
    """Parse list provided in string format."""
    if framelist is None:
        return []
    framelist = framelist.replace(" ", "")
    _framelist = framelist.split(",")

    out_framelist = []
    for frame in _framelist:
        _frame = frame.split(":")
        if len(_frame) == 1:
            out_framelist.append(int(_frame[0]))
        elif len(_frame) in [2, 3]:
            start, end, step = int(_frame[0]), int(_frame[1]), 1
            if len(_frame) == 3:
                step = int(_frame[2])
            out_framelist.extend(list(range(start, end, step)))
        else:
            print(f"Skipped {frame} - could not parse it")
    return out_framelist


# noinspection PyUnusedLocal
def arg_parse_framelist(ctx, param, value: str):
    """Parse framelist."""
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

    total = 0
    iterable = list(iterable)
    n_tasks = len(iterable)
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
            avg = f"avg={format_human_time_s(avg)}; "
            rem = f"rem={format_human_time_s(remaining)}; " if remaining > 0 else ""
            tot = f"tot={format_human_time_s(total)}"
            func(f"[{i}/{n_tasks}] {_text} {format_human_time_s(execution_time)} [{avg}{rem}{tot}]")


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


def exit_with_error(skip_error: bool = False):
    """Skip error or exit with error."""
    import sys

    if skip_error:
        return
    return sys.exit(1)


def select_from_list(
    item_list: ty.List[ty.Any],
    text: str = "Please select one of the options from the list",
    auto_select: str = "off",
    default: int = -1,
) -> int:
    """Select file from the list. The list is assumed to be sorted in descending order of creation time meaning that
    oldest files should be first and newest should be last.
    """
    if item_list:
        if len(item_list) == 1:
            return 0
        elif auto_select.lower() == "off":
            choice = click.prompt(text, type=click.INT, default=default)
            return choice
        elif auto_select.lower() == "newest":
            return len(item_list) - 1
        elif auto_select.lower() == "oldest":
            return 0
    return default
