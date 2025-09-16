"""Helper functions for PyInstaller hooks."""
from __future__ import annotations

from collections.abc import Iterator
import types

import importlib.util
import sys
from pathlib import Path

def load_module_from_path(path: str, module_name: str | None = None) -> types.ModuleType:
    """
    Load a .py file from an arbitrary path and return the module object.

    - path: full path to the .py file (e.g. "PATH/TO/SCRIPT/hook-ionglow.py")
    - module_name: optional name to register in sys.modules (e.g. "hook_ionglow").
                   If None, a unique name based on file name will be used.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"{path} not found or not a .py file")

    # default module name: make filename a valid identifier
    if module_name is None:
        # convert "hook-ionglow.py" -> "hook_ionglow"
        module_name = path.stem.replace("-", "_")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # register so imports inside the loaded module can use sys.modules if needed
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # runs the module code
    return module

def _iter_all_modules(
    package: str | types.ModuleType,
    prefix: str = "",
) -> Iterator[str]:
    """Iterate over the names of all modules that can be found in the given
    package, recursively.

        >>> import _pytest
        >>> list(_iter_all_modules(_pytest))
        ['_pytest._argcomplete', '_pytest._code.code', ...]
    """
    import os
    import pkgutil

    if isinstance(package, str):
        path = package
    else:
        # Type ignored because typeshed doesn't define ModuleType.__path__
        # (only defined on packages).
        package_path = package.__path__
        path, prefix = package_path[0], package.__name__ + "."
    for _, name, is_package in pkgutil.iter_modules([path]):
        if is_package:
            for m in _iter_all_modules(os.path.join(path, name), prefix=name + "."):
                yield prefix + m
        else:
            yield prefix + name