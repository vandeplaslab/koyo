import os
import sys
from contextlib import suppress

DEFAULT_HOOK = None


def debugger_hook(type, value, tb):
    """Drop into python debugger on uncaught exception."""
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode, or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        import traceback

        # we are NOT in interactive mode, print the exception...
        with suppress(Exception):
            traceback.print_exception(type, value, tb)
            pdb.post_mortem(tb)


def install_debugger_hook():
    """Activate the debugger hook."""
    global DEFAULT_HOOK
    os.environ["KOYO_DEV_MODE"] = "1"

    if DEFAULT_HOOK is None:
        DEFAULT_HOOK = sys.excepthook
    sys.excepthook = debugger_hook


def uninstall_debugger_hook():
    """Deactivate the debugger hook."""
    global DEFAULT_HOOK
    os.environ["KOYO_DEV_MODE"] = "0"

    if DEFAULT_HOOK is not None:
        sys.excepthook = DEFAULT_HOOK
        DEFAULT_HOOK = None
