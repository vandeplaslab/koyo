import inspect
import sys


def open_file_audit_hook(name, *args):
    """Open file hook."""
    if name == "open":
        print(name, *args, "was called:")
        caller = inspect.currentframe()
        while caller := caller.f_back:
            print(f"\tFunction {caller.f_code.co_name} " f"in {caller.f_code.co_filename}:" f"{caller.f_lineno}")


def install_open_file_audit_hook():
    """Add open file audit hook."""
    sys.addaudithook(open_file_audit_hook)
