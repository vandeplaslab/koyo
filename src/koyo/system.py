"""System utilities."""
import sys
import warnings

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"
IS_MAC = sys.platform == "darwin"


def get_cpu_count(keep_free: int = 2, max_cpu: int = 24):
    """Get number of cores available on the machine.

    Parameters
    ----------
    keep_free : int, optional
        using all cores can significantly slow the system down, so we allow for some to remain free, by default 2
    max_cpu : int
        maximum number of cores

    Returns
    -------
    n_cores : int
        number of cores to be used by any multicore enabled function

    Raises
    ------
    ValueError
        raises ValueError if number of cores is less than 1
    """
    from psutil import cpu_count

    n_cores = cpu_count() - keep_free
    if n_cores > max_cpu:
        n_cores = max_cpu
    if n_cores < 1:
        warnings.warn(
            f"Tried to reserve `{keep_free}` cores free ut failed be cores. Action will use all cores.", RuntimeWarning
        )
        n_cores = cpu_count()
    return n_cores
