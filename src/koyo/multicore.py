"""Multi-core utilities."""
import typing as ty
import warnings

from tqdm.auto import tqdm


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


def is_daemon() -> bool:
    """Check if current process is daemonic."""
    from multiprocessing import current_process

    return current_process().daemon


class MultiCoreExecutor:
    """MPIRE-based executor."""

    def __init__(self, n_cores: int, silent: bool = False, desc=""):
        self.n_cores = n_cores
        self.silent = silent
        self.desc = desc

    def run(self, func: ty.Callable, args: ty.Iterable):
        """Execute."""
        import mpire

        if args:
            args = list(args)

            if len(args) == 1 or self.n_cores == 1:
                res = []
                for arg in tqdm(args, disable=False, desc=self.desc):
                    res.append(func(*arg))
            else:
                with mpire.WorkerPool(n_jobs=self.n_cores, keep_alive=False) as pool:
                    res = pool.map(
                        func,
                        args,
                        iterable_len=len(args),
                        worker_lifespan=1,
                        progress_bar=not self.silent,
                    )
            return res
