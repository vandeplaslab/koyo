"""Multi-core utilities."""

import math
import time
import typing as ty
import warnings

from joblib import Parallel as _Parallel
from loguru import logger
from psutil import cpu_count, virtual_memory
from tqdm import tqdm

from koyo.timer import report_time
from koyo.utilities import running_as_pyinstaller_app


def estimate_cpu_count_from_size(max_obj_size_in_bytes, keep_free_in_bytes=4_000_000):
    """Get number of cores based on the amount of RAM and maximum size of an object.

    Parameters
    ----------
    max_obj_size_in_bytes : int
        maximum size of an object in bytes. If specified in GB, it will be automatically converted to bytes
    keep_free_in_bytes : int, optional
        minimal amount of RAM reserved for the OS and other actions, by default 4000000
        If specified in Gb, it will be automatically converted to bytes

    Returns
    -------
    n_cores : int
        number of cores to be used by any multicore enabled function

    Raises
    ------
    ValueError
        raises ValueError if number of cores is less than 1
    """
    # convert to GB
    if int(math.log10(keep_free_in_bytes)) + 1 <= 2:
        keep_free_in_bytes = keep_free_in_bytes * 1024**3
    if int(math.log10(max_obj_size_in_bytes)) + 1 <= 2:
        max_obj_size_in_bytes = max_obj_size_in_bytes * 1024**3

    # get currently available memory
    available_memory = virtual_memory().total - keep_free_in_bytes
    n_cores = int(round(available_memory / max_obj_size_in_bytes))
    if n_cores < 1:
        raise ValueError(
            "Based on the amount of RAM available on this system, there is not enough memory to perform"
            + " this action."
        )

    if n_cores > get_cpu_count():
        n_cores = get_cpu_count()
    print(
        f"Total RAM: {virtual_memory().total / 1024**3:0f} Gb",
        f"\nReserved memory: {keep_free_in_bytes / 1024**3} Gb",
        f"\nMax. size of object: {max_obj_size_in_bytes / 1024**3:0f} Gb",
        f"\nMax. recommended cores: {n_cores}",
        sep="",
    )
    return n_cores


def get_cpu_count(keep_free: int = 2, max_cpu: int = 24, n_tasks: int = 0) -> int:
    """Get number of cores available on the machine.

    Parameters
    ----------
    keep_free : int, optional
        using all cores can significantly slow the system down, so we allow for some to remain free, by default 2
    max_cpu : int
        maximum number of cores
    n_tasks : int
        number of tasks to be executed

    Returns
    -------
    n_cores : int
        number of cores to be used by any multicore enabled function

    Raises
    ------
    ValueError
        raises ValueError if number of cores is less than 1
    """
    n_cores = cpu_count() - keep_free
    if n_cores > max_cpu:
        n_cores = max_cpu
    if n_tasks > 0 and n_cores > n_tasks:
        n_cores = n_tasks
    if n_cores < 1:
        warnings.warn(
            f"Tried to reserve `{keep_free}` cores free ut failed be cores. Action will use all cores.",
            RuntimeWarning,
            stacklevel=2,
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

    def run(self, func: ty.Callable, args: ty.Iterable, auto_expand: bool = True):
        """Execute."""
        import mpire

        kws = {
            "worker_lifespan": 1,
            "progress_bar": not self.silent,
            "progress_bar_options": {"desc": self.desc, "mininterval": 5},
        }

        res = []
        if auto_expand:
            if args:
                args = list(args)
                if len(args) == 1 or self.n_cores == 1:
                    for arg in tqdm(args, disable=False, desc=self.desc):
                        res.append(func(*arg))
                else:
                    with mpire.WorkerPool(n_jobs=self.n_cores, keep_alive=False) as pool:
                        res = pool.map(func, args, iterable_len=len(args), **kws)
        else:
            with mpire.WorkerPool(n_jobs=self.n_cores, keep_alive=False) as pool:
                res = pool.map(func, args, **kws)
        return res

    def lazy_run(self, func: ty.Callable, args: ty.Iterable, auto_expand: bool = True):
        """Execute."""
        import mpire

        kws = {
            "worker_lifespan": 1,
            "progress_bar": not self.silent,
            "progress_bar_options": {"desc": self.desc, "mininterval": 5},
        }

        res = []
        if auto_expand:
            if args:
                args = list(args)
                if len(args) == 1 or self.n_cores == 1:
                    for arg in tqdm(args, disable=False, desc=self.desc):
                        res.append(func(*arg))
                        yield res[-1]
                else:
                    with mpire.WorkerPool(n_jobs=self.n_cores, keep_alive=True) as pool:
                        for res_ in pool.imap(func, args, iterable_len=len(args), **kws):
                            res.append(res_)
                            yield res_
        else:
            with mpire.WorkerPool(n_jobs=self.n_cores, keep_alive=True) as pool:
                for res_ in pool.imap(func, args, **kws):
                    res.append(res_)
                    yield res_
        return res


class MultiCoreDispatcher:
    """Base dispatcher class."""

    def __init__(self, n_cores: int = -1, chunksize: int = 100, backend: str = "mpire", quiet: bool = False, **kwargs):
        # accessible attributes
        self.n_cores = self.optimize_n_cores(n_cores, array_size=kwargs.get("array_size"))
        self.chunksize = self.optimize_chunksize(chunksize, self.n_cores)
        self.backend = backend if not running_as_pyinstaller_app() else "multiprocessing"

        # private attributes
        self._maxtasksperchild = kwargs.get("maxtasksperchild", 1)
        self.quiet = quiet

        logger.info(f"Using {self.n_cores} cores. Chunksize {self.chunksize}. Backend: {self.backend}")

    @staticmethod
    def optimize_n_cores(n_cores: int, array_size=None):
        """Calculate the optimal number of cores based on some criteria."""
        # check whether we should optimize it in first place
        if n_cores in [-1, 0, 512]:
            if array_size is not None:
                return estimate_cpu_count_from_size(array_size, 30)
            return get_cpu_count(4)
        if not isinstance(n_cores, int):
            raise ValueError("The value of 'n_cores' must be an integer!")
        return n_cores

    @staticmethod
    def optimize_n_threads(n_threads: int, n_cores: int):
        """If the number of cores exceeds 1, it is usually best not to do multi-threading."""
        if n_cores > 1:
            return 1
        if isinstance(n_threads, int):
            return n_threads
        return 1

    @staticmethod
    def optimize_chunksize(chunksize, n_cores):
        """Optimize chunksize."""
        # if chunk size is set to -1, set it to maximise speed by using maximum number of cores
        # if chunksize == -1:
        #     chunksize = np.max([math.ceil(n_extraction_windows / self._n_cores), 100])
        return chunksize

    def generate_iterable(self) -> ty.List:
        """Return iterable which can be consumed by the `generate_args_list` function."""
        return []

    def generate_args_list(self, post_process: bool = False):
        """Generate list of arguments which can be handled by the `dispatch` function."""
        raise NotImplementedError("Method must be implemented by the subclass")

    def dispatch(self, func: ty.Callable, *, pre_process: bool = True, post_process: bool = True):
        """Dispatches the job over multiple cores to speed-up processing.

        Parameters
        ----------
        func : Callable
            callable function which handles the processing
        pre_process : bool
            flag to run pre-processing before starting tasks
        post_process : bool
            flag to run post-processing after completing all tasks

        Notes
        -----
        Arguments have to pickleable
        """
        t_start = time.time()
        if pre_process:
            self.pre_process()

        try:
            args_list = self.generate_args_list(post_process)
        except TypeError:
            args_list = self.generate_args_list()
        if len(args_list) < self.n_cores:
            self.n_cores = len(args_list)
        if args_list:
            logger.info(f"{len(args_list)} tasks in the queue...")
            if self.n_cores == 1:
                self._single_executor(func, args_list)
            elif self.backend == "joblib":
                self._joblib_executor(func, args_list)
            else:
                self._mpire_executor(func, args_list)

        else:
            logger.info("There are no jobs to be done.")

        # post-process
        if post_process and args_list and self.n_cores > 0:
            self.post_process()
        logger.info(f"*** COMPLETED ACTION IN {report_time(t_start)} ***")

    def _single_executor(self, func, args_list):
        """Single-core executor."""
        for args in tqdm(args_list, disable=self.quiet):
            func(*args)

    def _joblib_executor(self, func, args_list):
        """Joblib executor."""
        from joblib import delayed

        _args_list = []
        for arg in args_list:
            _args_list.append(delayed(func)(*arg))
        del args_list
        Parallel(
            n_jobs=self.n_cores,
            verbose=50,
            prefer="processes",
        )(_args_list)

    def _multiprocessing_executor(self, func, args_list):
        """Multiprocessing executor."""
        import multiprocessing

        if args_list:
            # create processor pool
            with multiprocessing.Pool(processes=self.n_cores, maxtasksperchild=self._maxtasksperchild) as pool:
                # map function with tuple of arguments
                pool.starmap(func, args_list)
            # close pool and join it until all jobs have finished
            pool.close()
            pool.join()

    def _mpire_executor(self, func, args_list):
        """MPIRE executor."""
        import mpire

        if args_list:
            # create processor pool
            with mpire.WorkerPool(n_jobs=self.n_cores) as pool:
                pool.map_unordered(
                    func,
                    args_list,
                    iterable_len=len(args_list),
                    worker_lifespan=self._maxtasksperchild,
                    progress_bar=not self.quiet,
                )

    def pre_process(self):
        """Pre-process."""

    def post_process(self):
        """Post-process."""


class Parallel(_Parallel):
    """Parallel."""

    def __init__(self, *args, **kwargs):
        if "backend" in kwargs and kwargs["backed"] == "loky" and running_as_pyinstaller_app():
            kwargs["backend"] = "multiprocessing"
        if "backend" not in kwargs and running_as_pyinstaller_app():
            kwargs["backend"] = "multiprocessing"
        logger.info(f"Executing using backend - {kwargs.get('backend', 'loky')}")
        super().__init__(*args, **kwargs)


class ProgressParallel(Parallel):
    """joblib's Parallel wrapper that includes tqdm progress indicator."""

    def __init__(self, disable=False, total=None, desc: str = "", *args, **kwargs):
        self._disable = disable
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=self._disable, total=self._total, desc=self._desc) as self.progress_bar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        """Print progress."""
        if self._total is None:
            self.progress_bar.total = self.n_dispatched_tasks
        self.progress_bar.n = self.n_completed_tasks
        self.progress_bar.refresh()
