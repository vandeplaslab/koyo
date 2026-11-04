"""Tests for koyo.multicore."""

import subprocess
import sys

import pytest
from koyo.multicore import HAS_JOBLIB, MultiCoreDispatcher, Parallel


def test_import_without_joblib_does_not_fail():
    code = "import sys; sys.modules['joblib'] = None; import koyo.multicore as m; assert not m.HAS_JOBLIB"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(HAS_JOBLIB, reason="joblib is installed")
def test_parallel_without_joblib_raises_import_error():
    with pytest.raises(ImportError, match="joblib"):
        Parallel(n_jobs=1)


def _noop(*args):
    return None


class _Dispatcher(MultiCoreDispatcher):
    def generate_args_list(self, post_process: bool = False) -> list:
        return [(i,) for i in range(2)]


def test_dispatch_restores_n_cores():
    dispatcher = _Dispatcher(n_cores=8, quiet=True)
    dispatcher.n_cores = 8
    dispatcher.dispatch(_noop, pre_process=False, post_process=False)
    assert dispatcher.n_cores == 8
