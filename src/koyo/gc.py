"""Garbage collection utilities."""

import gc
import time
import weakref

from loguru import logger

_uncollectable_history = []  # store snapshots for later analysis


def _gc_debug_callback(phase, info):
    # We only care after GC finishes
    if phase != "stop":
        return

    gen = info["generation"]
    n_unc = info.get("uncollectable", 0)

    if n_unc > 0:
        ts = time.time()
        print(f"[gc debug] gen {gen}: {n_unc} uncollectable objects after collection at {ts}")

        # Snapshot what's currently uncollectable
        snapshot = []
        for obj in gc.garbage:
            snapshot.append(
                {
                    "type": type(obj),
                    "repr": repr(obj)[:300],
                    "id": id(obj),
                    # store weakref so we don't keep it alive ourselves
                    "weak": weakref.ref(obj),
                }
            )

        _uncollectable_history.append(
            {
                "time": ts,
                "generation": gen,
                "snapshot": snapshot,
            }
        )

        # Optional: print a little detail to console for quick triage
        for entry in snapshot[:5]:  # don't spam if it's huge
            logger.warning("   ->", entry["type"], entry["repr"])


def install_gc_debugging() -> None:
    """Install GC debugging callback to track uncollectable objects."""
    # Register the callback (only once!)
    if _gc_debug_callback not in gc.callbacks:
        gc.callbacks.append(_gc_debug_callback)

    # Make sure gc will actually put uncollectables in gc.garbage
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
