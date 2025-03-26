"""Fault handler."""

import atexit
from contextlib import suppress
from pathlib import Path

from loguru import logger

from koyo.typing import PathLike


def submit_sentry_attachment(message: str, path: PathLike) -> None:
    """Submit attachment to Sentry."""
    try:
        from sentry_sdk import configure_scope, get_current_scope

        scope = get_current_scope()
        scope.add_attachment(path=str(path))
        scope.capture_message(message)
    except ImportError:
        logger.exception("Failed to submit attachment to Sentry. Please report this issue to the developers.")
    except Exception:
        logger.exception("Failed to submit attachment to Sentry.")


def install_segfault_handler(output_dir: PathLike, filename: str = "segfault.log") -> None:
    """Install segfault handler."""
    import faulthandler

    segfault_path = Path(output_dir) / filename
    segfault_file = open(segfault_path, "w+")
    faulthandler.enable(segfault_file, all_threads=True)
    atexit.register(segfault_file.close)
    logger.trace(f"Enabled fault handler - logging to '{segfault_path}'")


# noinspection PyBroadException
def maybe_submit_segfault(output_dir: PathLike, filename: str = "segfault.log") -> None:
    """Submit segfault to Sentry if there is an existing segfault file."""
    from datetime import datetime

    segfault_path = Path(output_dir) / filename
    if not segfault_path.exists():
        return

    try:
        # read segfault data
        segfault_text = segfault_path.read_text()
        if not segfault_text:
            return

        logger.error("There was a segmentation fault previously - submitting to Sentry if it's available.")

        # create backup of the segfault file
        try:
            segfault_backup_path = Path(output_dir) / f"segfault_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            segfault_path.rename(segfault_backup_path)

            # submit to Sentry
            submit_sentry_attachment("Segfault detected", segfault_backup_path)
        except PermissionError:
            with suppress(Exception):
                submit_sentry_attachment("Segfault detected", segfault_path)
            logger.exception("Failed to backup segfault file.")
            return

    except Exception:
        logger.exception("Failed to submit segfault to Sentry.")
