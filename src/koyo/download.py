"""Download functions."""

import os.path
import typing as ty
from pathlib import Path
from urllib import request

from loguru import logger
from tqdm import tqdm

from koyo.timer import MeasureTimer
from koyo.typing import PathLike


def download_progress(pbar):
    """Wraps tqdm instance.

    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).

    Example
    -------
    >>> with tqdm(...) as p_bar:
    ...     reporthook = download_progress(p_bar)
    ...     request.urlretrieve(reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tot_size=None):
        """Update progress.

        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tot_size  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tot_size is not None:
            pbar.total = tot_size
        pbar.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_file(
    link: str,
    filename: ty.Optional[PathLike] = None,
    output_dir: ty.Optional[PathLike] = None,
    unzip: bool = True,
    remove_archive: bool = False,
):
    """Download a file.

    By default, this function download a file to ~/explorer_data.

    Parameters
    ----------
    link : str
        Name of the file to download or url.
    filename : PathLike
        Name of the file to be saved in case of url.
    output_dir : PathLike
        Download file to the path specified.
    unzip : bool | False
        Unzip archive if needed.
    remove_archive : bool | False
        Remove archive after unzip.

    Returns
    -------
    path_to_file : Path
        Path to the downloaded file.
    """
    out_path = Path(__file__).parent
    if "http" in link:
        if filename is None:
            filename = os.path.split(link)[1]
            if "?" in filename:
                filename = filename.split("?")[0]
        assert isinstance(filename, str)
        url = link
    else:
        raise ValueError("Expected a link!")

    output_dir = out_path if not isinstance(output_dir, (str, Path)) else output_dir
    path_to_file = Path(output_dir) / filename
    temp_file = path_to_file.parent / (path_to_file.name + ".temp")
    to_download = not path_to_file.exists()

    # download file if needed
    with MeasureTimer() as timer:
        if to_download:
            logger.info(f"Downloading {path_to_file}")
            # Check if directory exists else creates it
            if not os.path.exists(output_dir):
                logger.info(f"Folder {output_dir} created")
                os.makedirs(output_dir)
            # Download file
            with tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {filename}...",
                mininterval=1.0,
            ) as p_bar:
                reporthook = download_progress(p_bar)
                _, _ = request.urlretrieve(url, temp_file, reporthook=reporthook)
            temp_file.rename(path_to_file)
        else:
            logger.info(f"File already downloaded ({path_to_file}). Skipping")
    if unzip:
        from koyo.compression import unzip_directory

        path_to_file = unzip_directory(path_to_file, output_dir, remove_archive)

    logger.info(f"Downloaded '{path_to_file}' in {timer()}.")
    return path_to_file
