"""Compression utilities."""

import os
import zipfile
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from koyo.timer import MeasureTimer
from koyo.typing import PathLike

DECOMPRESSION_FORMATS = ["gz", ".gz", "*.gz"]


def unzip_directory(path_to_zip: PathLike, output_dir: PathLike, remove_archive: bool = True) -> PathLike:
    """Unzip an archive into `output_dir`.

    Returns the path of the first entry in the archive, joined with `output_dir`. The archive is deleted
    afterwards if `remove_archive` is True (default).
    """
    logger.debug("Unzipping directory...")
    # unzip archive
    with zipfile.ZipFile(path_to_zip, "r") as zip_file_object:
        zip_file_object.extractall(path=output_dir)
        path_to_file = os.path.join(output_dir, zip_file_object.namelist()[0])
    logger.debug("Unzipped directory")

    if remove_archive:  # Remove archive :
        logger.debug(f"Archive {path_to_zip} removed")
        os.remove(path_to_zip)
    return path_to_file


def zip_directory(path_to_directory: PathLike, zip_file: PathLike) -> PathLike:
    """Zip directory."""
    logger.debug("Zipping directory...")
    # zip directory
    walked = list(os.walk(path_to_directory))
    zip_path = Path(zip_file).resolve()
    with MeasureTimer() as timer, zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zip_file_object:
        for root, _, filenames in tqdm(walked, desc="Zipping", unit="file"):
            for file in filenames:
                file_path = os.path.join(root, file)
                if Path(file_path).resolve() == zip_path:  # don't add the archive to itself
                    continue
                zip_file_object.write(file_path, os.path.relpath(file_path, path_to_directory))
                logger.trace(f"Zipped file {file} in {timer(since_last=True)}.")
    logger.debug(f"Zipped directory to {zip_file} in {timer()}.")
    return zip_file


def zip_files(zip_file: PathLike, *files: PathLike, remove_zipped: bool = False) -> PathLike:
    """Zip files."""
    logger.debug("Zipping files...")
    zip_file = Path(zip_file)
    zip_file_object = zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED)
    with MeasureTimer() as timer:
        try:
            for file in tqdm(files, desc="Zipping", unit="file"):
                zip_file_object.write(file, os.path.basename(file))
                logger.trace(f"Zipped file {file} in {timer(since_last=True)}.")
            zip_file_object.close()
        except FileNotFoundError:
            logger.warning(f"Failed to zip files {files} to {zip_file}.")
            remove_zipped = False  # to prevent accidental deletion of files if zip failed
            zip_file_object.close()
            zip_file.unlink(missing_ok=True)

    if remove_zipped and zip_file.exists():
        for file in files:
            Path(file).unlink(missing_ok=True)
            logger.debug(f"Removing zipped file {file}...")
    logger.debug(f"Zipped files to {zip_file} in {timer()}.")
    return zip_file
