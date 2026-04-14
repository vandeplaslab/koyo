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
    """Unzip directory."""
    logger.debug("Unzipping directory...")
    # unzip archive
    zip_file_object = zipfile.ZipFile(path_to_zip, "r")
    zip_file_object.extractall(path=output_dir)
    path_to_file = os.path.join(output_dir, zip_file_object.namelist()[0])
    zip_file_object.close()
    logger.debug("Unzipped directory")

    if remove_archive:  # Remove archive :
        logger.debug(f"Archive {path_to_zip} removed")
        os.remove(path_to_zip)
    return path_to_file


def zip_directory(path_to_directory: PathLike, zip_file: PathLike) -> PathLike:
    """Zip directory."""
    logger.debug("Zipping directory...")
    # zip directory
    zip_file_object = zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED)
    files = list(os.walk(path_to_directory))
    with MeasureTimer() as timer:
        for root, _, files in tqdm(files, desc="Zipping", unit="file"):
            for file in files:
                zip_file_object.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), path_to_directory),
                )
                logger.trace(f"Zipped file {file} in {timer(since_last=True)}.")
    zip_file_object.close()
    logger.debug(f"Zipped directory to {zip_file} in {timer()}.")
    return zip_file


def zip_files(zip_file: PathLike, *files: PathLike, remove_zipped: bool = False) -> PathLike:
    """Zip files."""
    logger.debug("Zipping files...")
    zip_file_object = zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED)
    with MeasureTimer() as timer:
        for file in tqdm(files, desc="Zipping", unit="file"):
            zip_file_object.write(file, os.path.basename(file))
            logger.trace(f"Zipped file {file} in {timer(since_last=True)}.")
    zip_file_object.close()
    if remove_zipped:
        for file in files:
            Path(file).unlink(missing_ok=True)
            logger.debug(f"Removing zipped file {file}...")
    logger.debug(f"Zipped files to {zip_file} in {timer()}.")
    return zip_file
