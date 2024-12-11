"""Compression utilities."""

import os
import zipfile

from loguru import logger
from tqdm import tqdm

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


def zip_directory(path_to_directory: PathLike, output_dir: PathLike) -> PathLike:
    """Zip directory."""
    logger.debug("Zipping directory...")
    # zip directory
    zip_file_object = zipfile.ZipFile(output_dir, "w", zipfile.ZIP_DEFLATED)
    files = list(os.walk(path_to_directory))
    for root, _, files in tqdm(files, desc="Zipping", unit="file"):
        for file in files:
            zip_file_object.write(
                os.path.join(root, file), os.path.relpath(os.path.join(root, file), path_to_directory)
            )
    zip_file_object.close()
    logger.debug("Zipped directory")
    return output_dir
