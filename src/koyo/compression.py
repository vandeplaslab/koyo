"""Compression utilities."""
import os
import zipfile

from loguru import logger

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
        logger.debug("Archive %s removed" % path_to_zip)
        os.remove(path_to_zip)

    return path_to_file
