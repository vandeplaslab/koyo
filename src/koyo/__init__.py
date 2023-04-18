"""Simple python ulitities for various projects."""
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

try:
    __version__ = version("koyo")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"


logger.disable("koyo")
