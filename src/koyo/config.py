"""Configuration module."""
import typing as ty
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base config."""

    USER_CONFIG_DIR: ty.ClassVar[Path]
    USER_CONFIG_FILENAME: ty.ClassVar[str] = "config.json"

    @property
    def output_path(self) -> Path:
        """Get default output path."""
        self.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return self.USER_CONFIG_DIR / self.USER_CONFIG_FILENAME

    def save(self) -> None:
        """Export configuration to file."""
        try:
            self.output_path.write_text(self.json(indent=4, exclude_unset=True))
            logger.info(f"Saved configuration to {self.output_path}")
        except Exception as e:
            logger.warning(f"Failed to save configuration to {self.output_path}: {e}")

    def load(self) -> None:
        """Load configuration from file."""
        from koyo.json import read_json_data

        if self.output_path.exists():
            try:
                data = read_json_data(self.output_path)
                for key, value in data.items():
                    if hasattr(self, key):
                        try:
                            setattr(self, key, value)
                        except Exception as e:
                            logger.warning(f"Failed to set {key}={value}: {e}")
                    else:
                        logger.warning(f"Unknown key {key}={value} - perhaps it was deprecated?")
                logger.info(f"Loaded configuration from {self.output_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.output_path}: {e}")
                logger.exception(e)
