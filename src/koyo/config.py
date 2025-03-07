"""Configuration module."""

import typing as ty
from contextlib import contextmanager, suppress
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base config."""

    USER_CONFIG_DIR: ty.ClassVar[Path]
    USER_CONFIG_FILENAME: ty.ClassVar[str] = "config.json"

    def __init__(self, _auto_load: bool = False, **kwargs: ty.Any):
        super().__init__(**kwargs)
        if _auto_load:
            self.load()

    @property
    def output_path(self) -> Path:
        """Get default output path."""
        self.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return self.USER_CONFIG_DIR / self.USER_CONFIG_FILENAME

    def update(self, save: bool = True, **kwargs: ty.Any) -> None:
        """Update configuration and save to file."""
        changed = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                try:
                    setattr(self, key, value)
                    if old_value != getattr(self, key):
                        changed = True
                except Exception as e:
                    logger.warning(f"Failed to set {key}={value}: {e}")
            else:
                logger.warning(f"Unknown key {key}={value} - perhaps it was deprecated?")
        if save and changed:
            with suppress(OSError, PermissionError):
                self.save()

    def get_exclude_fields(self) -> set[str]:
        """Get fields to exclude from saving."""
        exclude = []
        schema = self.model_json_schema()["properties"]
        for field_name, _field in self.model_fields.items():
            field_schema = schema[field_name]
            if not field_schema.get("save", True):
                exclude.append(field_name)
        return set(exclude)

    def save(self) -> None:
        """Export configuration to file."""
        try:
            self.output_path.write_text(
                self.model_dump_json(indent=4, exclude_unset=True, exclude=self.get_exclude_fields())
            )
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

    @contextmanager
    def temporary_overwrite(self, **kwargs: ty.Any) -> ty.Generator[None, None, None]:
        """Temporarily overwrite configuration and then revert back."""
        original = {key: getattr(self, key) for key in kwargs}
        for key, value in kwargs.items():
            setattr(self, key, value)
        yield
        for key, value in original.items():
            setattr(self, key, value)
