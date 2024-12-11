"""Project interface.

In multiple projects in the Van de Plas lab we use the 'project' interface. It basically controls how we organize
our data and how we interact with it.
"""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from koyo.typing import PathLike

logger = logger.bind(src="Project")


def is_none(value: ty.Any, default: ty.Any) -> ty.Any:
    """Returns the value if value is not None, otherwise it returns the default."""
    return value if value is not None else default


class ProjectBase:
    """Project base."""

    BASE_FILENAME: str = "project.config.json"
    PROJECT_SUFFIX: str = ".project"

    # config attributes
    DATASETS_KEY: ty.Literal["datasets"] = "datasets"
    METADATA_KEY: ty.Literal["metadata"] = "metadata"
    PARAMS_KEY: ty.Literal["parameters"] = "parameters"
    EXECUTION_KEY: ty.Literal["execution"] = "execution"

    def __init__(self, project_dir: PathLike, cache_dir: PathLike | None = None):
        self.project_dir = Path(project_dir)
        if not self.project_dir.suffix == self.PROJECT_SUFFIX:
            self.project_dir = self.project_dir.with_suffix(self.PROJECT_SUFFIX)
        self.project_dir.mkdir(exist_ok=True, parents=True)
        self._cache_dir = cache_dir
        self._log_dir = self.project_dir / "Logs"
        self._figures_dir = self.project_dir / "Figures"
        self._results_dir = self.project_dir / "Results"
        self._config = None

    @property
    def name(self) -> str:
        """Return name of the project."""
        return self.project_dir.stem

    @property
    def filename(self) -> Path:
        """Get path to the project configuration file."""
        return self.project_dir / self.BASE_FILENAME

    @classmethod
    def new(cls, output_dir: PathLike, name: str = "Project", open_if_exists: bool = True) -> ProjectBase:
        """Create new project."""
        project_dir = (Path(output_dir) / name).with_suffix(".annotine")
        if project_dir.exists():
            if open_if_exists:
                return cls(project_dir)
            raise OSError("Project with this name already exists.")
        logger.info(f"Created new project '{project_dir.name}'")
        return cls(project_dir)

    @property
    def cache_dir(self) -> Path:
        """Return path to cache directory."""
        if self._cache_dir is None:
            self._cache_dir = self.project_dir / "Cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        return self._cache_dir  # type: ignore[return-value]

    @cache_dir.setter
    def cache_dir(self, value: PathLike) -> None:
        self._cache_dir = Path(value)

    @property
    def log_dir(self) -> Path:
        """Return path to Logs directory."""
        if self._log_dir is None:
            self._log_dir = self.project_dir / "Logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        return self._log_dir

    @property
    def figures_dir(self) -> Path:
        """Return path to Figure directory."""
        if self._figures_dir is None:
            self._figures_dir = self.project_dir / "Figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        return self._figures_dir

    @property
    def results_dir(self) -> Path:
        """Get path or dictionary of paths."""
        results_dir = self.project_dir / "Results"
        results_dir.mkdir(exist_ok=True, parents=True)
        return results_dir

    @property
    def project_config(self) -> dict:
        """Get dict containing project configuration."""
        from json import load
        from json.decoder import JSONDecodeError

        if self._config is None:
            if self.filename.exists():
                try:
                    with open(self.filename) as f_ptr:
                        data = load(f_ptr)
                    self._config = self._parse_config(data)
                except JSONDecodeError:
                    self._config = {}
            else:
                self._config = {}
            self._validate_config()
        return self._config

    def validate_inputs(self) -> None:
        """Validate input paths."""
        raise NotImplementedError("Must implement method")

    def _parse_config(self, config: dict) -> dict:
        """Parse configuration."""
        return config

    def _validate_config(self) -> None:
        """Validate config."""
        raise NotImplementedError("Must implement method")

    def _cleanup_config(self) -> dict:
        """Cleanup config."""
        raise NotImplementedError("Must implement method")

    def save(self) -> None:
        """Save project configuration."""
        self._export()
        logger.info(f"Project configuration saved to {self.filename}")

    @contextmanager
    def autosave(self) -> ty.Generator[None, None, None]:
        """Context manager to automatically save project configuration."""
        yield
        self._export()

    def _export(self) -> None:
        """Export configuration file."""
        from json import dump

        if not self.project_config or self.project_config is None:
            raise ValueError("Project configuration file is not setup.")

        # create backup of the project file
        if self.filename.exists():
            backup_filename = self.filename.with_suffix(".bak")
            backup_filename.write_text(self.filename.read_text())

        # validate config
        self._validate_config()
        project_config = self._cleanup_config()
        with open(self.filename, "w") as f_ptr:
            dump(project_config, f_ptr, indent=2)

    @property
    def n_datasets(self) -> int:
        """Get number of datasets."""
        return len(self.datasets)

    @property
    def datasets(self) -> list[str]:
        """List set of available datasets that have been registered to the project."""
        return list(self.project_config.get(self.DATASETS_KEY, {}).keys())

    @property
    def paths(self) -> list[Path]:
        """List set of available datasets that have been registered to the project."""
        return list(map(Path, self.project_config.get(self.DATASETS_KEY, {}).values()))

    def dataset_iter(self) -> ty.Iterable[str]:
        """Iterate dataset name and path."""
        yield from self.project_config.get(self.DATASETS_KEY, {}).keys()

    def dataset_path_iter(self) -> ty.Iterator[tuple[str, Path]]:
        """Iterator of dataset:path values."""
        yield from zip(self.datasets, self.paths)
