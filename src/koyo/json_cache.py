"""Json source."""

import typing as ty
import warnings
from json import JSONDecodeError
from pathlib import Path

from koyo.json import read_json_data, write_json_data
from koyo.typing import PathLike


class JSONCache:
    """Cache that data is stored in a JSON file."""

    FILENAME: str

    def __init__(self, path: PathLike):
        self._dir_path = Path(path)

    def __getattr__(self, item: str):
        return self[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.as_str()}>"

    def __getitem__(self, key: str) -> ty.Any:
        return self.read_key(key)

    def __setitem__(self, key: str, value: bool):
        self.write_key(key, value)

    def __contains__(self, item):
        return item in self.read()

    def __delitem__(self, key: str) -> None:
        data = self.read()
        if key in data:
            del data[key]
            self.write(data)

    def remove(self, key: str) -> None:
        """Remove key from cache."""
        data = self.read()
        if key in data:
            del data[key]
            self.write(data)

    def remove_multiple(self, *keys: str) -> None:
        """Remove multiple keys from cache."""
        data = self.read()
        for key in keys:
            if key in data:
                del data[key]
        self.write(data)

    def _set_default(self):
        data = {}
        self.write(data)

    def clear(self):
        """Remove existing tags."""
        self.write({})

    @property
    def name(self) -> str:
        """Return basename of the directory."""
        return self._dir_path.name

    def keys(self):
        """Return all keys in cache."""
        return self.read().keys()

    def items(self) -> ty.ItemsView:
        """Return all items in cache."""
        return self.read().items()

    def update(self, **kwargs: ty.Any) -> None:
        """Update cache with new key-value pairs."""
        data = self.read()
        data.update(**kwargs)
        self.write(data)

    def get(self, key: str, default=None):
        """Get key."""
        return self.get_key(key, default)

    @property
    def path(self) -> Path:
        """Get path of flag's data."""
        return self._dir_path / self.FILENAME

    def exists(self) -> bool:
        """Check whether flags file exists."""
        return self.path.exists()

    def as_str(self, sep="; ", exclude: ty.Optional[ty.Tuple[str, ...]] = None) -> str:
        """Get string representation of the flags."""
        if exclude is None:
            exclude = ()
        if self.exists():
            data = self.read()
            ret = ""
            n_stop = len(data) - 1
            for i, (k, v) in enumerate(data.items()):
                if k in exclude:
                    continue
                ret += f"{k}: {v}"
                if i != n_stop:
                    ret += sep
            return ret
        return ""

    def as_dict(self, exclude: ty.Optional[ty.Tuple[str, ...]] = None) -> dict:
        """Get dictionary representation of the flags."""
        if exclude is None:
            exclude = ()
        if self.exists():
            data = self.read()
            return {k: v for k, v in data.items() if k not in exclude}
        return {}

    def print_summary(self, name: str = "", pre: str = "\t", sep: str = "\n"):
        """Print summary about JSON store."""
        if name:
            print(name)
        data = self.read()
        if not data:
            print(f"{pre}<no data>")
        else:
            for k, v in data.items():
                print(f"{pre}{k}: {v}", sep=sep)

    @staticmethod
    def format_value(v) -> ty.Any:
        """Format value."""
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
            elif v.isnumeric():
                return int(v)
            try:
                return float(v)
            except ValueError:
                return v
        return v

    def read(self) -> ty.Dict:
        """Read data."""
        if self.exists():
            try:
                data = read_json_data(self.path)
                data = {k: self.format_value(v) for k, v in data.items()}
                return data
            except JSONDecodeError:
                warnings.warn(f"Failed to read JSON file: {self.path}")
        return {}

    def write(self, data: ty.Dict):
        """Write data to disk."""
        self._dir_path.mkdir(parents=True, exist_ok=True)
        write_json_data(self.path, data)

    def save(self, path: PathLike):
        """Save data to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_data(path, self.read())

    def get_key(self, key: str, default=None):
        """Read flag from the flag file."""
        if not self.exists():
            self._set_default()
        return self.read().get(key, default)

    def read_key(self, key: str):
        """Read flag from the flag file."""
        if not self.exists():
            self._set_default()
        return self.read()[key]

    def write_key(self, key: str, value: ty.Union[int, float, str, bool]):
        """Write flag to the flag file."""
        if not self.exists():
            self._set_default()
        data = self.read()
        data[key] = value
        self.write(data)

    def get_for_keys(self, keys: list[str], default: ty.Any = None) -> dict[str, ty.Any]:
        """Get values for specified keys."""
        return {k: self.get_key(k, default) for k in keys}
