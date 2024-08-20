"""Dataframe utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path

from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    import pandas as pd


def read_csv_with_comments(path: PathLike) -> pd.DataFrame:
    """Read CSV with comments."""
    import pandas as pd

    path = Path(path)
    try:
        df = pd.read_csv(path)
        first_col = df.iloc[:, 0]
        # check whether any of the values start with # (comment)
        for value in first_col:
            if value.startswith("# "):
                df = df[~first_col.str.contains("# ")]
                df.reset_index(drop=True, inplace=True)
                df.columns = df.iloc[0]
                df.drop(df.index[0], inplace=True)
                break
    except pd.errors.ParserError:
        from io import StringIO

        data = path.read_text().split("\n")
        start_index, end_index = 0, 0
        for row in data:
            if row.startswith("#"):
                start_index += 1
                continue
            elif not row:
                end_index += 1
        df = pd.read_csv(StringIO("\n".join(data[start_index:-end_index])), sep=",")
    except Exception:
        raise pd.errors.ParserError(f"Failed to parse grid '{path}'.")
    return df
