"""Daily return preparation and CSV loading.

Unit convention: returns are percentage-point log returns,
``r_t = 100 * log(P_t / P_{t-1})``; a 1% move is stored as ``1.0``.

Accepted CSV schemas (column names are matched case-insensitively after
stripping whitespace):

* ``date`` + ``adjusted_close`` -> returns are computed from prices;
* ``date`` + ``return_pct``     -> returns are used as supplied.

A file containing both ``adjusted_close`` and ``return_pct`` (or neither) is
rejected as ambiguous. Additional columns (for example the synthetic
``true_vol_pct``) are preserved.

Policies:

* Dates are parsed and sorted ascending; an unsorted file is sorted (not
  rejected) and this is reported through the ``was_sorted`` attribute.
* Duplicate dates are **rejected** (no silent de-duplication).
* Missing or non-finite values in the price/return column are **rejected**;
  the caller must clean the data explicitly beforehand.
* Prices must be strictly positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COL = "adjusted_close"
RETURN_COL = "return_pct"
DATE_COL = "date"


class SchemaError(ValueError):
    """Raised when a CSV does not match exactly one supported schema."""


@dataclass
class LoadedReturns:
    """Validated return frame plus information about how it was prepared."""

    frame: pd.DataFrame
    schema: str
    was_sorted: bool
    dropped_first_row: bool


def log_returns_pct(prices: np.ndarray | pd.Series) -> np.ndarray:
    """Percentage-point log returns ``100 * log(P_t / P_{t-1})``.

    The output has one fewer element than ``prices``. Prices must be finite
    and strictly positive.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be a 1-D array with at least two values")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices contain non-finite values")
    if np.any(p <= 0.0):
        raise ValueError("prices must be strictly positive")
    return 100.0 * np.diff(np.log(p))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: str(c).strip().lower() for c in df.columns}
    out = df.rename(columns=renamed)
    if out.columns.duplicated().any():
        raise SchemaError("column names collide after case normalisation")
    return out


def detect_schema(columns: list[str]) -> str:
    """Return ``'price'`` or ``'return'`` for a normalised column list."""
    cols = set(columns)
    if DATE_COL not in cols:
        raise SchemaError("CSV must contain a 'date' column")
    has_price = PRICE_COL in cols
    has_return = RETURN_COL in cols
    if has_price and has_return:
        raise SchemaError(
            "ambiguous CSV: both 'adjusted_close' and 'return_pct' present; supply exactly one"
        )
    if not (has_price or has_return):
        raise SchemaError("CSV must contain either 'adjusted_close' or 'return_pct'")
    return "price" if has_price else "return"


def prepare_frame(df: pd.DataFrame) -> LoadedReturns:
    """Validate a raw frame and return a date-indexed frame with ``return_pct``."""
    df = _normalise_columns(df)
    schema = detect_schema(list(df.columns))

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum())} unparseable date value(s)")
    df = df.assign(**{DATE_COL: dates})

    if df[DATE_COL].duplicated().any():
        dups = df.loc[df[DATE_COL].duplicated(keep=False), DATE_COL].dt.date.unique()
        raise ValueError(f"duplicate dates are not allowed: {list(dups)[:5]}")

    was_sorted = not df[DATE_COL].is_monotonic_increasing
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    value_col = PRICE_COL if schema == "price" else RETURN_COL
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column '{value_col}' contains missing or non-finite values")

    if schema == "price":
        rets = log_returns_pct(values)
        out = df.iloc[1:].copy()
        out[RETURN_COL] = rets
        dropped = True
    else:
        out = df.copy()
        out[RETURN_COL] = values
        dropped = False

    out = out.set_index(DATE_COL)
    out.index.name = DATE_COL
    return LoadedReturns(frame=out, schema=schema, was_sorted=was_sorted, dropped_first_row=dropped)


def extract_adjusted_close(raw: pd.DataFrame, symbol: str, field: str = "Adj Close") -> pd.Series:
    """Pull the adjusted-close column out of a yfinance-style download.

    Handles both flat columns (``'Adj Close'``) and the MultiIndex layouts that
    recent yfinance versions return (``('Adj Close', 'SPY')`` or
    ``('SPY', 'Adj Close')``). Raises if the field is absent rather than
    silently falling back to unadjusted ``Close``.
    """
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if field in cols.get_level_values(level):
                sub = raw.xs(field, axis=1, level=level)
                if isinstance(sub, pd.Series):
                    return sub.rename(field)
                if symbol in sub.columns:
                    return sub[symbol].rename(field)
                if sub.shape[1] == 1:
                    return sub.iloc[:, 0].rename(field)
                raise ValueError(f"multiple tickers in download and {symbol!r} not among them")
        raise ValueError(f"field {field!r} not found in downloaded columns {list(cols)[:6]}")
    if field not in cols:
        raise ValueError(f"field {field!r} not found in downloaded columns {list(cols)}; "
                         "download with auto_adjust=False")
    return raw[field].rename(field)


def load_config(path: str | Path) -> dict:
    """Read a JSON experiment configuration file."""
    import json

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve(root: Path, relative: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths kept)."""
    rel = Path(relative)
    return rel if rel.is_absolute() else root / rel


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes (dataset identity in metadata)."""
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_returns_csv(path: str | Path) -> LoadedReturns:
    """Load and validate a CSV file using :func:`prepare_frame`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input data file not found: {p}")
    return prepare_frame(pd.read_csv(p))
