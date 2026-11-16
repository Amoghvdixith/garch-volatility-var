"""Small input-validation helpers shared across the package.

Each helper raises ``ValueError`` (or ``TypeError``) with a message that names
the offending argument, so failures surface early and explicitly instead of
propagating NaNs into risk numbers.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_finite_1d(values: Iterable[float], name: str = "values") -> np.ndarray:
    """Return ``values`` as a 1-D float array, rejecting empty or non-finite input."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"{name} contains {bad} non-finite value(s)")
    return arr


def check_confidence(confidence: float) -> float:
    """Validate a VaR confidence level ``c`` strictly inside (0, 1)."""
    c = float(confidence)
    if not np.isfinite(c) or not 0.0 < c < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence!r}")
    return c


def check_positive(value: float, name: str) -> float:
    """Validate a strictly positive finite scalar."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return v


def check_non_negative(value: float, name: str) -> float:
    """Validate a non-negative finite scalar."""
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    return v


def check_positive_int(value: int, name: str) -> int:
    """Validate a strictly positive integer (bools rejected)."""
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
