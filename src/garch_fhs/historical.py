"""Empirical VaR / ES and plain historical simulation.

Empirical distribution convention (used by HS **and** FHS):

* Every scenario loss ``L_1..L_n`` receives equal probability mass ``1/n``.
* ``VaR_c`` is the inverse empirical CDF at ``c``: with ascending order
  statistics ``L_(1) <= ... <= L_(n)``, ``VaR_c = L_(k)`` where
  ``k = ceil(n * c)`` is the smallest index with ``k/n >= c``.
* ``ES_c`` is the average loss over the worst ``p = 1 - c`` probability mass.
  With ``m = n * p``, the ``floor(m)`` largest losses enter with full mass and
  the next-largest loss enters with the fractional remaining mass
  ``m - floor(m)``::

      ES_c = ( sum_{j=1}^{floor(m)} L_[j] + (m - floor(m)) * L_[floor(m)+1] ) / m

  where ``L_[j]`` are losses in descending order. This equals the
  Acerbi-Tasche / Rockafellar-Uryasev value
  ``VaR_c + E[(L - VaR_c)^+] / p`` under the empirical measure, so
  ``ES_c >= VaR_c`` always holds.

Losses are ``loss_pct = -return_pct`` (negative log return in percentage
points). Risk numbers are never clipped to be positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .validation import as_finite_1d, check_confidence, check_positive

# Tolerance used when n*c or n*(1-c) is an integer up to floating-point error
# (e.g. 1000 * 0.95). Without it ceil(950.0000000001) would give 951.
_INT_TOL = 1e-9


@dataclass(frozen=True)
class RiskEstimate:
    """VaR and ES at one confidence level, in the units of the input losses."""

    confidence: float
    var: float
    es: float
    n_scenarios: int


def _tail_split(n: int, confidence: float) -> tuple[int, int, float]:
    """Return ``(k, full, frac)``: VaR order-statistic index and ES tail mass split."""
    k = math.ceil(n * confidence - _INT_TOL)
    k = min(max(k, 1), n)
    m = n * (1.0 - confidence)
    full = math.floor(m + _INT_TOL)
    frac = m - full
    if frac < _INT_TOL:
        frac = 0.0
    return k, full, frac


def empirical_var(losses: Iterable[float], confidence: float) -> float:
    """Empirical VaR: inverse empirical CDF of ``losses`` at ``confidence``."""
    c = check_confidence(confidence)
    x = np.sort(as_finite_1d(losses, "losses"))
    k, _, _ = _tail_split(x.size, c)
    return float(x[k - 1])


def empirical_es(losses: Iterable[float], confidence: float) -> float:
    """Empirical ES: mean of the worst ``1 - c`` probability mass (fractional boundary)."""
    c = check_confidence(confidence)
    x = np.sort(as_finite_1d(losses, "losses"))[::-1]  # descending
    n = x.size
    _, full, frac = _tail_split(n, c)
    m = n * (1.0 - c)
    total = float(np.sum(x[:full]))
    if frac > 0.0:
        total += frac * float(x[full])
    return total / m


def empirical_var_es(losses: Iterable[float], confidence: float) -> RiskEstimate:
    """Compute VaR and ES together under the shared convention."""
    arr = as_finite_1d(losses, "losses")
    return RiskEstimate(
        confidence=check_confidence(confidence),
        var=empirical_var(arr, confidence),
        es=empirical_es(arr, confidence),
        n_scenarios=int(arr.size),
    )


def returns_to_losses(returns_pct: Iterable[float]) -> np.ndarray:
    """Primary loss convention: ``loss_pct = -return_pct`` (long position)."""
    return -as_finite_1d(returns_pct, "returns_pct")


def monetary_losses(returns_pct: Iterable[float], notional: float) -> np.ndarray:
    """Exact monetary loss of a long position: ``notional * (1 - exp(r / 100))``.

    ``r`` is a percentage-point log return. Apply this to *scenario* returns
    before taking VaR/ES to obtain monetary risk measures; because the map is
    monotone decreasing in ``r`` the empirical quantile ordering is preserved.
    """
    n = check_positive(notional, "notional")
    r = as_finite_1d(returns_pct, "returns_pct")
    return n * (1.0 - np.exp(r / 100.0))


def historical_simulation(train_returns_pct: Iterable[float], confidence: float) -> RiskEstimate:
    """Plain HS: empirical VaR/ES of the trailing-window losses ``-r``."""
    return empirical_var_es(returns_to_losses(train_returns_pct), confidence)
