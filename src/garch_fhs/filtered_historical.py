"""Filtered historical simulation (FHS) for one-day VaR and ES.

Algorithm for forecast date ``t`` (training window ends at ``t-1``):

1. Fit the volatility model on the training window (see ``volatility.py``).
2. Standardised residuals ``z_i = eps_i / sigma_i`` for every training
   observation; only non-finite values (or non-positive ``sigma_i``) are
   dropped. ``z`` is *not* recentred or renormalised.
3. One-step forecast ``sigma_t`` from information through ``t-1``.
4. Scenario returns ``r_t^(i) = sigma_t * z_i`` (percentage points; no extra
   factor of 100).
5. Scenario losses ``-r_t^(i)``; VaR/ES with the same empirical convention as
   plain HS (``historical.empirical_var_es``).

All residual scenarios are used directly with equal weight, which is the
deterministic one-step FHS calculation; no Monte Carlo resampling is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .historical import RiskEstimate, empirical_var_es
from .validation import check_positive


@dataclass(frozen=True)
class StandardizedResiduals:
    """Standardised residuals and how many raw values were excluded."""

    z: np.ndarray
    n_input: int
    n_dropped: int


def standardized_residuals(resid: np.ndarray, cond_vol: np.ndarray) -> StandardizedResiduals:
    """``z_i = eps_i / sigma_i``, dropping only invalid (non-finite / sigma <= 0) entries."""
    e = np.asarray(resid, dtype=float)
    s = np.asarray(cond_vol, dtype=float)
    if e.shape != s.shape:
        raise ValueError("resid and cond_vol must have the same shape")
    valid = np.isfinite(e) & np.isfinite(s) & (s > 0)
    z = e[valid] / s[valid]
    return StandardizedResiduals(z=z, n_input=int(e.size), n_dropped=int((~valid).sum()))


def fhs_scenario_returns(sigma_forecast: float, z: np.ndarray) -> np.ndarray:
    """Scenario returns ``sigma_t * z_i`` in percentage points."""
    s = check_positive(sigma_forecast, "sigma_forecast")
    return s * np.asarray(z, dtype=float)


def fhs_var_es(
    sigma_forecast: float, z: np.ndarray, confidence: float, min_residuals: int = 1
) -> RiskEstimate:
    """FHS VaR/ES of losses ``-sigma_t * z_i`` using the shared empirical convention."""
    z = np.asarray(z, dtype=float)
    if z.size < min_residuals:
        raise ValueError(f"only {z.size} valid standardised residuals; at least {min_residuals} required")
    return empirical_var_es(-fhs_scenario_returns(sigma_forecast, z), confidence)
