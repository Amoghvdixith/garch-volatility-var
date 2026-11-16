"""Synthetic GJR-GARCH(1,1,1) returns with Student-t innovations and stress regimes.

Data-generating process (percentage-point units)::

    r_t       = mu + eps_t
    eps_t     = sigma_t * z_t,         z_t ~ standardised Student-t(nu), Var(z_t) = 1
    sigma_t^2 = omega_t + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2
                + beta * sigma_{t-1}^2

``z_t = T_t * sqrt((nu - 2) / nu)`` with ``T_t`` a standard Student-t draw, so
``Var(z_t) = 1`` (requires ``nu > 2``).

Initialisation: the first burn-in observation uses the unconditional variance
of the baseline (non-stress) process, ``sigma_0^2 = omega / (1 - alpha - gamma/2 - beta)``.
The burn-in observations are then discarded.

Stress regimes: for retained (post-burn-in, zero-based) indices inside each
configured range, ``omega_t = omega * stress_omega_multiplier``; elsewhere
``omega_t = omega``. These temporary regime shifts intentionally depart from a
single constant-parameter GJR process, so no estimated constant-parameter
model is exactly correctly specified across the full sample.

Finite-variance condition for symmetric innovations: ``alpha + gamma/2 + beta < 1``
(since ``P(z_t < 0) = 1/2``). It is checked for the baseline parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .validation import check_positive_int

REGIME_NORMAL = "normal"


@dataclass(frozen=True)
class StressRegime:
    """A predefined stress interval on retained zero-based indices (inclusive)."""

    label: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class SyntheticConfig:
    """All parameters of the synthetic data-generating process."""

    seed: int = 42
    mu: float = 0.0
    omega: float = 0.02
    alpha: float = 0.05
    gamma: float = 0.08
    beta: float = 0.88
    nu: float = 8.0
    burn_in: int = 1000
    n_obs: int = 3500
    start_date: str = "2000-01-03"
    stress_omega_multiplier: float = 9.0
    stress_regimes: tuple[StressRegime, ...] = field(
        default_factory=lambda: (
            StressRegime("stress_1", 1400, 1499),
            StressRegime("stress_2", 2500, 2599),
        )
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticConfig":
        """Build from a JSON-style dict (``stress_regimes`` as a list of dicts)."""
        d = dict(d)
        regimes = tuple(StressRegime(**r) for r in d.pop("stress_regimes", []))
        return cls(stress_regimes=regimes, **d)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["stress_regimes"] = [asdict(r) for r in self.stress_regimes]
        return out


def persistence(alpha: float, gamma: float, beta: float) -> float:
    """GJR variance persistence with symmetric innovations: ``alpha + gamma/2 + beta``."""
    return alpha + 0.5 * gamma + beta


def unconditional_variance(omega: float, alpha: float, gamma: float, beta: float) -> float:
    """Unconditional variance ``omega / (1 - persistence)`` of a stationary GJR process."""
    return omega / (1.0 - persistence(alpha, gamma, beta))


def standardized_t_scale(nu: float) -> float:
    """Factor ``sqrt((nu-2)/nu)`` that maps a Student-t(nu) draw to unit variance."""
    if nu <= 2.0:
        raise ValueError(f"nu must exceed 2 for finite variance, got {nu}")
    return float(np.sqrt((nu - 2.0) / nu))


def validate_config(cfg: SyntheticConfig) -> None:
    """Check parameter admissibility, the finite-variance condition and regimes."""
    if not cfg.omega > 0:
        raise ValueError("omega must be > 0")
    if cfg.alpha < 0 or cfg.beta < 0:
        raise ValueError("alpha and beta must be >= 0")
    if cfg.alpha + cfg.gamma < 0:
        raise ValueError("alpha + gamma must be >= 0 so negative shocks cannot reduce variance")
    if not cfg.nu > 2:
        raise ValueError("nu must be > 2 for unit-variance standardisation")
    if persistence(cfg.alpha, cfg.gamma, cfg.beta) >= 1.0:
        raise ValueError("alpha + gamma/2 + beta must be < 1 (finite unconditional variance)")
    if not cfg.stress_omega_multiplier > 0:
        raise ValueError("stress_omega_multiplier must be > 0")
    check_positive_int(cfg.n_obs, "n_obs")
    if cfg.burn_in < 0 or int(cfg.burn_in) != cfg.burn_in:
        raise ValueError("burn_in must be a non-negative integer")
    taken = np.zeros(cfg.n_obs, dtype=bool)
    labels = set()
    for r in cfg.stress_regimes:
        if r.label == REGIME_NORMAL or r.label in labels:
            raise ValueError(f"invalid or duplicate regime label {r.label!r}")
        labels.add(r.label)
        if not 0 <= r.start_index <= r.end_index < cfg.n_obs:
            raise ValueError(f"regime {r.label} indices outside [0, n_obs)")
        if taken[r.start_index : r.end_index + 1].any():
            raise ValueError(f"regime {r.label} overlaps another regime")
        taken[r.start_index : r.end_index + 1] = True


def regime_labels(cfg: SyntheticConfig) -> np.ndarray:
    """Regime label for every retained observation (``'normal'`` outside stress)."""
    labels = np.full(cfg.n_obs, REGIME_NORMAL, dtype=object)
    for r in cfg.stress_regimes:
        labels[r.start_index : r.end_index + 1] = r.label
    return labels


def omega_path(cfg: SyntheticConfig) -> np.ndarray:
    """``omega_t`` for burn-in + retained observations (burn-in is never stressed)."""
    om = np.full(cfg.burn_in + cfg.n_obs, cfg.omega, dtype=float)
    for r in cfg.stress_regimes:
        om[cfg.burn_in + r.start_index : cfg.burn_in + r.end_index + 1] *= cfg.stress_omega_multiplier
    return om


def simulate_gjr_path(
    z: np.ndarray,
    omega_t: np.ndarray,
    alpha: float,
    gamma: float,
    beta: float,
    sigma2_init: float,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the GJR recursion for given unit-variance shocks ``z``.

    Returns ``(returns, sigma2)``. ``sigma2[0] = sigma2_init``; for ``t >= 1``
    ``sigma2[t]`` uses ``eps[t-1]`` and ``sigma2[t-1]`` only.
    """
    n = len(z)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = sigma2_init
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        e = eps[t - 1]
        sigma2[t] = omega_t[t] + (alpha + gamma * (e < 0.0)) * e * e + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return mu + eps, sigma2


def generate(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate the retained synthetic sample as a DataFrame.

    Columns: ``date`` (synthetic Mon-Fri business-day index, *not* an exchange
    calendar), ``return_pct``, ``true_vol_pct`` (conditional sigma_t of r_t),
    ``stress_flag`` (0/1) and ``regime``.
    """
    validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    total = cfg.burn_in + cfg.n_obs
    z = rng.standard_t(cfg.nu, size=total) * standardized_t_scale(cfg.nu)
    sigma2_init = unconditional_variance(cfg.omega, cfg.alpha, cfg.gamma, cfg.beta)
    returns, sigma2 = simulate_gjr_path(
        z, omega_path(cfg), cfg.alpha, cfg.gamma, cfg.beta, sigma2_init, cfg.mu
    )
    keep = slice(cfg.burn_in, total)
    labels = regime_labels(cfg)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(cfg.start_date, periods=cfg.n_obs),
            "return_pct": returns[keep],
            "true_vol_pct": np.sqrt(sigma2[keep]),
            "stress_flag": (labels != REGIME_NORMAL).astype(int),
            "regime": labels,
        }
    )
