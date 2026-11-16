"""In-sample standardised-residual diagnostics.

These describe the fitted residuals of one estimation window. They are
diagnostics, not proof that the shocks are i.i.d., and they say nothing
direct about out-of-sample forecast accuracy. Ljung-Box statistics are
computed by ``statsmodels`` (``acorr_ljungbox``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def residual_diagnostics(z: np.ndarray, lags: Iterable[int] = (10, 20)) -> list[dict]:
    """Mean, variance, skewness, excess kurtosis and Ljung-Box tests on ``z`` and ``z**2``."""
    z = np.asarray(z, dtype=float)
    n = int(z.size)
    centred = z - z.mean()
    sd = centred.std()
    rows = [
        {"diagnostic": "mean", "lag": np.nan, "statistic": float(z.mean()), "p_value": np.nan, "n": n},
        {"diagnostic": "variance", "lag": np.nan, "statistic": float(z.var(ddof=1)), "p_value": np.nan, "n": n},
        {"diagnostic": "skewness", "lag": np.nan, "statistic": float(np.mean(centred**3) / sd**3), "p_value": np.nan, "n": n},
        {"diagnostic": "excess_kurtosis", "lag": np.nan, "statistic": float(np.mean(centred**4) / sd**4 - 3.0), "p_value": np.nan, "n": n},
    ]
    lags = list(lags)
    for name, series in (("ljung_box_z", z), ("ljung_box_z_squared", z**2)):
        lb = acorr_ljungbox(series, lags=lags, return_df=True)
        for lag in lags:
            rows.append(
                {
                    "diagnostic": name,
                    "lag": lag,
                    "statistic": float(lb.loc[lag, "lb_stat"]),
                    "p_value": float(lb.loc[lag, "lb_pvalue"]),
                    "n": n,
                }
            )
    return rows


def autocorrelations(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelations at lags 1..nlags (statsmodels ``acf``)."""
    return acf(np.asarray(x, dtype=float), nlags=nlags, fft=False)[1:]
