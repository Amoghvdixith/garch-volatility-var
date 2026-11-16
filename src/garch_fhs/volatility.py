"""GARCH(1,1) and GJR-GARCH(1,1,1) estimation and variance recursions.

Parameter estimation is delegated to the third-party ``arch`` package
(Kevin Sheppard). This module only configures ``arch``, records diagnostics
and provides plain-NumPy recursions used to check ``arch`` output.

``arch`` specification used for both models::

    arch_model(y, mean="zero", vol="garch", p=1, o=o, q=1, power=2.0,
               dist="normal", rescale=False)

with ``o = 0`` (GARCH) or ``o = 1`` (GJR). ``dist="normal"`` means the
parameters are Gaussian quasi-maximum-likelihood (QML) estimates; it does
*not* make the FHS tails Gaussian, because FHS uses the empirical
standardised residuals. ``rescale=False`` because inputs are already in
percentage points (daily variance of order 1), the scale ``arch``'s own
rescaling heuristic targets; fixing it keeps parameters in documented units.

``arch`` parameter names map to the documented symbols as::

    omega -> omega, alpha[1] -> alpha, gamma[1] -> gamma, beta[1] -> beta

and the GJR recursion in ``arch`` is
``sigma_t^2 = omega + (alpha + gamma*1[eps_{t-1}<0]) eps_{t-1}^2 + beta sigma_{t-1}^2``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

MODEL_SPECS: dict[str, dict[str, int]] = {"GARCH": {"o": 0}, "GJR": {"o": 1}}
ARCH_PARAM_MAP: dict[str, str] = {
    "omega": "omega",
    "alpha[1]": "alpha",
    "gamma[1]": "gamma",
    "beta[1]": "beta",
}


@dataclass
class FitResult:
    """Outcome of fitting one volatility model on one training window."""

    model: str
    status: str  # "ok", "ok_retry" or "failed"
    params: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    attempts: int = 0
    loglik: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    resid: np.ndarray | None = None
    cond_vol: np.ndarray | None = None
    sigma_forecast: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_retry")

    @property
    def persistence(self) -> float:
        return model_persistence(self.params)


def model_persistence(params: dict[str, float]) -> float:
    """``alpha + beta`` (GARCH) or ``alpha + gamma/2 + beta`` (GJR, symmetric-shock approximation)."""
    if not params:
        return float("nan")
    return params["alpha"] + 0.5 * params.get("gamma", 0.0) + params["beta"]


def garch_variance_recursion(
    eps: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: float
) -> np.ndarray:
    """GARCH(1,1) variances for ``eps`` plus the one-step-ahead value.

    Returns an array of length ``len(eps) + 1``: element ``t`` is the variance
    of ``eps[t]`` given information through ``t-1``; the last element is the
    forecast for the next, unobserved period.
    """
    return gjr_variance_recursion(eps, omega, alpha, 0.0, beta, sigma2_0)


def gjr_variance_recursion(
    eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, sigma2_0: float
) -> np.ndarray:
    """GJR-GARCH(1,1,1) variances for ``eps`` plus the one-step-ahead value."""
    e = np.asarray(eps, dtype=float)
    out = np.empty(e.size + 1)
    out[0] = sigma2_0
    for t in range(1, e.size + 1):
        prev = e[t - 1]
        out[t] = omega + (alpha + gamma * (prev < 0.0)) * prev * prev + beta * out[t - 1]
    return out


def one_step_variance(params: dict[str, float], last_eps: float, last_sigma2: float) -> float:
    """Documented one-step recursion from the final in-sample residual and variance."""
    a = params["alpha"] + params.get("gamma", 0.0) * (last_eps < 0.0)
    return params["omega"] + a * last_eps**2 + params["beta"] * last_sigma2


def build_arch_model(y: np.ndarray, model: str):
    """Construct the ``arch`` model object used for ``model`` ('GARCH' or 'GJR')."""
    from arch import arch_model

    if model not in MODEL_SPECS:
        raise ValueError(f"unknown volatility model {model!r}; expected one of {list(MODEL_SPECS)}")
    return arch_model(
        y,
        mean="zero",
        vol="garch",
        p=1,
        o=MODEL_SPECS[model]["o"],
        q=1,
        power=2.0,
        dist="normal",
        rescale=False,
    )


def _retry_starting_values(y: np.ndarray, model: str) -> np.ndarray:
    """Documented alternative starting values for the single retry."""
    var = float(np.var(y))
    if model == "GARCH":
        return np.array([0.05 * var, 0.05, 0.90])
    return np.array([0.05 * var, 0.03, 0.06, 0.90])


def fit_volatility(train_returns: np.ndarray, model: str) -> FitResult:
    """Fit GARCH or GJR on one training window and produce the one-step sigma forecast.

    Attempt 1 uses ``arch`` defaults. If it raises, does not converge or yields
    a non-finite/non-positive forecast, one retry is made **on the same
    window** with documented starting values and a larger iteration limit.
    If that also fails the result has ``status="failed"`` and no forecast;
    no alternative model is substituted.
    """
    y = np.asarray(train_returns, dtype=float)
    result = FitResult(model=model, status="failed")
    attempts = [
        {},
        {"starting_values": _retry_starting_values(y, model), "options": {"maxiter": 1000}},
    ]
    for i, kwargs in enumerate(attempts, start=1):
        result.attempts = i
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = build_arch_model(y, model).fit(disp="off", show_warning=True, **kwargs)
                fc_var = float(res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0])
            except Exception as exc:  # recorded, never silently ignored
                result.error = f"attempt {i}: {type(exc).__name__}: {exc}"
                result.warnings = [f"attempt {i}: {w.category.__name__}: {w.message}" for w in caught]
                continue
        result.warnings = [f"attempt {i}: {w.category.__name__}: {w.message}" for w in caught]
        converged = int(res.convergence_flag) == 0
        params = {ARCH_PARAM_MAP[k]: float(v) for k, v in res.params.items()}
        result.params = params
        result.converged = converged
        result.loglik = float(res.loglikelihood)
        if converged and np.isfinite(fc_var) and fc_var > 0:
            result.resid = np.asarray(res.resid, dtype=float)
            result.cond_vol = np.asarray(res.conditional_volatility, dtype=float)
            result.sigma_forecast = float(np.sqrt(fc_var))
            result.status = "ok" if i == 1 else "ok_retry"
            result.error = ""
            return result
        result.error = f"attempt {i}: not converged (flag {res.convergence_flag}) or invalid forecast"
    result.status = "failed"
    return result
