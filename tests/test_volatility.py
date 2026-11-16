"""Volatility recursions and the arch wrapper."""

import numpy as np
import pytest

from garch_fhs.synthetic import SyntheticConfig, generate
from garch_fhs.volatility import (
    ARCH_PARAM_MAP,
    build_arch_model,
    fit_volatility,
    garch_variance_recursion,
    gjr_variance_recursion,
    model_persistence,
    one_step_variance,
)


def test_hand_calculated_garch_recursion():
    # omega=0.1, alpha=0.1, beta=0.8, sigma2_0=1, eps=[2, -1]
    # s1 = 0.1 + 0.1*4 + 0.8*1   = 1.3
    # s2 = 0.1 + 0.1*1 + 0.8*1.3 = 1.24   (sign irrelevant for GARCH)
    out = garch_variance_recursion(np.array([2.0, -1.0]), 0.1, 0.1, 0.8, 1.0)
    np.testing.assert_allclose(out, [1.0, 1.3, 1.24])


def test_hand_calculated_gjr_recursion_positive_and_negative():
    # omega=0.1, alpha=0.05, gamma=0.1, beta=0.8, sigma2_0=1, eps=[2, -1]
    # s1 (eps0=+2, no gamma) = 0.1 + 0.05*4 + 0.8*1      = 1.1
    # s2 (eps1=-1, gamma on) = 0.1 + 0.15*1 + 0.8*1.1    = 1.13
    out = gjr_variance_recursion(np.array([2.0, -1.0]), 0.1, 0.05, 0.1, 0.8, 1.0)
    np.testing.assert_allclose(out, [1.0, 1.1, 1.13])
    # Same magnitude, opposite sign: negative shock gives larger variance.
    up = gjr_variance_recursion(np.array([1.5]), 0.1, 0.05, 0.1, 0.8, 1.0)[-1]
    down = gjr_variance_recursion(np.array([-1.5]), 0.1, 0.05, 0.1, 0.8, 1.0)[-1]
    assert down - up == pytest.approx(0.1 * 1.5**2)


def test_one_step_variance_function():
    p = {"omega": 0.1, "alpha": 0.05, "gamma": 0.1, "beta": 0.8}
    assert one_step_variance(p, -1.0, 1.1) == pytest.approx(1.13)
    assert one_step_variance(p, 1.0, 1.1) == pytest.approx(0.1 + 0.05 + 0.88)


def test_persistence_definitions():
    assert model_persistence({"omega": 0.1, "alpha": 0.1, "beta": 0.8}) == pytest.approx(0.9)
    assert model_persistence({"omega": 0.1, "alpha": 0.05, "gamma": 0.1, "beta": 0.8}) == pytest.approx(0.9)


def test_arch_model_configuration_and_parameter_mapping():
    y = generate(SyntheticConfig(n_obs=500, burn_in=100, stress_regimes=()))["return_pct"].to_numpy()
    garch = build_arch_model(y, "GARCH")
    gjr = build_arch_model(y, "GJR")
    assert (garch.volatility.p, garch.volatility.o, garch.volatility.q, garch.volatility.power) == (1, 0, 1, 2.0)
    assert (gjr.volatility.p, gjr.volatility.o, gjr.volatility.q) == (1, 1, 1)
    assert type(garch).__name__ == "ZeroMean"
    assert garch.distribution.name == "Normal"
    assert garch.rescale is False
    res = gjr.fit(disp="off")
    assert list(res.params.index) == ["omega", "alpha[1]", "gamma[1]", "beta[1]"]
    assert set(ARCH_PARAM_MAP) == set(res.params.index)
    assert res.scale == 1.0  # no automatic rescaling applied


@pytest.fixture(scope="module")
def gjr_sample():
    cfg = SyntheticConfig(seed=11, n_obs=1000, burn_in=500, stress_regimes=())
    return generate(cfg)["return_pct"].to_numpy()


@pytest.mark.parametrize("model", ["GARCH", "GJR"])
def test_arch_in_sample_variance_follows_documented_recursion(gjr_sample, model):
    fit = fit_volatility(gjr_sample, model)
    assert fit.ok
    s2 = fit.cond_vol**2
    eps = fit.resid
    np.testing.assert_allclose(eps, gjr_sample)  # zero mean: residual = return
    p = fit.params
    expected = p["omega"] + (p["alpha"] + p.get("gamma", 0.0) * (eps[:-1] < 0)) * eps[:-1] ** 2 + p["beta"] * s2[:-1]
    np.testing.assert_allclose(s2[1:], expected, rtol=1e-10)


@pytest.mark.parametrize("model", ["GARCH", "GJR"])
def test_arch_one_step_forecast_agrees_with_recursion(gjr_sample, model):
    fit = fit_volatility(gjr_sample, model)
    manual = one_step_variance(fit.params, fit.resid[-1], fit.cond_vol[-1] ** 2)
    assert fit.sigma_forecast**2 == pytest.approx(manual, rel=1e-10)


def test_fit_records_diagnostics(gjr_sample):
    fit = fit_volatility(gjr_sample, "GJR")
    assert fit.status == "ok" and fit.converged and fit.attempts == 1
    assert set(fit.params) == {"omega", "alpha", "gamma", "beta"}
    assert np.isfinite(fit.loglik)
    assert fit.persistence < 1.0
    # Loose sanity bound only (true values 0.05 / 0.08 / 0.88): fixed seed, deterministic.
    assert 0.0 <= fit.params["gamma"] < 0.3
    assert 0.7 < fit.params["beta"] < 1.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_arch_model(np.ones(10), "EGARCH")


def test_failed_fit_is_marked_not_substituted():
    fit = fit_volatility(np.array([np.nan] * 50), "GARCH")
    assert fit.status == "failed"
    assert fit.attempts == 2
    assert np.isnan(fit.sigma_forecast)
    assert fit.error
