"""Filtered historical simulation scenario construction and risk measures."""

import numpy as np
import pytest

from garch_fhs.filtered_historical import fhs_scenario_returns, fhs_var_es, standardized_residuals
from garch_fhs.historical import empirical_es, empirical_var


def test_standardized_residuals_drop_only_invalid_values():
    sr = standardized_residuals(np.array([1.0, -2.0, np.nan, 3.0]), np.array([1.0, 2.0, 1.0, 0.0]))
    np.testing.assert_allclose(sr.z, [1.0, -1.0])
    assert sr.n_input == 4 and sr.n_dropped == 2


def test_residuals_not_recentred_or_renormalised():
    resid = np.array([2.0, 4.0, 6.0])  # mean of z = 1.0, deliberately non-zero
    sr = standardized_residuals(resid, np.full(3, 2.0))
    np.testing.assert_allclose(sr.z, [1.0, 2.0, 3.0])


def test_scenario_scaling_has_no_extra_factor_100():
    out = fhs_scenario_returns(2.0, np.array([1.0, -1.0, 0.5]))
    np.testing.assert_allclose(out, [2.0, -2.0, 1.0])


def test_hand_calculated_fhs_var_es():
    # sigma=1.5, z=[-2,-1,0,1,2] -> losses -1.5 z = [3, 1.5, 0, -1.5, -3]
    # c=0.6: k=ceil(3)=3 -> VaR = 0 ; m = 2 -> ES = (3 + 1.5)/2 = 2.25
    est = fhs_var_es(1.5, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 0.6)
    assert est.var == pytest.approx(0.0)
    assert est.es == pytest.approx(2.25)
    assert est.n_scenarios == 5


def test_fhs_scales_linearly_with_sigma_and_uses_same_convention_as_hs():
    rng = np.random.default_rng(5)
    z = rng.standard_t(6, size=1000)
    for c in (0.95, 0.99):
        est = fhs_var_es(1.7, z, c)
        assert est.var == pytest.approx(1.7 * empirical_var(-z, c))
        assert est.es == pytest.approx(1.7 * empirical_es(-z, c))
        assert est.es >= est.var


def test_fhs_is_deterministic_no_monte_carlo():
    z = np.linspace(-3, 3, 801)
    np.random.seed(1)
    a = fhs_var_es(1.2, z, 0.99)
    np.random.seed(999)
    b = fhs_var_es(1.2, z, 0.99)
    assert a == b


def test_minimum_residual_pool_enforced():
    with pytest.raises(ValueError, match="at least"):
        fhs_var_es(1.0, np.ones(10), 0.95, min_residuals=11)


def test_non_positive_sigma_rejected():
    with pytest.raises(ValueError):
        fhs_scenario_returns(0.0, np.ones(3))
