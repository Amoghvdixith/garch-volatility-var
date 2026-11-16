"""Empirical VaR/ES convention and plain historical simulation.

Expected values below were worked out by hand from the documented convention
(VaR = L_(ceil(n c)), ES = average of the worst n(1-c) mass with a fractional
boundary), not by calling the implementation.
"""

import numpy as np
import pytest

from garch_fhs.historical import (
    empirical_es,
    empirical_var,
    empirical_var_es,
    historical_simulation,
    monetary_losses,
    returns_to_losses,
)

LOSSES_1_TO_10 = np.arange(1.0, 11.0)


@pytest.mark.parametrize(
    "confidence, var, es",
    [
        (0.95, 10.0, 10.0),  # m = 0.5 -> only half of the worst loss
        (0.90, 9.0, 10.0),  # m = 1 exactly
        (0.80, 8.0, 9.5),  # m = 2 -> (10 + 9) / 2
        (0.75, 8.0, 9.2),  # m = 2.5 -> (10 + 9 + 0.5*8) / 2.5
    ],
)
def test_hand_calculated_small_array(confidence, var, es):
    assert empirical_var(LOSSES_1_TO_10, confidence) == pytest.approx(var)
    assert empirical_es(LOSSES_1_TO_10, confidence) == pytest.approx(es)


def test_order_of_input_does_not_matter():
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(LOSSES_1_TO_10)
    assert empirical_var(shuffled, 0.75) == pytest.approx(8.0)
    assert empirical_es(shuffled, 0.75) == pytest.approx(9.2)


def test_integer_tail_size_is_robust_to_float_error():
    # 1000 * 0.95 and 1000 * 0.99 must be treated as exact integers.
    losses = np.arange(1.0, 1001.0)
    assert empirical_var(losses, 0.95) == pytest.approx(950.0)
    assert empirical_es(losses, 0.95) == pytest.approx(975.5)  # mean(951..1000)
    assert empirical_var(losses, 0.99) == pytest.approx(990.0)
    assert empirical_es(losses, 0.99) == pytest.approx(995.5)  # mean(991..1000)


def test_fractional_es_boundary_mass_non_integer_tail():
    # n = 250, c = 0.99 -> m = 2.5 (typical rolling-window size)
    losses = np.arange(1.0, 251.0)
    assert empirical_var(losses, 0.99) == pytest.approx(248.0)
    assert empirical_es(losses, 0.99) == pytest.approx((250 + 249 + 0.5 * 248) / 2.5)


def test_ties_at_the_boundary():
    losses = [1.0, 2.0, 2.0, 2.0, 5.0]
    # c = 0.6: k = 3 -> VaR = 2; m = 2 -> ES = (5 + 2) / 2
    assert empirical_var(losses, 0.6) == pytest.approx(2.0)
    assert empirical_es(losses, 0.6) == pytest.approx(3.5)


def test_matches_rockafellar_uryasev_representation():
    # Independent formula: ES = VaR + E[(L - VaR)^+] / (1 - c) under equal weights.
    rng = np.random.default_rng(123)
    for n in (37, 250, 1000):
        losses = rng.standard_t(5, size=n)
        for c in (0.95, 0.975, 0.99):
            var = empirical_var(losses, c)
            ru = var + np.mean(np.maximum(losses - var, 0.0)) / (1.0 - c)
            assert empirical_es(losses, c) == pytest.approx(ru, rel=1e-12, abs=1e-12)


def test_es_not_below_var():
    rng = np.random.default_rng(7)
    for _ in range(50):
        losses = rng.normal(size=rng.integers(5, 400))
        for c in (0.95, 0.99):
            est = empirical_var_es(losses, c)
            assert est.es >= est.var - 1e-12


def test_negative_risk_is_not_clipped():
    losses = [-5.0, -4.0, -3.0, -2.0, -1.0]  # every scenario is a gain
    assert empirical_var(losses, 0.6) == pytest.approx(-3.0)
    assert empirical_es(losses, 0.6) == pytest.approx(-1.5)


def test_loss_sign_and_units():
    returns_pct = np.array([-2.0, 1.0, 0.5])  # -2.0 means a 2% log-return fall
    np.testing.assert_allclose(returns_to_losses(returns_pct), [2.0, -1.0, -0.5])
    est = historical_simulation(returns_pct, 0.6)
    # losses sorted: -1, -0.5, 2 ; k = ceil(1.8) = 2 -> VaR = -0.5
    assert est.var == pytest.approx(-0.5)
    assert est.n_scenarios == 3


def test_monetary_mapping_is_exact():
    out = monetary_losses([-10.0, 0.0, 5.0], notional=100.0)
    np.testing.assert_allclose(out, [100 * (1 - np.exp(-0.1)), 0.0, 100 * (1 - np.exp(0.05))])
    # Exact loss differs from the linearised notional * (-r/100)
    assert out[0] != pytest.approx(10.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_confidence_rejected(bad):
    with pytest.raises(ValueError):
        empirical_var(LOSSES_1_TO_10, bad)


def test_non_finite_losses_rejected():
    with pytest.raises(ValueError):
        empirical_es([1.0, np.nan, 2.0], 0.95)
    with pytest.raises(ValueError):
        empirical_var([], 0.95)
