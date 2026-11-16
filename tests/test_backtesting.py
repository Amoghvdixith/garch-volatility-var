"""Kupiec and Christoffersen coverage tests.

Expected likelihood ratios are written out with ``math.log`` directly from
the textbook formulas, independently of the xlogy-based implementation.
"""

import math

import numpy as np
import pytest
from scipy.stats import chi2

from garch_fhs.backtesting import christoffersen_tests, coverage_tests, kupiec_test, transition_counts


def test_kupiec_known_case():
    n, x, p = 250, 5, 0.01
    pi = x / n
    lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(1 - pi) - x * math.log(pi))
    out = kupiec_test(np.r_[np.ones(x), np.zeros(n - x)], p)
    assert out["statistic"] == pytest.approx(lr)
    assert out["p_value"] == pytest.approx(chi2.sf(lr, 1))
    assert out["status"] == "defined" and out["df"] == 1


def test_kupiec_zero_exceptions():
    out = kupiec_test(np.zeros(100), 0.05)
    assert out["statistic"] == pytest.approx(-2 * 100 * math.log(0.95))  # 0*log(0) handled
    assert 0 <= out["p_value"] <= 1


def test_kupiec_all_exceptions():
    out = kupiec_test(np.ones(10), 0.05)
    assert out["statistic"] == pytest.approx(-2 * 10 * math.log(0.05))
    assert 0 <= out["p_value"] <= 1


def test_kupiec_exact_nominal_rate_gives_zero_statistic():
    out = kupiec_test(np.r_[np.ones(5), np.zeros(95)], 0.05)
    assert out["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert out["p_value"] == pytest.approx(1.0)


def test_kupiec_empty_sample_undefined():
    out = kupiec_test(np.array([], dtype=int), 0.05)
    assert out["status"] == "undefined" and math.isnan(out["p_value"]) and out["note"]


SEQ = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])


def test_transition_counts_known_example():
    # pairs: 00,01,11,10,01,10,00,00,01
    assert transition_counts(SEQ) == {"n00": 3, "n01": 3, "n10": 2, "n11": 1}


def test_christoffersen_hand_calculation():
    n00, n01, n10, n11, p = 3, 3, 2, 1, 0.05
    pi01, pi11, pi1 = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 9
    l1 = n00 * math.log(1 - pi01) + n01 * math.log(pi01) + n10 * math.log(1 - pi11) + n11 * math.log(pi11)
    l0 = (n00 + n10) * math.log(1 - pi1) + (n01 + n11) * math.log(pi1)
    lp = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    ind, cc = christoffersen_tests(SEQ, p)
    assert ind["statistic"] == pytest.approx(-2 * (l0 - l1))
    assert cc["statistic"] == pytest.approx(-2 * (lp - l1))
    assert ind["df"] == 1 and cc["df"] == 2
    assert ind["p_value"] == pytest.approx(chi2.sf(-2 * (l0 - l1), 1))


def test_cc_decomposes_into_uc_on_transitions_plus_ind():
    rng = np.random.default_rng(1)
    e = (rng.random(500) < 0.06).astype(int)
    ind, cc = christoffersen_tests(e, 0.05)
    uc_transitions = kupiec_test(e[1:], 0.05)
    assert cc["statistic"] == pytest.approx(uc_transitions["statistic"] + ind["statistic"])


def test_zero_one_cells_handled_without_nan():
    # Exceptions never cluster: n11 = 0 -> pi11 = 0, needs 0*log(0) = 0.
    e = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    ind, cc = christoffersen_tests(e, 0.05)
    assert ind["status"] == "defined" and np.isfinite(ind["statistic"])
    assert "no consecutive exceptions" in ind["note"]


@pytest.mark.parametrize(
    "seq, reason",
    [
        (np.zeros(50, dtype=int), "exception state"),
        (np.r_[np.zeros(49, dtype=int), 1], "exception state"),
        (np.ones(20, dtype=int), "non-exception state"),
    ],
)
def test_degenerate_transition_tables_are_undefined(seq, reason):
    for res in christoffersen_tests(seq, 0.01):
        assert res["status"] == "undefined"
        assert math.isnan(res["statistic"]) and math.isnan(res["p_value"])
        assert reason in res["note"]


def test_p_values_in_unit_interval_when_defined():
    rng = np.random.default_rng(2)
    for _ in range(30):
        e = (rng.random(rng.integers(20, 400)) < rng.uniform(0.005, 0.2)).astype(int)
        for res in coverage_tests(e, 0.05):
            if res["status"] == "defined":
                assert 0.0 <= res["p_value"] <= 1.0
            else:
                assert res["note"]


def test_low_power_note_for_small_expected_counts():
    e = np.r_[np.zeros(99, dtype=int), 1]
    assert "low power" in kupiec_test(e, 0.01)["note"]
