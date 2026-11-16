"""Evaluation metrics, rolling alignment and regime classification."""

import math

import numpy as np
import pandas as pd
import pytest

from garch_fhs.metrics import (
    classify_regimes,
    max_consecutive,
    qlike,
    quantile_loss,
    rolling_exception_rate,
    stress_table,
    summarize_group,
    volatility_rmse,
)


@pytest.mark.parametrize(
    "loss, var, c, expected",
    [
        (3.0, 2.0, 0.95, 0.95),  # exception: c * (L - VaR)
        (1.0, 2.0, 0.95, 0.05),  # no exception: (1 - c) * (VaR - L)
        (2.0, 2.0, 0.95, 0.0),
        (-1.0, 2.0, 0.99, 0.03),
        (5.0, 1.0, 0.99, 3.96),
    ],
)
def test_quantile_loss_examples(loss, var, c, expected):
    assert quantile_loss(loss, var, c) == pytest.approx(expected)


def test_quantile_loss_vectorised():
    out = quantile_loss(np.array([3.0, 1.0]), np.array([2.0, 2.0]), 0.95)
    np.testing.assert_allclose(out, [0.95, 0.05])


@pytest.mark.parametrize(
    "seq, expected",
    [([0, 1, 1, 0, 1, 1, 1, 0], 3), ([], 0), ([0, 0, 0], 0), ([1], 1), ([1, 0, 1], 1)],
)
def test_max_consecutive(seq, expected):
    assert max_consecutive(seq) == expected


def test_rolling_window_alignment():
    s = pd.Series([1, 0, 0, 1, 1], index=pd.date_range("2020-01-01", periods=5))
    out = rolling_exception_rate(s, 2)
    assert math.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:], [0.5, 0.0, 0.5, 1.0])
    assert out.index.equals(s.index)  # value at t uses t-1 and t, labelled t


def test_stress_classification_inclusive_bounds():
    regimes = [
        {"label": "covid_2020", "start_date": "2020-02-20", "end_date": "2020-04-30"},
        {"label": "year_2022", "start_date": "2022-01-01", "end_date": "2022-12-31"},
    ]
    dates = pd.to_datetime(["2020-02-19", "2020-02-20", "2020-04-30", "2020-05-01", "2022-06-01", "2023-01-02"])
    labels = classify_regimes(dates, regimes)
    assert list(labels) == ["normal", "covid_2020", "covid_2020", "normal", "year_2022", "normal"]


def test_overlapping_regimes_rejected():
    regimes = [
        {"label": "a", "start_date": "2020-01-01", "end_date": "2020-02-01"},
        {"label": "b", "start_date": "2020-01-15", "end_date": "2020-03-01"},
    ]
    with pytest.raises(ValueError):
        classify_regimes(pd.date_range("2020-01-01", periods=60), regimes)


def test_volatility_rmse_and_qlike():
    assert volatility_rmse([1.0, 2.0], [1.0, 4.0]) == pytest.approx(math.sqrt(2.0))
    assert qlike([1.0, 4.0], [1.0, 4.0]) == pytest.approx(0.0)
    assert qlike([1.0], [2.0]) == pytest.approx(2.0 - math.log(2.0) - 1.0)


def _frame(losses, vars_, es, regime="normal"):
    n = len(losses)
    loss = np.asarray(losses, float)
    v = np.asarray(vars_, float)
    return pd.DataFrame(
        {
            "forecast_date": pd.date_range("2020-01-01", periods=n),
            "model": "HS",
            "confidence": 0.95,
            "loss_pct": loss,
            "var_pct": v,
            "es_pct": np.asarray(es, float),
            "exception": (loss > v).astype(int),
            "quantile_loss": quantile_loss(loss, v, 0.95),
            "regime": regime,
        }
    )


def test_summarize_group_hand_example():
    g = _frame([3.0, 1.0, 2.5, 0.0], [2.0, 2.0, 2.0, 2.0], [2.5, 2.5, 2.5, 2.5])
    s = summarize_group(g, 0.95)
    assert s["n"] == 4 and s["exceptions"] == 2
    assert s["expected_exceptions"] == pytest.approx(0.2)
    assert s["exception_rate"] == pytest.approx(0.5)
    assert s["rate_difference"] == pytest.approx(0.45)
    assert s["mean_excess_loss_on_exceptions_pct"] == pytest.approx((1.0 + 0.5) / 2)
    assert s["mean_quantile_loss"] == pytest.approx((0.95 + 0.05 + 0.475 + 0.1) / 4)
    assert s["max_consecutive_exceptions"] == 1
    assert s["es_below_var_count"] == 0


def test_stress_table_flags_short_samples_and_pools_stress():
    df = pd.concat(
        [
            _frame([0.0] * 300, [2.0] * 300, [2.5] * 300, "normal"),
            _frame([3.0] * 10, [2.0] * 10, [2.5] * 10, "s1"),
            _frame([0.0] * 10, [2.0] * 10, [2.5] * 10, "s2"),
        ],
        ignore_index=True,
    )
    df["forecast_date"] = pd.date_range("2020-01-01", periods=len(df))
    out = stress_table(df).set_index("regime")
    assert out.loc["s1", "short_sample"] and "short sample" in out.loc["s1", "notes"]
    assert not out.loc["normal", "short_sample"]
    assert out.loc["all_stress", "n"] == 20 and out.loc["all_stress", "exceptions"] == 10
    assert out.loc["s1", "max_consecutive_exceptions"] == 10
