"""Return preparation and CSV schema validation."""

import math

import numpy as np
import pandas as pd
import pytest

from garch_fhs.data import (
    SchemaError,
    extract_adjusted_close,
    load_returns_csv,
    log_returns_pct,
    prepare_frame,
)

IDX = pd.date_range("2020-01-01", periods=3)


@pytest.mark.parametrize(
    "columns",
    [
        ["Adj Close", "Close"],
        pd.MultiIndex.from_tuples([("Adj Close", "SPY"), ("Close", "SPY")]),
        pd.MultiIndex.from_tuples([("SPY", "Adj Close"), ("SPY", "Close")]),
    ],
)
def test_extract_adjusted_close_handles_yfinance_layouts(columns):
    raw = pd.DataFrame([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]], index=IDX, columns=columns)
    out = extract_adjusted_close(raw, "SPY")
    np.testing.assert_allclose(out.to_numpy(), [1.0, 2.0, 3.0])  # never the unadjusted Close


def test_extract_adjusted_close_missing_field_rejected():
    raw = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=IDX)
    with pytest.raises(ValueError, match="Adj Close"):
        extract_adjusted_close(raw, "SPY")


def test_log_returns_are_percentage_points():
    r = log_returns_pct([100.0, 110.0, 99.0])
    assert r[0] == pytest.approx(100 * math.log(1.1))  # ~9.531, not 0.0953
    assert r[1] == pytest.approx(100 * math.log(0.9))
    assert len(r) == 2


def test_one_percent_log_return_stored_as_one():
    r = log_returns_pct([100.0, 100.0 * math.exp(0.01)])
    assert r[0] == pytest.approx(1.0)


@pytest.mark.parametrize("prices", [[100.0, 0.0, 99.0], [100.0, -1.0], [100.0, np.nan, 101.0]])
def test_invalid_prices_rejected(prices):
    with pytest.raises(ValueError):
        log_returns_pct(prices)


def test_price_schema_and_sorting():
    raw = pd.DataFrame(
        {"Date": ["2020-01-03", "2020-01-01", "2020-01-02"], "Adjusted_Close": [121.0, 100.0, 110.0]}
    )
    out = prepare_frame(raw)
    assert out.schema == "price"
    assert out.was_sorted is True
    assert out.dropped_first_row is True
    assert list(out.frame.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    np.testing.assert_allclose(out.frame["return_pct"], [100 * math.log(1.1)] * 2)


def test_return_schema_preserves_values_and_extra_columns():
    raw = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "return_pct": [0.5, -1.25], "true_vol_pct": [1.0, 1.1]}
    )
    out = prepare_frame(raw)
    assert out.schema == "return"
    assert out.was_sorted is False
    np.testing.assert_allclose(out.frame["return_pct"], [0.5, -1.25])
    assert "true_vol_pct" in out.frame.columns


def test_duplicate_dates_rejected():
    raw = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "return_pct": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicate"):
        prepare_frame(raw)


def test_non_finite_returns_rejected():
    raw = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "return_pct": [0.1, np.inf]})
    with pytest.raises(ValueError):
        prepare_frame(raw)
    raw2 = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "return_pct": [0.1, None]})
    with pytest.raises(ValueError):
        prepare_frame(raw2)


def test_ambiguous_schema_rejected():
    raw = pd.DataFrame({"date": ["2020-01-01"], "adjusted_close": [1.0], "return_pct": [0.1]})
    with pytest.raises(SchemaError, match="ambiguous"):
        prepare_frame(raw)


@pytest.mark.parametrize(
    "columns",
    [{"date": ["2020-01-01"], "close": [1.0]}, {"timestamp": ["2020-01-01"], "return_pct": [0.1]}],
)
def test_missing_required_columns_rejected(columns):
    with pytest.raises(SchemaError):
        prepare_frame(pd.DataFrame(columns))


def test_unparseable_date_rejected():
    raw = pd.DataFrame({"date": ["2020-01-01", "not a date"], "return_pct": [0.1, 0.2]})
    with pytest.raises(ValueError):
        prepare_frame(raw)


def test_load_csv_from_path_with_spaces(tmp_path):
    folder = tmp_path / "folder with spaces"
    folder.mkdir()
    path = folder / "my prices.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "adjusted_close": [50.0, 51.0]}).to_csv(
        path, index=False
    )
    out = load_returns_csv(path)
    assert out.frame["return_pct"].iloc[0] == pytest.approx(100 * math.log(51 / 50))


def test_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_returns_csv(tmp_path / "absent.csv")
