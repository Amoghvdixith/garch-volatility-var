"""Forecast alignment, no-look-ahead and common-sample handling."""

import numpy as np
import pandas as pd
import pytest

from garch_fhs import MODELS
from garch_fhs.forecasting import common_sample, forecast_from_window, run_rolling_forecasts
from garch_fhs.historical import historical_simulation
from garch_fhs.synthetic import StressRegime, SyntheticConfig, generate
from garch_fhs.volatility import FitResult, fit_volatility

WINDOW = 120
N_FC = 4
CONF = [0.95, 0.99]


@pytest.fixture(scope="module")
def data():
    cfg = SyntheticConfig(seed=3, burn_in=200, n_obs=WINDOW + 10, stress_regimes=(StressRegime("s", 121, 125),))
    return generate(cfg).set_index("date")


def run(df, **kw):
    return run_rolling_forecasts(df, WINDOW, CONF, MODELS, min_residuals=100, max_forecasts=N_FC, **kw)


@pytest.fixture(scope="module")
def baseline(data):
    return run(data)


def key_cols(fc):
    return fc[["forecast_date", "model", "confidence", "var_pct", "es_pct", "vol_forecast_pct"]].reset_index(drop=True)


def test_training_window_ends_strictly_before_forecast_date(baseline, data):
    fc = baseline.forecasts
    assert (fc["train_end"] < fc["forecast_date"]).all()
    pos = {d: i for i, d in enumerate(data.index)}
    for _, row in fc.iterrows():
        t = pos[row["forecast_date"]]
        assert pos[row["train_end"]] == t - 1
        assert pos[row["train_start"]] == t - WINDOW


def test_all_models_receive_identical_windows(baseline):
    fc = baseline.forecasts
    per_date = fc.groupby("forecast_date")[["train_start", "train_end"]].nunique()
    assert (per_date == 1).all().all()
    assert set(fc["model"]) == set(MODELS)
    assert len(fc) == N_FC * len(MODELS) * len(CONF)


def test_changing_future_observations_does_not_change_earlier_forecasts(baseline, data):
    perturbed = data.copy()
    cut = WINDOW + 2  # first forecast positions 120, 121 must be unaffected
    perturbed.iloc[cut:, perturbed.columns.get_loc("return_pct")] *= -5.0
    out = run(perturbed).forecasts
    early = baseline.forecasts["forecast_date"] < data.index[cut]
    pd.testing.assert_frame_equal(key_cols(baseline.forecasts[early]), key_cols(out[early.to_numpy()]))
    # sanity: later forecasts do change
    late = ~early
    assert not np.allclose(baseline.forecasts.loc[late, "var_pct"], out.loc[late.to_numpy(), "var_pct"])


def test_changing_realised_return_on_t_does_not_change_its_forecast(baseline, data):
    t = WINDOW + 1
    perturbed = data.copy()
    perturbed.iloc[t, perturbed.columns.get_loc("return_pct")] = -25.0
    out = run(perturbed, ).forecasts
    d = data.index[t]
    a = baseline.forecasts[baseline.forecasts["forecast_date"] == d]
    b = out[out["forecast_date"] == d]
    pd.testing.assert_frame_equal(key_cols(a), key_cols(b))
    assert (b["return_pct"] == -25.0).all() and (b["loss_pct"] == 25.0).all()
    assert (b.loc[b["status"].isin(["ok", "ok_retry"]), "exception"] == 1).all()


def test_hs_forecast_uses_training_window_only(baseline, data):
    first = baseline.forecasts.iloc[0]
    train = data["return_pct"].to_numpy()[:WINDOW]
    hs = baseline.forecasts[(baseline.forecasts["model"] == "HS") & (baseline.forecasts["forecast_date"] == first["forecast_date"])]
    for _, row in hs.iterrows():
        est = historical_simulation(train, row["confidence"])
        assert row["var_pct"] == pytest.approx(est.var)
        assert row["es_pct"] == pytest.approx(est.es)


def test_exception_flag_is_strict_and_consistent(baseline):
    fc = baseline.forecasts.dropna(subset=["var_pct"])
    np.testing.assert_array_equal(fc["exception"].astype(int), (fc["loss_pct"] > fc["var_pct"]).astype(int))
    assert (fc["es_pct"] >= fc["var_pct"] - 1e-12).all()


def test_forecast_from_window_never_sees_more_than_window():
    seen = []

    def spy_fit(train, model):
        seen.append(train.size)
        return fit_volatility(train, model)

    rng = np.random.default_rng(0)
    forecast_from_window(rng.standard_normal(150), MODELS, CONF, 100, fit_fn=spy_fit)
    assert seen == [150, 150]


def failing_on(bad_last_value):
    """Fit function that fails whenever the training window ends with ``bad_last_value``."""

    def fit_fn(train, model):
        if model == "GJR" and train[-1] == bad_last_value:
            return FitResult(model=model, status="failed", attempts=2, error="forced failure for test")
        return fit_volatility(train, model)

    return fit_fn


def test_failed_fit_recorded_and_excluded_from_common_sample(data):
    bad_value = data["return_pct"].iloc[WINDOW + 1]  # last training obs for forecast position WINDOW+2
    out = run(data, fit_fn=failing_on(bad_value))
    fc = out.forecasts
    bad_date = data.index[WINDOW + 2]
    gjr_bad = fc[(fc["forecast_date"] == bad_date) & (fc["model"] == "GJR-FHS")]
    assert (gjr_bad["status"] == "fit_failed").all()
    assert gjr_bad["var_pct"].isna().all() and gjr_bad["exception"].isna().all()
    rec = out.fits[(out.fits["forecast_date"] == bad_date) & (out.fits["model"] == "GJR-FHS")]
    assert rec["error"].iloc[0] == "forced failure for test"

    common, info = common_sample(fc)
    assert bad_date not in set(common["forecast_date"])
    assert info["n_dates_excluded"] == 1
    assert info["excluded_dates"] == [bad_date.strftime("%Y-%m-%d")]
    assert info["invalid_forecasts_by_model"] == {"GJR-FHS": 1}
    # Excluded for every model, not just the one that failed
    assert common.groupby("model").size().nunique() == 1
    assert len(common) == (N_FC - 1) * len(MODELS) * len(CONF)


def test_insufficient_residual_pool_is_marked(data):
    out = run_rolling_forecasts(data, WINDOW, CONF, ["GARCH-FHS"], min_residuals=WINDOW + 1, max_forecasts=1)
    assert (out.forecasts["status"] == "insufficient_residuals").all()
    assert out.forecasts["var_pct"].isna().all()


def test_regime_labels_carried_to_forecasts(baseline, data):
    fc = baseline.forecasts.drop_duplicates("forecast_date").set_index("forecast_date")
    np.testing.assert_array_equal(fc["regime"].to_numpy(), data.loc[fc.index, "regime"].to_numpy())
