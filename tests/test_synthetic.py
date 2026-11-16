"""Synthetic GJR-GARCH Student-t generator."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garch_fhs.synthetic import (
    StressRegime,
    SyntheticConfig,
    generate,
    omega_path,
    persistence,
    simulate_gjr_path,
    standardized_t_scale,
    unconditional_variance,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMALL = SyntheticConfig(
    seed=1,
    burn_in=50,
    n_obs=300,
    stress_regimes=(StressRegime("s1", 100, 149), StressRegime("s2", 250, 259)),
)


def test_reproducible_with_fixed_seed():
    a = generate(SMALL)
    b = generate(SMALL)
    pd.testing.assert_frame_equal(a, b)
    c = generate(replace(SMALL, seed=2))
    assert not np.allclose(a["return_pct"], c["return_pct"])


def test_lengths_and_columns():
    df = generate(SMALL)
    assert len(df) == SMALL.n_obs
    assert list(df.columns) == ["date", "return_pct", "true_vol_pct", "stress_flag", "regime"]


def test_variance_positive_and_finite():
    df = generate(SMALL)
    assert np.all(np.isfinite(df["true_vol_pct"])) and np.all(df["true_vol_pct"] > 0)
    assert np.all(np.isfinite(df["return_pct"]))


def test_stress_intervals_exact_and_inclusive():
    df = generate(SMALL)
    flagged = np.flatnonzero(df["stress_flag"].to_numpy())
    expected = np.r_[100:150, 250:260]
    np.testing.assert_array_equal(flagged, expected)
    assert set(df.loc[100:149, "regime"]) == {"s1"}
    assert set(df.loc[250:259, "regime"]) == {"s2"}
    assert df.loc[99, "regime"] == "normal" and df.loc[150, "regime"] == "normal"


def test_omega_multiplier_applies_only_to_retained_stress_indices():
    om = omega_path(SMALL)
    assert om.size == SMALL.burn_in + SMALL.n_obs
    stressed = np.flatnonzero(om != SMALL.omega)
    np.testing.assert_array_equal(stressed, SMALL.burn_in + np.r_[100:150, 250:260])
    assert np.allclose(om[stressed], SMALL.omega * 9.0)


def test_baseline_stress_indices_match_specification():
    cfg = SyntheticConfig()  # defaults equal the documented baseline
    labels = generate(replace(cfg, burn_in=10))["regime"].to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels == "stress_1"), np.arange(1400, 1500))
    np.testing.assert_array_equal(np.flatnonzero(labels == "stress_2"), np.arange(2500, 2600))


def test_baseline_config_file_matches_documented_parameters():
    cfg_json = json.loads((PROJECT_ROOT / "configs" / "synthetic_baseline.json").read_text())
    cfg = SyntheticConfig.from_dict(cfg_json["synthetic"])
    assert (cfg.seed, cfg.omega, cfg.alpha, cfg.gamma, cfg.beta, cfg.nu) == (42, 0.02, 0.05, 0.08, 0.88, 8.0)
    assert (cfg.burn_in, cfg.n_obs, cfg.stress_omega_multiplier, cfg.mu) == (1000, 3500, 9.0, 0.0)
    assert cfg.stress_regimes == (StressRegime("stress_1", 1400, 1499), StressRegime("stress_2", 2500, 2599))


def test_hand_calculated_recursion_positive_and_negative_shocks():
    # omega=0.1, alpha=0.1, gamma=0.2, beta=0.5, sigma2_0=1, z = [1, -1, 2]
    # t=0: sigma2=1,                       eps=+1
    # t=1: sigma2=0.1+0.1*1+0.5*1 = 0.7    (positive shock: no gamma), eps=-sqrt(0.7)
    # t=2: sigma2=0.1+0.3*0.7+0.5*0.7=0.66 (negative shock: alpha+gamma)
    r, s2 = simulate_gjr_path(np.array([1.0, -1.0, 2.0]), np.full(3, 0.1), 0.1, 0.2, 0.5, 1.0)
    np.testing.assert_allclose(s2, [1.0, 0.7, 0.66])
    np.testing.assert_allclose(r, [1.0, -np.sqrt(0.7), 2 * np.sqrt(0.66)])


def test_unit_variance_student_t_scaling():
    assert standardized_t_scale(8.0) == pytest.approx(np.sqrt(0.75))
    # Large fixed-seed sample: deterministic, tolerance far wider than sampling error (~0.006).
    rng = np.random.default_rng(2024)
    z = rng.standard_t(8.0, size=400_000) * standardized_t_scale(8.0)
    assert np.var(z) == pytest.approx(1.0, abs=0.03)
    with pytest.raises(ValueError):
        standardized_t_scale(2.0)


def test_persistence_and_unconditional_variance():
    assert persistence(0.05, 0.08, 0.88) == pytest.approx(0.97)
    assert unconditional_variance(0.02, 0.05, 0.08, 0.88) == pytest.approx(0.02 / 0.03)


def test_dates_are_weekdays_only():
    dates = generate(SMALL)["date"]
    assert dates.is_monotonic_increasing and dates.is_unique
    assert (dates.dt.day_of_week < 5).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"omega": 0.0},
        {"alpha": -0.01},
        {"beta": -0.1},
        {"alpha": 0.05, "gamma": -0.1},
        {"nu": 2.0},
        {"alpha": 0.1, "gamma": 0.1, "beta": 0.86},  # persistence = 1.01
        {"stress_omega_multiplier": 0.0},
        {"n_obs": 0},
        {"stress_regimes": (StressRegime("x", 290, 310),)},
        {"stress_regimes": (StressRegime("a", 10, 20), StressRegime("b", 20, 30))},
        {"stress_regimes": (StressRegime("normal", 10, 20),)},
    ],
)
def test_invalid_parameters_rejected(changes):
    with pytest.raises(ValueError):
        validate_config(replace(SMALL, **changes))
