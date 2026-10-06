"""Tests for feature engineering."""

import json

import numpy as np
import pandas as pd
import pytest

from sgp4corr.features import (
    HISTORY_FEATURES, add_orbit_features, add_space_weather, flag_outliers,
    history_features, load_space_weather,
)
from sgp4corr.pairs import load_history
from test_pairs import record


def test_orbit_features_iss_altitude():
    """The ISS (15.5 rev/day, near-circular) orbits at roughly 400-420 km."""
    out = add_orbit_features(pd.DataFrame({"MEAN_MOTION": [15.50103472], "ECCENTRICITY": [0.0007417]}))
    assert 395 < out["PERIGEE_ALT_KM"][0] < out["APOGEE_ALT_KM"][0] < 430
    assert out["APOGEE_ALT_KM"][0] - out["PERIGEE_ALT_KM"][0] == pytest.approx(
        2 * 0.0007417 * (6378.135 + 412), rel=0.01
    )


@pytest.fixture
def weather(tmp_path):
    dates = pd.date_range("2025-01-01", periods=10)
    path = tmp_path / "SW-All.csv"
    pd.DataFrame({
        "DATE": dates.strftime("%Y-%m-%d"),
        "F10.7_OBS": np.arange(100.0, 110.0),
        "F10.7_OBS_LAST81": np.arange(200.0, 210.0),
        "F10.7_OBS_CENTER81": 999.0,
        "AP_AVG": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    }).to_csv(path, index=False)
    return load_space_weather(path)


def test_space_weather_uses_only_completed_days(weather):
    """An epoch during 5 Jan may only see values up to 4 Jan."""
    table = pd.DataFrame({"EPOCH_I": pd.to_datetime(
        ["2025-01-05 18:00", "2025-01-02 00:00", "2025-01-01 12:00"])}, index=[7, 3, 5])
    out = add_space_weather(table, weather)
    assert list(out.index) == [7, 3, 5]                       # row order preserved
    assert out.loc[7, "F107"] == 103.0 and out.loc[7, "F107_AVG81"] == 203.0
    assert out.loc[7, "AP"] == 12 and out.loc[7, "AP_AVG3"] == pytest.approx((6 + 9 + 12) / 3)
    assert out.loc[3, "F107"] == 100.0                        # midnight: 1 Jan is complete
    assert np.isnan(out.loc[5, "F107"])                       # nothing known yet


def test_space_weather_missing_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"DATE": ["2025-01-01"], "F10.7_OBS": [100.0]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="AP_AVG"):
        load_space_weather(path)


def test_history_features_gap_selection_and_values(tmp_path):
    days = [0.0, 0.4, 1.5, 1.8, 6.0]
    records = [record(d, mean_anomaly_shift=2.0 * d, BSTAR=str(1e-4 * (k + 1)))
               for k, d in enumerate(days)]
    path = tmp_path / "25544.json"
    path.write_text(json.dumps(records))
    history, _ = load_history(path)
    out = history_features(history, min_gap_days=1.0, max_gap_days=3.0)

    assert len(out) == 5
    # sets 0 and 1 have nothing >= 1 day older; set 4's nearest is 4.2 days older (> max gap)
    assert out[HISTORY_FEATURES].iloc[[0, 1, 4]].isna().all().all()
    # set 2 (day 1.5) uses set 1 (day 0.4); set 3 (day 1.8) also uses set 1
    assert out["PREV_GAP_DAYS"].iloc[2] == pytest.approx(1.1)
    assert out["PREV_GAP_DAYS"].iloc[3] == pytest.approx(1.4)
    assert out["BSTAR_DELTA"].iloc[2] == pytest.approx(1e-4)
    assert out["BSTAR_DELTA"].iloc[3] == pytest.approx(2e-4)
    row = out.iloc[2]
    assert row["PREV_DT_RATE"] == pytest.approx(row["PREV_DT_KM"] / 1.1**2)
    assert np.isfinite(row[["PREV_DR_KM", "PREV_DT_KM", "PREV_DN_KM"]].astype(float)).all()


def test_history_features_rejects_bad_gaps(tmp_path):
    path = tmp_path / "25544.json"
    path.write_text(json.dumps([record(), record(2.0)]))
    history, _ = load_history(path)
    with pytest.raises(ValueError):
        history_features(history, min_gap_days=2.0, max_gap_days=1.0)


def test_flag_outliers_is_relative_to_band_and_horizon():
    rng = np.random.default_rng(0)
    n = 4000
    table = pd.DataFrame({
        "BAND": np.repeat(["low", "high"], n // 2),
        "HORIZON_DAYS": rng.uniform(0, 2, n),
        "DR_KM": 0.0, "DN_KM": 0.0,
        "DT_KM": np.r_[rng.normal(0, 100, n // 2), rng.normal(0, 1, n // 2)],
    })
    table.loc[n - 1, "DT_KM"] = 50.0          # ordinary for "low", absurd for "high"
    flags = flag_outliers(table, quantile=0.99)
    assert flags.iloc[-1]
    assert flags.groupby(table["BAND"]).mean().between(0.005, 0.015).all()
    with pytest.raises(ValueError):
        flag_outliers(table, quantile=0)