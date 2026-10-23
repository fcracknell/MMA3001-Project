"""Tests for the split, the models and the evaluation."""

import numpy as np
import pandas as pd
import pytest

from sgp4corr.evaluate import bootstrap_median_reduction, error_table, remaining_error, summarise
from sgp4corr.features import FEATURE_COLUMNS
from sgp4corr.models import (
    GB_COLUMNS, add_physics_terms, assign_splits, predict_gb, train_and_predict,
)


@pytest.fixture(scope="module")
def table():
    """Synthetic feature table in which the along-track error is learnable.

    Along-track error = 2 x (recent error rate) x horizon^2 + noise, the
    radial and cross-track errors are pure noise.
    """
    rng = np.random.default_rng(0)
    n_objects, per_object = 40, 400
    n = n_objects * per_object
    data = {column: rng.normal(size=n) for column in FEATURE_COLUMNS}
    data["HORIZON_DAYS"] = rng.uniform(0.01, 7.0, n)
    table = pd.DataFrame(data)
    table["NORAD_CAT_ID"] = np.repeat(np.arange(n_objects), per_object)
    table["BAND"] = np.where(table["NORAD_CAT_ID"] % 2 == 0, "(300, 500]", "(500, 700]")
    start = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    table["EPOCH_I"] = start
    table["EPOCH_J"] = start + pd.to_timedelta(table["HORIZON_DAYS"], unit="D")
    table["DT_KM"] = 2.0 * table["PREV_DT_RATE"] * table["HORIZON_DAYS"] ** 2 + rng.normal(0, 0.5, n)
    table["DR_KM"] = rng.normal(0, 0.1, n)
    table["DN_KM"] = rng.normal(0, 0.1, n)
    table["OUTLIER"] = False
    table.loc[rng.choice(n, 20, replace=False), "PREV_DT_RATE"] = np.nan
    return table


def test_split_separates_objects_and_time(table):
    split, cutoff = assign_splits(table, seed=1)
    objects = {name: set(table.loc[split == name, "NORAD_CAT_ID"]) for name in ("train", "val", "test")}
    assert objects["train"] and objects["val"] and objects["test"]
    assert not objects["train"] & objects["test"]
    assert not objects["train"] & objects["val"]
    assert not objects["val"] & objects["test"]
    assert (table.loc[split.isin(["train", "val"]), "EPOCH_J"] < cutoff).all()
    # validation comes after all training data, so tuning cannot reward memorising a period
    assert table.loc[split == "train", "EPOCH_J"].max() <= table.loc[split == "val", "EPOCH_I"].min()
    assert (table.loc[split == "test", "EPOCH_I"] >= cutoff).all()
    # stratified: every band appears in the test set
    assert table.loc[split == "test", "BAND"].nunique() == 2
    pd.testing.assert_series_equal(split, assign_splits(table, seed=1)[0])


def test_split_rejects_bad_input(table):
    with pytest.raises(ValueError):
        assign_splits(table, fractions=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        assign_splits(table[table["NORAD_CAT_ID"] < 3])        # too few objects per band


def test_physics_terms(table):
    out = add_physics_terms(table.head(5))
    assert np.allclose(out["H2"], out["HORIZON_DAYS"] ** 2)
    assert np.allclose(out["BSTAR_H2"], out["BSTAR"] * out["HORIZON_DAYS"] ** 2)


def test_models_learn_a_known_relationship(table):
    predictions, models, search = train_and_predict(table, n_candidates=2, max_iter=60)
    assert set(models) == {"RIDGE", "GB"} and len(search) == 2
    assert search["VAL_MAE_KM"].is_monotonic_increasing

    errors = error_table(predictions)
    median = errors.groupby("METHOD")["ALONG_KM"].median()
    assert median["RIDGE"] < 0.3 * median["SGP4"]
    assert median["GB"] < 0.5 * median["SGP4"]
    # nothing to learn in the radial direction, so the correction must not make it much worse
    radial = {m: np.median(np.abs(remaining_error(predictions, m)[:, 0])) for m in ("SGP4", "GB")}
    assert radial["GB"] < 1.2 * radial["SGP4"]

    overall = summarise(errors, [], column="ALONG_KM")
    assert len(overall) == 1 and overall[("median", "SGP4")].iloc[0] == pytest.approx(median["SGP4"])
    stats = summarise(errors, ["BAND"], column="ALONG_KM")
    assert (stats[("median_cut_%", "RIDGE")] > 50).all()
    estimate, low, high = bootstrap_median_reduction(errors, "RIDGE", column="ALONG_KM", n_resamples=50)
    assert low <= estimate <= high and low > 50
    for statistic in ("p95", "rmse"):
        estimate, low, high = bootstrap_median_reduction(
            errors, "RIDGE", column="ALONG_KM", n_resamples=20, statistic=statistic)
        assert low <= estimate <= high


def test_remaining_error_for_sgp4_is_the_raw_residual():
    predictions = pd.DataFrame({"DR_KM": [1.0], "DT_KM": [-2.0], "DN_KM": [0.5],
                                "PRED_DR_KM_GB": [1.0], "PRED_DT_KM_GB": [-1.0], "PRED_DN_KM_GB": [0.0]})
    assert np.allclose(remaining_error(predictions, "SGP4"), [[1.0, -2.0, 0.5]])
    assert np.allclose(remaining_error(predictions, "GB"), [[0.0, -1.0, 0.5]])
    with pytest.raises(KeyError):
        remaining_error(predictions, "RIDGE")


def test_corrections_vanish_at_zero_horizon(table):
    """Physically, there is no propagation error at the element set's own epoch."""
    _, models, _ = train_and_predict(table, n_candidates=1, max_iter=30)
    rows = add_physics_terms(table.head(200))
    at_zero = rows.assign(HORIZON_DAYS=0.0)
    at_zero = add_physics_terms(at_zero)
    at_week = add_physics_terms(rows.assign(HORIZON_DAYS=7.0))
    from sgp4corr.models import RIDGE_COLUMNS
    for target, ridge in models["RIDGE"].items():
        assert np.allclose(ridge.predict(at_zero[RIDGE_COLUMNS]), 0.0)
    for target, gb in models["GB"].items():
        near_zero = np.abs(predict_gb(gb, at_zero)).mean()
        week = np.abs(predict_gb(gb, at_week)).mean()
        assert near_zero <= week / 100          # floor of 1 hour vs 7 days = 1/168