"""Tests for element-set loading and pair generation."""

import json

import numpy as np
import pandas as pd
import pytest
from sgp4.api import Satrec
from sgp4.exporter import export_omm

from sgp4corr.pairs import build_pairs, load_history, make_satrec, select_pair_indices

# ISS element set from the sgp4 library documentation.
LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"


def record(days_after=0.0, mean_anomaly_shift=0.0, **overrides):
    """One Space-Track-style record (string values) based on the ISS element set."""
    fields = export_omm(Satrec.twoline2rv(LINE1, LINE2), "ISS")
    epoch = pd.Timestamp(fields["EPOCH"]) + pd.Timedelta(days=days_after)
    fields["EPOCH"] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
    fields["MEAN_ANOMALY"] = (fields["MEAN_ANOMALY"] + mean_anomaly_shift) % 360.0
    out = {k: str(v) for k, v in fields.items()}
    out.update(overrides)
    return out


def write_history(tmp_path, records):
    path = tmp_path / "25544.json"
    path.write_text(json.dumps(records))
    return path


def test_make_satrec_matches_two_line_parser(tmp_path):
    """Our OMM loader must give the same orbit as the library's TLE parser."""
    history, _ = load_history(write_history(tmp_path, [record()]))
    ours = make_satrec(history.iloc[0])
    reference = Satrec.twoline2rv(LINE1, LINE2)
    for minutes in (0.0, 90.0, 1440.0, 10080.0):
        _, r1, v1 = ours.sgp4_tsince(minutes)
        _, r2, v2 = reference.sgp4_tsince(minutes)
        assert np.allclose(r1, r2, atol=1e-6)      # km (1 mm)
        assert np.allclose(v1, v2, atol=1e-9)


def test_load_history_drops_malformed_and_duplicates(tmp_path):
    records = [
        record(2.0),
        record(0.0),
        record(1.0, ECCENTRICITY="not-a-number"),
        record(1.5, ECCENTRICITY="1.2"),
        record(1.7, MEAN_MOTION=None),
        record(3.0, EPOCH="garbage"),
        record(2.0, CREATION_DATE="2030-01-01T00:00:00", BSTAR="0.5"),   # newer duplicate
    ]
    records[0]["CREATION_DATE"] = "2020-01-01T00:00:00"
    history, dropped = load_history(write_history(tmp_path, records))
    assert dropped == {"malformed": 4, "duplicate": 1}
    assert len(history) == 2
    assert history["EPOCH"].is_monotonic_increasing
    assert history["BSTAR"].iloc[1] == 0.5          # the newer duplicate was kept


def test_load_history_empty_file(tmp_path):
    history, dropped = load_history(write_history(tmp_path, []))
    assert history.empty and dropped == {"malformed": 0, "duplicate": 0}
    assert build_pairs(history)[0].empty


def test_select_pair_indices_respects_horizon_and_order():
    epochs = np.sort(np.random.default_rng(3).uniform(0, 30, 200))
    i, j = select_pair_indices(epochs, 7.0, 5, np.random.default_rng(0))
    horizon = epochs[j] - epochs[i]
    assert np.all(j > i) and np.all(horizon > 0) and np.all(horizon <= 7.0)
    counts = np.bincount(i)
    assert counts.max() <= 5
    # each pair of a given i falls in a different horizon bin
    bins = pd.DataFrame({"i": i, "bin": np.ceil(horizon / 1.4)})
    assert not bins.duplicated().any()


def test_mean_anomaly_offset_appears_as_along_track_error(tmp_path):
    """An object slightly ahead in its orbit must show a positive along-track error of the arc length."""
    shift_deg = 0.01
    base, shifted = record(), record(mean_anomaly_shift=shift_deg)
    ours = make_satrec(load_history(write_history(tmp_path, [base]))[0].iloc[0])
    ahead = make_satrec(load_history(write_history(tmp_path, [shifted]))[0].iloc[0])
    _, r0, v0 = ours.sgp4_tsince(0.0)
    _, r1, _ = ahead.sgp4_tsince(0.0)

    from sgp4corr.frames import to_rtn
    dr, dt, dn = to_rtn(np.subtract(r1, r0), r0, v0)
    expected = np.radians(shift_deg) * np.linalg.norm(r0)       # arc length, km
    assert dt == pytest.approx(expected, rel=0.01)              # ahead => positive along-track
    assert abs(dr) < 0.05 * expected and abs(dn) < 0.05 * expected


def test_build_pairs_matches_direct_computation(tmp_path):
    records = [record(d, mean_anomaly_shift=3.0 * d) for d in (0.0, 0.6, 1.3, 2.9, 4.4, 6.8, 9.5)]
    history, _ = load_history(write_history(tmp_path, records))
    pairs, n_err = build_pairs(history, max_horizon_days=7.0, pairs_per_tle=5, seed=1)

    assert n_err == 0 and len(pairs) > 0
    assert (pairs["EPOCH_J"] > pairs["EPOCH_I"]).all()
    assert pairs["HORIZON_DAYS"].between(0, 7.0, inclusive="right").all()
    assert np.allclose(
        pairs["HORIZON_DAYS"], (pairs["EPOCH_J"] - pairs["EPOCH_I"]) / pd.Timedelta(days=1)
    )

    by_epoch = history.set_index("EPOCH")
    for row in pairs.itertuples():
        sat_i = make_satrec(by_epoch.loc[row.EPOCH_I].to_dict() | {"EPOCH": row.EPOCH_I})
        sat_j = make_satrec(by_epoch.loc[row.EPOCH_J].to_dict() | {"EPOCH": row.EPOCH_J})
        _, r_pred, _ = sat_i.sgp4_tsince(row.HORIZON_DAYS * 1440.0)
        _, r_ref, _ = sat_j.sgp4_tsince(0.0)
        distance = np.linalg.norm(np.subtract(r_pred, r_ref))
        assert np.linalg.norm([row.DR_KM, row.DT_KM, row.DN_KM]) == pytest.approx(distance, abs=1e-5)


def test_build_pairs_is_reproducible(tmp_path):
    records = [record(0.3 * k, mean_anomaly_shift=k) for k in range(40)]
    history, _ = load_history(write_history(tmp_path, records))
    a, _ = build_pairs(history, seed=7)
    b, _ = build_pairs(history, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_build_pairs_rejects_bad_arguments(tmp_path):
    history, _ = load_history(write_history(tmp_path, [record(), record(1.0)]))
    with pytest.raises(ValueError):
        build_pairs(history, max_horizon_days=0)
    with pytest.raises(ValueError):
        build_pairs(history, pairs_per_tle=0)