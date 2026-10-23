"""Feature engineering for the SGP4 residual-correction model.

Adds model inputs to the pairs table produced by ``pairs.py``. Every feature
uses only information available at the epoch of the starting element set *i*,
so nothing from the prediction window leaks into the model:

* Orbit geometry   - perigee and apogee altitude from the mean elements.
* Space weather    - solar flux (F10.7) and geomagnetic index (Ap) for the
  last complete day before epoch *i*, plus trailing averages.
* Recent history   - how wrong an earlier element set was when propagated to
  epoch *i*, and how much B* changed. This shows how badly drag has recently
  been mis-estimated for this object.

It also flags extreme residuals (``OUTLIER`` column) without deleting them.

Usage (from the repository root):
    python src/sgp4corr/features.py

Reads ``data/processed/pairs.parquet``, ``data/raw/gp_history/`` and
``data/raw/SW-All.csv``; writes ``data/processed/features.parquet``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):                      # allow `python src/sgp4corr/features.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgp4corr.frames import to_rtn
from sgp4corr.pairs import load_history, make_satrec

MU_KM3_S2 = 398600.8          # WGS72 gravitational parameter
EARTH_RADIUS_KM = 6378.135    # WGS72 equatorial radius

# Space weather columns read from the CelesTrak file.
SW_COLUMNS = ["DATE", "F10.7_OBS", "F10.7_OBS_LAST81", "AP_AVG"]

HISTORY_FEATURES = ["PREV_GAP_DAYS", "PREV_DR_KM", "PREV_DT_KM", "PREV_DN_KM",
                    "PREV_DT_RATE", "BSTAR_DELTA"]
WEATHER_FEATURES = ["F107", "F107_AVG81", "AP", "AP_AVG3"]

# Model inputs, in the order used for training.
FEATURE_COLUMNS = [
    "HORIZON_DAYS", "BSTAR", "MEAN_MOTION", "MEAN_MOTION_DOT", "ECCENTRICITY", "INCLINATION",
    "PERIGEE_ALT_KM", "APOGEE_ALT_KM", *WEATHER_FEATURES, *HISTORY_FEATURES,
]
TARGET_COLUMNS = ["DR_KM", "DT_KM", "DN_KM"]


def add_orbit_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add perigee and apogee altitude computed from mean motion and eccentricity.

    Args:
        table: Table with ``MEAN_MOTION`` (rev/day) and ``ECCENTRICITY``.

    Returns:
        Copy of ``table`` with ``PERIGEE_ALT_KM`` and ``APOGEE_ALT_KM`` added.
    """
    out = table.copy()
    n_rad_s = out["MEAN_MOTION"] * 2.0 * math.pi / 86400.0
    semi_major_axis = (MU_KM3_S2 / n_rad_s**2) ** (1.0 / 3.0)
    out["PERIGEE_ALT_KM"] = semi_major_axis * (1.0 - out["ECCENTRICITY"]) - EARTH_RADIUS_KM
    out["APOGEE_ALT_KM"] = semi_major_axis * (1.0 + out["ECCENTRICITY"]) - EARTH_RADIUS_KM
    return out


def load_space_weather(path: Path) -> pd.DataFrame:
    """Load daily space weather and build trailing-average features.

    Each returned row is stamped with the first date on which its values are
    fully known: values for day D are stamped D + 1, because the daily solar
    flux and Ap for day D are only complete once that day has ended.

    Args:
        path: Path to CelesTrak's ``SW-All.csv``.

    Returns:
        Table sorted by ``KNOWN_FROM`` with the ``WEATHER_FEATURES`` columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If an expected column is missing (the message lists what was found).
    """
    sw = pd.read_csv(path)
    missing = [c for c in SW_COLUMNS if c not in sw.columns]
    if missing:
        raise KeyError(f"Space weather file lacks {missing}; it has {list(sw.columns)}.")

    sw = sw[SW_COLUMNS].copy()
    sw["DATE"] = pd.to_datetime(sw["DATE"])
    sw = sw.sort_values("DATE").dropna(subset=["F10.7_OBS", "AP_AVG"])
    return pd.DataFrame({
        "KNOWN_FROM": (sw["DATE"] + pd.Timedelta(days=1)).astype("datetime64[us]"),
        "F107": sw["F10.7_OBS"].to_numpy(),
        "F107_AVG81": sw["F10.7_OBS_LAST81"].to_numpy(),      # trailing, not centred
        "AP": sw["AP_AVG"].to_numpy(),
        "AP_AVG3": sw["AP_AVG"].rolling(3, min_periods=1).mean().to_numpy(),
    }).reset_index(drop=True)


def add_space_weather(table: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent fully known space weather to each row.

    Args:
        table: Table with an ``EPOCH_I`` timestamp column.
        weather: Output of ``load_space_weather``.

    Returns:
        Copy of ``table`` with the ``WEATHER_FEATURES`` columns added (NaN
        where the epoch precedes the weather record).
    """
    order = table["EPOCH_I"].argsort(kind="stable")
    left = table.iloc[order].copy()
    left["_ROW"] = order.to_numpy()
    left["EPOCH_I"] = left["EPOCH_I"].astype("datetime64[us]")
    merged = pd.merge_asof(left, weather, left_on="EPOCH_I", right_on="KNOWN_FROM",
                           direction="backward")
    merged = merged.sort_values("_ROW").drop(columns=["_ROW", "KNOWN_FROM"])
    merged.index = table.index
    merged["EPOCH_I"] = table["EPOCH_I"]
    return merged


def history_features(
    history: pd.DataFrame,
    min_gap_days: float = 1.0,
    max_gap_days: float = 3.0,
) -> pd.DataFrame:
    """Compute, for each element set, how well an earlier one predicted it.

    For element set *i*, take the most recent earlier element set *p* that is
    at least ``min_gap_days`` older (and no more than ``max_gap_days``),
    propagate *p* to the epoch of *i* and express the difference in RTN. A
    large along-track value means drag has recently been mis-estimated.

    This is the vectorised version: propagations are collected first and all
    residuals are rotated into RTN in a single array operation (see
    ``history_features_loop`` for the original).

    Args:
        history: Clean history for one object from ``pairs.load_history``.
        min_gap_days: Minimum age of the earlier element set.
        max_gap_days: Maximum age of the earlier element set.

    Returns:
        Table with ``NORAD_CAT_ID``, ``EPOCH`` and the ``HISTORY_FEATURES``
        columns, one row per element set. Values are NaN where no earlier
        element set falls in the allowed gap or SGP4 reports an error.

    Raises:
        ValueError: If the gaps are not ``0 < min_gap_days <= max_gap_days``.
    """
    if not 0 < min_gap_days <= max_gap_days:
        raise ValueError("Require 0 < min_gap_days <= max_gap_days.")
    out = history[["NORAD_CAT_ID", "EPOCH"]].copy()
    for column in HISTORY_FEATURES:
        out[column] = np.nan
    if len(history) < 2:
        return out

    t = ((history["EPOCH"] - history["EPOCH"].iloc[0]) / pd.Timedelta(days=1)).to_numpy()
    prev = np.searchsorted(t, t - min_gap_days, side="right") - 1     # last index old enough
    gap = t - t[np.maximum(prev, 0)]
    usable = np.flatnonzero((prev >= 0) & (gap <= max_gap_days))
    if len(usable) == 0:
        return out

    sats = [make_satrec(row) for row in history.to_dict("records")]
    pred = [sats[prev[i]].sgp4_tsince(gap[i] * 1440.0) for i in usable]
    ref = [sats[i].sgp4_tsince(0.0) for i in usable]
    ok = np.array([p[0] == 0 and r[0] == 0 for p, r in zip(pred, ref)])
    usable = usable[ok]
    if len(usable) == 0:
        return out
    r_pred = np.array([p[1] for p, good in zip(pred, ok) if good])
    r_ref = np.array([r[1] for r, good in zip(ref, ok) if good])
    v_ref = np.array([r[2] for r, good in zip(ref, ok) if good])

    rtn = to_rtn(r_pred - r_ref, r_ref, v_ref)
    g = gap[usable]
    bstar = history["BSTAR"].to_numpy()
    out.iloc[usable, out.columns.get_indexer(HISTORY_FEATURES)] = np.column_stack(
        [g, rtn[:, 0], rtn[:, 1], rtn[:, 2], rtn[:, 1] / g**2, bstar[usable] - bstar[prev[usable]]]
    )
    return out


def history_features_loop(
    history: pd.DataFrame,
    min_gap_days: float = 1.0,
    max_gap_days: float = 3.0,
) -> pd.DataFrame:
    """Original, unvectorised version of ``history_features``.

    Kept for verification (the tests check both give identical results) and
    for the before/after timing in ``performance.py``. It rotates each
    residual into RTN with a separate call, which profiling showed to be the
    pipeline's bottleneck.

    Compute, for each element set, how well an earlier one predicted it.

    For element set *i*, take the most recent earlier element set *p* that is
    at least ``min_gap_days`` older (and no more than ``max_gap_days``),
    propagate *p* to the epoch of *i* and express the difference in RTN. A
    large along-track value means drag has recently been mis-estimated.

    Args:
        history: Clean history for one object from ``pairs.load_history``.
        min_gap_days: Minimum age of the earlier element set.
        max_gap_days: Maximum age of the earlier element set.

    Returns:
        Table with ``NORAD_CAT_ID``, ``EPOCH`` and the ``HISTORY_FEATURES``
        columns, one row per element set. Values are NaN where no earlier
        element set falls in the allowed gap or SGP4 reports an error.

    Raises:
        ValueError: If the gaps are not ``0 < min_gap_days <= max_gap_days``.
    """
    if not 0 < min_gap_days <= max_gap_days:
        raise ValueError("Require 0 < min_gap_days <= max_gap_days.")
    out = history[["NORAD_CAT_ID", "EPOCH"]].copy()
    for column in HISTORY_FEATURES:
        out[column] = np.nan
    if len(history) < 2:
        return out

    t = ((history["EPOCH"] - history["EPOCH"].iloc[0]) / pd.Timedelta(days=1)).to_numpy()
    prev = np.searchsorted(t, t - min_gap_days, side="right") - 1     # last index old enough
    gap = t - t[np.maximum(prev, 0)]
    usable = np.flatnonzero((prev >= 0) & (gap <= max_gap_days))

    sats = [make_satrec(row) for row in history.to_dict("records")]
    bstar = history["BSTAR"].to_numpy()
    values = np.full((len(history), len(HISTORY_FEATURES)), np.nan)
    for i in usable:
        err_p, r_pred, _ = sats[prev[i]].sgp4_tsince(gap[i] * 1440.0)
        err_i, r_ref, v_ref = sats[i].sgp4_tsince(0.0)
        if err_p or err_i:
            continue
        d_r, d_t, d_n = to_rtn(np.subtract(r_pred, r_ref), r_ref, v_ref)
        values[i] = [gap[i], d_r, d_t, d_n, d_t / gap[i] ** 2, bstar[i] - bstar[prev[i]]]
    out[HISTORY_FEATURES] = values
    return out


def flag_outliers(
    table: pd.DataFrame,
    quantile: float = 0.999,
    bin_days: float = 1.0,
) -> pd.Series:
    """Flag pairs whose total error is extreme for their altitude band and horizon.

    Error size depends strongly on altitude and horizon, so a single threshold
    would remove genuine low-altitude errors and miss bad high-altitude ones.
    The threshold is therefore the given quantile of the 3-D error magnitude
    within each (altitude band, horizon bin) group.

    Args:
        table: Pairs table with ``DR_KM, DT_KM, DN_KM, HORIZON_DAYS`` and ``BAND``.
        quantile: Quantile above which a pair is flagged.
        bin_days: Width of the horizon bins in days.

    Returns:
        Boolean series aligned with ``table``; True marks an outlier.

    Raises:
        ValueError: If ``quantile`` is not in (0, 1].
    """
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")
    magnitude = np.sqrt((table[TARGET_COLUMNS] ** 2).sum(axis=1))
    horizon_bin = np.floor(table["HORIZON_DAYS"] / bin_days)
    threshold = magnitude.groupby([table["BAND"], horizon_bin], observed=True).transform(
        lambda x: x.quantile(quantile)
    )
    return (magnitude > threshold).rename("OUTLIER")


def build_features(
    pairs: pd.DataFrame,
    history_dir: Path,
    weather: pd.DataFrame,
    outlier_quantile: float = 0.999,
) -> pd.DataFrame:
    """Add all features and the outlier flag to the pairs table.

    Args:
        pairs: Pairs table from ``pairs.build_all`` (must include ``BAND``).
        history_dir: Directory of per-object JSON history files.
        weather: Output of ``load_space_weather``.
        outlier_quantile: Quantile passed to ``flag_outliers``.

    Returns:
        The pairs table with ``FEATURE_COLUMNS`` and ``OUTLIER`` added.
    """
    table = add_orbit_features(pairs)
    table = add_space_weather(table, weather)

    ids = table["NORAD_CAT_ID"].unique()
    per_object = []
    for n, norad_id in enumerate(ids, start=1):
        history, _ = load_history(Path(history_dir) / f"{norad_id}.json")
        per_object.append(history_features(history))
        if n % 25 == 0 or n == len(ids):
            print(f"Features: {n}/{len(ids)} objects processed.")
    hist = pd.concat(per_object, ignore_index=True).rename(columns={"EPOCH": "EPOCH_I"})
    hist["EPOCH_I"] = hist["EPOCH_I"].astype(table["EPOCH_I"].dtype)
    table = table.merge(hist, on=["NORAD_CAT_ID", "EPOCH_I"], how="left", validate="many_to_one")

    table["OUTLIER"] = flag_outliers(table, outlier_quantile).to_numpy()
    return table


def main() -> None:
    """Build the feature table from the command line."""
    parser = argparse.ArgumentParser(description="Build the model feature table.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data directory.")
    parser.add_argument("--outlier-quantile", default=0.999, type=float)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.data_dir / "processed" / "pairs.parquet")
    weather = load_space_weather(args.data_dir / "raw" / "SW-All.csv")
    table = build_features(pairs, args.data_dir / "raw" / "gp_history", weather,
                           args.outlier_quantile)

    out_path = args.data_dir / "processed" / "features.parquet"
    table.to_parquet(out_path, index=False)
    print(f"Saved {out_path}: {len(table):,} rows, {len(FEATURE_COLUMNS)} features.")
    print(f"Flagged as outliers: {int(table['OUTLIER'].sum()):,}")
    print("\nMissing values per feature (%):")
    print((table[FEATURE_COLUMNS].isna().mean() * 100).round(2).to_string())
    print("\nSpace weather range seen: F10.7 "
          f"{table['F107'].min():.0f}-{table['F107'].max():.0f}, Ap "
          f"{table['AP'].min():.0f}-{table['AP'].max():.0f}")


if __name__ == "__main__":
    main()