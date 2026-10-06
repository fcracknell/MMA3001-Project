"""Build the training table of SGP4 prediction errors.

For each object, every element set *i* is propagated with SGP4 to the epoch of
several later element sets *j*. The later element set, evaluated at its own
epoch, is treated as the reference ("truth"). The position difference
(prediction minus reference) is expressed in the RTN frame of the reference.

Note that the reference is itself an SGP4 fit, so these residuals measure
element-set-to-element-set consistency, not true position error.

Usage (from the repository root):
    python src/sgp4corr/pairs.py

Reads ``data/objects.csv`` and ``data/raw/gp_history/``; writes
``data/processed/pairs.parquet`` and a checkpoint plot to
``reports/figures/checkpoint_error_growth.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec

if __package__ in (None, ""):                      # allow `python src/sgp4corr/pairs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgp4corr.frames import to_rtn

# Fields that must be present and numeric for an element set to be usable.
ELEMENT_FIELDS = [
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
]
# Element-set values of TLE i copied into the pairs table as raw features.
FEATURE_FIELDS = ["BSTAR", "MEAN_MOTION", "MEAN_MOTION_DOT", "ECCENTRICITY", "INCLINATION"]

_SGP4_EPOCH0 = pd.Timestamp("1949-12-31")          # sgp4init counts days from here
_DEG = math.pi / 180.0
_REV_PER_DAY_TO_RAD_PER_MIN = 2.0 * math.pi / 1440.0


def load_history(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one object's element-set history and remove unusable records.

    Records are dropped if any orbital element is missing or non-numeric, if
    the epoch cannot be parsed, or if the elements are physically invalid
    (eccentricity outside [0, 1) or non-positive mean motion). Where several
    records share an epoch, the most recently created one is kept.

    Args:
        path: Path to a ``<norad_id>.json`` file written by ``download.py``.

    Returns:
        Tuple of (clean history sorted by epoch, counts of dropped records
        keyed by ``"malformed"`` and ``"duplicate"``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    records = json.loads(Path(path).read_text())
    dropped = {"malformed": 0, "duplicate": 0}
    df = pd.DataFrame(records)
    if df.empty or "EPOCH" not in df.columns:
        return pd.DataFrame(columns=["NORAD_CAT_ID", "EPOCH", *ELEMENT_FIELDS]), dropped

    for column in ELEMENT_FIELDS:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce")
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], errors="coerce", format="ISO8601")

    valid = (
        df[["NORAD_CAT_ID", "EPOCH", *ELEMENT_FIELDS]].notna().all(axis=1)
        & df["ECCENTRICITY"].between(0.0, 1.0, inclusive="left")
        & (df["MEAN_MOTION"] > 0.0)
    )
    dropped["malformed"] = int((~valid).sum())
    df = df[valid]

    sort_columns = ["EPOCH", "CREATION_DATE"] if "CREATION_DATE" in df.columns else ["EPOCH"]
    df = df.sort_values(sort_columns)
    before = len(df)
    df = df.drop_duplicates("EPOCH", keep="last")
    dropped["duplicate"] = before - len(df)

    df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype(int)
    return df[["NORAD_CAT_ID", "EPOCH", *ELEMENT_FIELDS]].reset_index(drop=True), dropped


def make_satrec(row) -> Satrec:
    """Initialise an SGP4 satellite record from one element set.

    Args:
        row: Mapping or row with ``NORAD_CAT_ID``, ``EPOCH`` (a timestamp)
            and the fields in ``ELEMENT_FIELDS`` (angles in degrees, mean
            motion in revolutions per day, as published in OMM/TLE data).

    Returns:
        An initialised ``sgp4.api.Satrec`` using WGS72 constants, as TLEs assume.
    """
    epoch_days = (pd.Timestamp(row["EPOCH"]) - _SGP4_EPOCH0) / pd.Timedelta(days=1)
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",                                                   # improved mode
        int(row["NORAD_CAT_ID"]),
        float(epoch_days),
        float(row["BSTAR"]),
        float(row["MEAN_MOTION_DOT"]) * _REV_PER_DAY_TO_RAD_PER_MIN / 1440.0,
        float(row["MEAN_MOTION_DDOT"]) * _REV_PER_DAY_TO_RAD_PER_MIN / 1440.0**2,
        float(row["ECCENTRICITY"]),
        float(row["ARG_OF_PERICENTER"]) * _DEG,
        float(row["INCLINATION"]) * _DEG,
        float(row["MEAN_ANOMALY"]) * _DEG,
        float(row["MEAN_MOTION"]) * _REV_PER_DAY_TO_RAD_PER_MIN,
        float(row["RA_OF_ASC_NODE"]) * _DEG,
    )
    return sat


def select_pair_indices(
    epoch_days: np.ndarray,
    max_horizon_days: float,
    pairs_per_tle: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose which later element sets each element set is compared against.

    The horizon range (0, ``max_horizon_days``] is split into
    ``pairs_per_tle`` equal bins and one later element set is drawn at random
    from each non-empty bin, so that horizons are covered evenly rather than
    dominated by whatever spacing the catalogue happens to have.

    Args:
        epoch_days: Sorted, strictly increasing epochs in days.
        max_horizon_days: Longest prediction horizon to include.
        pairs_per_tle: Maximum number of pairs per starting element set.
        rng: Random generator (seeded by the caller for reproducibility).

    Returns:
        Tuple ``(i, j)`` of index arrays with ``epoch_days[j] > epoch_days[i]``.
    """
    edges = np.linspace(0.0, max_horizon_days, pairs_per_tle + 1)
    idx_i, idx_j = [], []
    for i, t0 in enumerate(epoch_days):
        # bounds[k] = first index with epoch > t0 + edges[k]; bin k is bounds[k]:bounds[k+1]
        bounds = np.searchsorted(epoch_days, t0 + edges, side="right")
        bounds[0] = max(bounds[0], i + 1)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi > lo:
                idx_i.append(i)
                idx_j.append(int(rng.integers(lo, hi)))
    return np.asarray(idx_i, dtype=int), np.asarray(idx_j, dtype=int)


def build_pairs(
    history: pd.DataFrame,
    max_horizon_days: float = 7.0,
    pairs_per_tle: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Compute SGP4 prediction residuals for one object.

    Args:
        history: Clean history from ``load_history`` (one object, sorted by epoch).
        max_horizon_days: Longest prediction horizon to include.
        pairs_per_tle: Maximum number of pairs per starting element set.
        seed: Seed for the random choice of pairs.

    Returns:
        Tuple of (pairs table, number of pairs dropped because SGP4 reported
        an error). The table has one row per pair with columns
        ``NORAD_CAT_ID, EPOCH_I, EPOCH_J, HORIZON_DAYS``, the residual
        ``DR_KM, DT_KM, DN_KM`` (prediction minus reference, in the
        reference's RTN frame) and the ``FEATURE_FIELDS`` of element set *i*.

    Raises:
        ValueError: If ``max_horizon_days`` or ``pairs_per_tle`` is not positive.
    """
    if max_horizon_days <= 0 or pairs_per_tle < 1:
        raise ValueError("max_horizon_days must be > 0 and pairs_per_tle >= 1.")
    columns = ["NORAD_CAT_ID", "EPOCH_I", "EPOCH_J", "HORIZON_DAYS",
               "DR_KM", "DT_KM", "DN_KM", *FEATURE_FIELDS]
    if len(history) < 2:
        return pd.DataFrame(columns=columns), 0

    epochs = history["EPOCH"].to_numpy()
    epoch_days = ((history["EPOCH"] - history["EPOCH"].iloc[0]) / pd.Timedelta(days=1)).to_numpy()
    sats = [make_satrec(row) for row in history.to_dict("records")]

    # Reference state of every element set at its own epoch.
    ref = [sat.sgp4_tsince(0.0) for sat in sats]
    ref_err = np.array([e for e, _, _ in ref])
    ref_r = np.array([r for _, r, _ in ref])
    ref_v = np.array([v for _, _, v in ref])

    idx_i, idx_j = select_pair_indices(
        epoch_days, max_horizon_days, pairs_per_tle, np.random.default_rng(seed)
    )
    if len(idx_i) == 0:
        return pd.DataFrame(columns=columns), 0
    horizon = epoch_days[idx_j] - epoch_days[idx_i]

    # Propagate each element set i to all of its target epochs in one call.
    pred_r = np.empty((len(idx_i), 3))
    pred_err = np.empty(len(idx_i), dtype=int)
    starts = np.flatnonzero(np.r_[True, np.diff(idx_i) != 0, True])
    for a, b in zip(starts[:-1], starts[1:]):
        sat = sats[idx_i[a]]
        err, r, _ = sat.sgp4_array(
            np.full(b - a, sat.jdsatepoch), sat.jdsatepochF + horizon[a:b]
        )
        pred_err[a:b], pred_r[a:b] = err, r

    ok = (pred_err == 0) & (ref_err[idx_j] == 0) & (ref_err[idx_i] == 0)
    idx_i, idx_j, horizon, pred_r = idx_i[ok], idx_j[ok], horizon[ok], pred_r[ok]
    if len(idx_i) == 0:
        return pd.DataFrame(columns=columns), int((~ok).sum())

    rtn = to_rtn(pred_r - ref_r[idx_j], ref_r[idx_j], ref_v[idx_j])

    pairs = pd.DataFrame({
        "NORAD_CAT_ID": history["NORAD_CAT_ID"].to_numpy()[idx_i],
        "EPOCH_I": epochs[idx_i],
        "EPOCH_J": epochs[idx_j],
        "HORIZON_DAYS": horizon,
        "DR_KM": rtn[:, 0],
        "DT_KM": rtn[:, 1],
        "DN_KM": rtn[:, 2],
    })
    for field in FEATURE_FIELDS:
        pairs[field] = history[field].to_numpy()[idx_i]
    return pairs, int((~ok).sum())


def build_all(
    objects: pd.DataFrame,
    history_dir: Path,
    max_horizon_days: float = 7.0,
    pairs_per_tle: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the pairs table for every object and print a cleaning summary.

    Args:
        objects: Final object list (``data/objects.csv``); needs
            ``NORAD_CAT_ID`` and, optionally, ``BAND`` and ``OBJECT_TYPE``.
        history_dir: Directory of per-object JSON history files.
        max_horizon_days: Longest prediction horizon to include.
        pairs_per_tle: Maximum number of pairs per starting element set.
        seed: Base random seed; each object uses ``seed + NORAD_CAT_ID``.

    Returns:
        Concatenated pairs table for all objects.
    """
    tables = []
    totals = {"element_sets": 0, "malformed": 0, "duplicate": 0, "sgp4_error": 0, "missing_file": 0}
    for n, norad_id in enumerate(objects["NORAD_CAT_ID"], start=1):
        path = Path(history_dir) / f"{norad_id}.json"
        if not path.exists():
            totals["missing_file"] += 1
            continue
        history, dropped = load_history(path)
        pairs, n_err = build_pairs(history, max_horizon_days, pairs_per_tle, seed + int(norad_id))
        totals["element_sets"] += len(history)
        totals["malformed"] += dropped["malformed"]
        totals["duplicate"] += dropped["duplicate"]
        totals["sgp4_error"] += n_err
        tables.append(pairs)
        if n % 25 == 0 or n == len(objects):
            print(f"Pairs: {n}/{len(objects)} objects processed.")

    result = pd.concat(tables, ignore_index=True)
    extra = [c for c in ("BAND", "OBJECT_TYPE") if c in objects.columns]
    if extra:
        result = result.merge(objects[["NORAD_CAT_ID", *extra]], on="NORAD_CAT_ID", how="left")

    print(f"Usable element sets: {totals['element_sets']:,}")
    print(f"Dropped: {totals['malformed']:,} malformed, {totals['duplicate']:,} duplicate-epoch "
          f"element sets; {totals['sgp4_error']:,} pairs with SGP4 errors; "
          f"{totals['missing_file']} objects with no history file.")
    print(f"Pairs built: {len(result):,}")
    return result


def plot_error_growth(pairs: pd.DataFrame, out_path: Path, bin_days: float = 0.5) -> pd.DataFrame:
    """Plot the checkpoint figure: SGP4 error against prediction horizon.

    Shows the median (solid) and 95th percentile (dashed) of the absolute
    radial, along-track and cross-track error in each horizon bin. Along-track
    error should be the largest component and grow fastest.

    Args:
        pairs: Pairs table from ``build_all``.
        out_path: Where to save the PNG.
        bin_days: Width of the horizon bins in days.

    Returns:
        The plotted statistics, indexed by bin-centre horizon in days.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = {"DR_KM": "Radial", "DT_KM": "Along-track", "DN_KM": "Cross-track"}
    centre = (np.floor(pairs["HORIZON_DAYS"] / bin_days) + 0.5) * bin_days
    grouped = pairs[list(components)].abs().groupby(centre.rename("HORIZON_DAYS"))
    stats = pd.concat({"median": grouped.median(), "p95": grouped.quantile(0.95)}, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (column, label), colour in zip(components.items(), ["C0", "C3", "C2"]):
        ax.plot(stats.index, stats[("median", column)], color=colour, marker="o", ms=3, label=label)
        ax.plot(stats.index, stats[("p95", column)], color=colour, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon (days)")
    ax.set_ylabel("Absolute error (km)")
    ax.set_title("SGP4 error vs horizon (solid: median, dashed: 95th percentile)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stats


def main() -> None:
    """Build the pairs table and checkpoint plot from the command line."""
    parser = argparse.ArgumentParser(description="Build the SGP4 residual pairs table.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data directory.")
    parser.add_argument("--max-horizon", default=7.0, type=float, help="Longest horizon (days).")
    parser.add_argument("--pairs-per-tle", default=5, type=int, help="Pairs per element set.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--figure", default="reports/figures/checkpoint_error_growth.png", type=Path)
    args = parser.parse_args()

    objects = pd.read_csv(args.data_dir / "objects.csv")
    pairs = build_all(objects, args.data_dir / "raw" / "gp_history",
                      args.max_horizon, args.pairs_per_tle, args.seed)

    out_path = args.data_dir / "processed" / "pairs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"Saved {out_path}")

    stats = plot_error_growth(pairs, args.figure)
    print(f"Saved {args.figure}")
    print("\nMedian absolute error (km) by horizon:")
    print(stats["median"].iloc[::2].round(3).to_string())


if __name__ == "__main__":
    main()