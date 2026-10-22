"""Evaluate SGP4 corrections on the held-out test set.

Works from the predictions table written by ``models.py``. For every method
the remaining error is the SGP4 residual minus the method's predicted
residual; for plain SGP4 the predicted residual is zero.

Because the residuals are heavy-tailed, the median and 95th percentile are
reported alongside RMSE, which is dominated by a few very large errors.

Usage (from the repository root):
    python src/sgp4corr/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):                      # allow `python src/sgp4corr/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgp4corr.features import TARGET_COLUMNS

METHODS = ["SGP4", "RIDGE", "GB"]
HORIZON_EDGES = [0, 1, 3, 5, 7]


def remaining_error(predictions: pd.DataFrame, method: str) -> np.ndarray:
    """Error left after applying a method's correction.

    Args:
        predictions: Test rows with ``DR_KM, DT_KM, DN_KM`` and, for learned
            methods, ``PRED_<target>_<method>`` columns.
        method: ``"SGP4"`` (no correction) or a learned method name.

    Returns:
        Array of shape (n, 3): remaining radial, along-track, cross-track error in km.

    Raises:
        KeyError: If the method's prediction columns are missing.
    """
    error = predictions[TARGET_COLUMNS].to_numpy(dtype=float)
    if method == "SGP4":
        return error
    return error - predictions[[f"PRED_{t}_{method}" for t in TARGET_COLUMNS]].to_numpy(dtype=float)


def error_table(predictions: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    """Long table of per-pair along-track and total error for each method.

    Args:
        predictions: Test rows from ``models.py``.
        methods: Methods to include; defaults to those with predictions present.

    Returns:
        Table with ``NORAD_CAT_ID, BAND, HORIZON_DAYS, HORIZON_BIN, OUTLIER,
        METHOD, ALONG_KM`` (absolute along-track error) and ``TOTAL_KM``
        (3-D position error).
    """
    if methods is None:
        methods = [m for m in METHODS
                   if m == "SGP4" or f"PRED_{TARGET_COLUMNS[0]}_{m}" in predictions.columns]
    base = predictions[["NORAD_CAT_ID", "BAND", "HORIZON_DAYS", "OUTLIER"]].reset_index(drop=True)
    base["HORIZON_BIN"] = pd.cut(base["HORIZON_DAYS"], HORIZON_EDGES).astype(str)
    parts = []
    for method in methods:
        error = remaining_error(predictions, method)
        parts.append(base.assign(METHOD=method, ALONG_KM=np.abs(error[:, 1]),
                                 TOTAL_KM=np.linalg.norm(error, axis=1)))
    return pd.concat(parts, ignore_index=True)


def summarise(errors: pd.DataFrame, by: list[str], column: str = "TOTAL_KM") -> pd.DataFrame:
    """Median, 95th percentile and RMSE of an error column, per group and method.

    Args:
        errors: Output of ``error_table``.
        by: Grouping columns, e.g. ``["BAND", "HORIZON_BIN"]`` (may be empty).
        column: ``"TOTAL_KM"`` or ``"ALONG_KM"``.

    Returns:
        Table indexed by ``by`` with columns (statistic, method), plus the
        percentage reduction in the median relative to SGP4 for each method.
    """
    if not by:                                  # single overall row
        errors, by = errors.assign(SCOPE="all"), ["SCOPE"]
    grouped = errors.groupby([*by, "METHOD"], observed=True)[column]
    stats = pd.DataFrame({
        "median": grouped.median(),
        "p95": grouped.quantile(0.95),
        "rmse": grouped.apply(lambda x: float(np.sqrt(np.mean(x**2)))),
    }).unstack("METHOD")
    methods = [m for m in METHODS if m in errors["METHOD"].unique()]
    stats = stats.reindex(columns=pd.MultiIndex.from_product([["median", "p95", "rmse"], methods]))
    for method in methods:
        if method != "SGP4":
            stats[("median_cut_%", method)] = (
                100.0 * (1.0 - stats[("median", method)] / stats[("median", "SGP4")])
            )
    return stats


def bootstrap_median_reduction(
    errors: pd.DataFrame,
    method: str,
    column: str = "TOTAL_KM",
    n_resamples: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Confidence interval for a method's reduction in median error.

    Pairs from one object are strongly correlated, so whole objects are
    resampled with replacement rather than individual pairs.

    Args:
        errors: Output of ``error_table``.
        method: Learned method to compare with SGP4.
        column: Error column to use.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Tuple of (point estimate, 2.5th percentile, 97.5th percentile) of the
        percentage reduction in median error relative to SGP4.
    """
    def per_object(name):
        subset = errors[errors["METHOD"] == name]
        return {k: v.to_numpy() for k, v in subset.groupby("NORAD_CAT_ID")[column]}

    base, corrected = per_object("SGP4"), per_object(method)
    ids = np.array(sorted(base))

    def reduction(sample_ids):
        b = np.median(np.concatenate([base[i] for i in sample_ids]))
        c = np.median(np.concatenate([corrected[i] for i in sample_ids]))
        return 100.0 * (1.0 - c / b)

    rng = np.random.default_rng(seed)
    draws = [reduction(rng.choice(ids, size=len(ids), replace=True)) for _ in range(n_resamples)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(reduction(ids)), float(low), float(high)


def plot_error_growth(errors: pd.DataFrame, out_path: Path, bin_days: float = 0.5) -> None:
    """Plot median total error against horizon for each method, one panel per band.

    Args:
        errors: Output of ``error_table``.
        out_path: Where to save the PNG.
        bin_days: Width of the horizon bins in days.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def lower_edge(band):                      # sort "(299.999, 500.0]" numerically
        try:
            return float(str(band).strip("([").split(",")[0])
        except ValueError:
            return 0.0

    bands = sorted(errors["BAND"].unique(), key=lower_edge)
    styles = {"SGP4": ("0.35", "-"), "RIDGE": ("C0", "--"), "GB": ("C3", "-")}
    labels = {"SGP4": "SGP4", "RIDGE": "SGP4 + ridge", "GB": "SGP4 + gradient boosting"}
    centre = (np.floor(errors["HORIZON_DAYS"] / bin_days) + 0.5) * bin_days

    fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3.6), squeeze=False)
    for ax, band in zip(axes[0], bands):
        in_band = errors["BAND"] == band
        for method in [m for m in METHODS if m in errors["METHOD"].unique()]:
            rows = in_band & (errors["METHOD"] == method)
            curve = errors.loc[rows, "TOTAL_KM"].groupby(centre[rows]).median()
            colour, line = styles[method]
            ax.plot(curve.index, curve.values, color=colour, ls=line, label=labels[method])
        ax.set_yscale("log")
        ax.set_title(f"Perigee {band} km", fontsize=9)
        ax.set_xlabel("Horizon (days)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("Median position error (km)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def report(predictions: pd.DataFrame, out_dir: Path) -> None:
    """Print the evaluation and save the tables and figure.

    Args:
        predictions: Test rows from ``models.py``.
        out_dir: Directory for ``results_*.csv`` and ``figures/error_growth_by_band.png``.
    """
    pd.set_option("display.width", 200)
    errors = error_table(predictions)
    n_objects = predictions["NORAD_CAT_ID"].nunique()
    print(f"Test set: {len(predictions):,} pairs from {n_objects} unseen objects, "
          f"{int(predictions['OUTLIER'].sum())} flagged as outliers.\n")

    overall = summarise(errors, [])
    by_band = summarise(errors, ["BAND"])
    by_band_horizon = summarise(errors, ["BAND", "HORIZON_BIN"])
    along = summarise(errors, ["BAND"], column="ALONG_KM")
    clean = summarise(errors[~errors["OUTLIER"]], ["BAND"])

    print("Position error (km), all test pairs:")
    print(overall.round(3).to_string(), "\n")
    print("Position error (km) by altitude band:")
    print(by_band.round(3).to_string(), "\n")
    print("Median position error (km) by band and horizon:")
    print(by_band_horizon[["median", "median_cut_%"]].round(3).to_string(), "\n")

    for method in [m for m in METHODS[1:] if m in errors["METHOD"].unique()]:
        estimate, low, high = bootstrap_median_reduction(errors, method)
        print(f"{method}: median position error reduced by {estimate:.1f}% "
              f"(95% CI {low:.1f}% to {high:.1f}%, bootstrap over objects)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_band.to_csv(out_dir / "results_by_band.csv")
    by_band_horizon.to_csv(out_dir / "results_by_band_horizon.csv")
    along.to_csv(out_dir / "results_along_track_by_band.csv")
    clean.to_csv(out_dir / "results_by_band_no_outliers.csv")
    plot_error_growth(errors, out_dir / "figures" / "error_growth_by_band.png")
    print(f"\nSaved results_*.csv and figures/error_growth_by_band.png in {out_dir}")


def main() -> None:
    """Evaluate saved predictions from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate SGP4 corrections.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="reports", type=Path)
    args = parser.parse_args()
    report(pd.read_parquet(args.data_dir / "processed" / "predictions.parquet"), args.out_dir)


if __name__ == "__main__":
    main()
