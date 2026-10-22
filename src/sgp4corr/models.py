"""Train and compare corrections to SGP4 predictions.

Three methods are compared on held-out objects and a held-out time period:

* ``SGP4``  - the baseline: no correction.
* ``RIDGE`` - a simple alternative: ridge regression on a handful of
  physics-motivated terms (e.g. recent along-track error rate x horizon^2).
* ``GB``    - the main model: gradient-boosted trees on all features, one
  model per RTN component.

Both learned methods are constrained so that the correction vanishes at zero
horizon, as it must physically (an element set has no propagation error at its
own epoch): every ridge term is proportional to the horizon and there is no
intercept, and the trees predict the residual *per day of horizon*, which is
then multiplied by the horizon.

Each model predicts the SGP4 residual (prediction minus reference). The
corrected prediction is the SGP4 position minus the predicted residual.

Usage (from the repository root):
    python src/sgp4corr/models.py

Reads ``data/processed/features.parquet``; writes
``data/processed/predictions.parquet`` (test rows with each method's
predicted residual), ``models/*.joblib`` and
``reports/hyperparameter_search.csv``, then prints the evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):                      # allow `python src/sgp4corr/models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgp4corr.features import FEATURE_COLUMNS, TARGET_COLUMNS

PHYSICS_TERMS = ["H2", "BSTAR_H2", "PREV_DR_H", "PREV_DT_H", "PREV_DN_H",
                 "PREV_RATE_H", "PREV_RATE_H2"]
# Every ridge term is zero at zero horizon, so the ridge correction is too.
RIDGE_COLUMNS = ["HORIZON_DAYS", "H2", "BSTAR_H2", "PREV_DR_H", "PREV_DT_H", "PREV_DN_H",
                 "PREV_RATE_H", "PREV_RATE_H2"]
# Shortest horizon used when scaling the tree target, to avoid dividing by ~0.
HORIZON_FLOOR_DAYS = 1.0 / 24.0
GB_COLUMNS = [*FEATURE_COLUMNS, *PHYSICS_TERMS]

RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]
DEFAULT_GB_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 31,
                     "min_samples_leaf": 50, "l2_regularization": 0.0}
GB_SEARCH_SPACE = {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63, 127],
                   "min_samples_leaf": [20, 50, 200], "l2_regularization": [0.0, 1.0, 10.0]}


def add_physics_terms(table: pd.DataFrame) -> pd.DataFrame:
    """Add interaction terms suggested by the physics of drag error.

    A constant error in the drag estimate produces an along-track error that
    grows with the square of the horizon, so the terms combine horizon^2 with
    B* and with the recently observed along-track error rate.

    Args:
        table: Feature table with ``HORIZON_DAYS``, ``BSTAR`` and ``PREV_DT_RATE``.

    Returns:
        Copy of ``table`` with the ``PHYSICS_TERMS`` columns added.
    """
    out = table.copy()
    h = out["HORIZON_DAYS"]
    out["H2"] = h**2
    out["BSTAR_H2"] = out["BSTAR"] * h**2
    out["PREV_DR_H"] = out["PREV_DR_KM"] * h
    out["PREV_DT_H"] = out["PREV_DT_KM"] * h
    out["PREV_DN_H"] = out["PREV_DN_KM"] * h
    out["PREV_RATE_H"] = out["PREV_DT_RATE"] * h
    out["PREV_RATE_H2"] = out["PREV_DT_RATE"] * h**2
    return out


def assign_splits(
    table: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    test_months: float = 3.0,
    seed: int = 0,
    val_months: float = 3.0,
) -> tuple[pd.Series, pd.Timestamp]:
    """Split pairs into train, validation and test sets by object and by time.

    Whole objects are assigned to one set, stratified by altitude band, so the
    validation and test sets contain only objects the models have never seen.
    The sets are also separated in time: training data ends where the
    ``val_months`` validation period begins, and validation ends where the
    final ``test_months`` test period begins. Separating validation in time
    matters: if it shared the training period, hyperparameter tuning and early
    stopping would reward models that memorise conditions of that period
    (e.g. through slowly varying space weather), which do not carry over to
    the future.

    Args:
        table: Feature table with ``NORAD_CAT_ID, BAND, EPOCH_I, EPOCH_J``.
        fractions: Fractions of objects for (train, validation, test).
        test_months: Length of the held-out test period at the end of the data.
        seed: Random seed for the object assignment.
        val_months: Length of the validation period just before the test period.

    Returns:
        Tuple of (series of ``"train" / "val" / "test" / "unused"`` aligned
        with ``table``, the cutoff timestamp that starts the test period).

    Raises:
        ValueError: If fractions do not sum to 1 or a band has fewer than three objects.
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1.")
    rng = np.random.default_rng(seed)
    objects = table[["NORAD_CAT_ID", "BAND"]].drop_duplicates().sort_values("NORAD_CAT_ID")
    role = {}
    for band, group in objects.groupby("BAND", observed=True):
        ids = rng.permutation(group["NORAD_CAT_ID"].to_numpy())
        if len(ids) < 3:
            raise ValueError(f"Band {band} has fewer than three objects; cannot split it.")
        n_test = max(1, round(len(ids) * fractions[2]))
        n_val = max(1, round(len(ids) * fractions[1]))
        role.update(dict.fromkeys(ids[:n_test], "test"))
        role.update(dict.fromkeys(ids[n_test:n_test + n_val], "val"))
        role.update(dict.fromkeys(ids[n_test + n_val:], "train"))

    cutoff = table["EPOCH_J"].max() - pd.Timedelta(days=30.44 * test_months)
    val_cutoff = cutoff - pd.Timedelta(days=30.44 * val_months)
    object_role = table["NORAD_CAT_ID"].map(role)
    split = pd.Series("unused", index=table.index, name="SPLIT")
    split[(object_role == "train") & (table["EPOCH_J"] < val_cutoff)] = "train"
    split[(object_role == "val") & (table["EPOCH_I"] >= val_cutoff)
          & (table["EPOCH_J"] < cutoff)] = "val"
    split[(object_role == "test") & (table["EPOCH_I"] >= cutoff)] = "test"
    return split, cutoff


def fit_ridge(train: pd.DataFrame, val: pd.DataFrame, target: str):
    """Fit the ridge baseline for one residual component.

    Missing history features are filled with zero (no recent error known).
    There is no intercept and the features are scaled but not centred, so the
    prediction is exactly zero at zero horizon. Without this constraint the
    intercept absorbs the mean of the heavy-tailed residuals and shifts every
    prediction, including short-horizon ones where SGP4 is already accurate.
    The regularisation strength is the candidate with the lowest validation
    mean absolute error.

    Args:
        train: Training rows (with physics terms).
        val: Validation rows.
        target: Target column, e.g. ``"DT_KM"``.

    Returns:
        Tuple of (fitted scikit-learn pipeline, chosen alpha).
    """
    best = (np.inf, None, None)
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                              StandardScaler(with_mean=False),
                              Ridge(alpha=alpha, fit_intercept=False))
        model.fit(train[RIDGE_COLUMNS], train[target])
        mae = np.mean(np.abs(val[target] - model.predict(val[RIDGE_COLUMNS])))
        if mae < best[0]:
            best = (mae, model, alpha)
    return best[1], best[2]


def horizon_scale(table: pd.DataFrame) -> np.ndarray:
    """Horizon in days, floored at ``HORIZON_FLOOR_DAYS``, used to scale the tree target."""
    return np.maximum(table["HORIZON_DAYS"].to_numpy(dtype=float), HORIZON_FLOOR_DAYS)


def predict_gb(model: HistGradientBoostingRegressor, table: pd.DataFrame) -> np.ndarray:
    """Residual in km predicted by a model from ``fit_gb``.

    The trees predict the residual per day of horizon; multiplying by the
    horizon converts it back to km and makes the correction shrink to
    (almost) zero at short horizons.

    Args:
        model: A model returned by ``fit_gb``.
        table: Rows with the ``GB_COLUMNS`` features.

    Returns:
        Predicted residual in km for each row.
    """
    return model.predict(table[GB_COLUMNS]) * horizon_scale(table)


def fit_gb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    params: dict | None = None,
    max_iter: int = 300,
    seed: int = 0,
):
    """Fit gradient-boosted trees for one residual component.

    Uses absolute-error loss, which is robust to the heavy-tailed residuals.
    The trees learn the residual divided by the horizon, weighted by the
    horizon, which is equivalent to minimising the absolute error in km while
    forcing the correction to scale with the horizon (use ``predict_gb``).
    The number of trees is chosen by early stopping on the validation set:
    the model is trained for ``max_iter`` trees, the validation error after
    each tree is evaluated, and the model is refitted with the best count.

    Args:
        train: Training rows (with physics terms).
        val: Validation rows.
        target: Target column, e.g. ``"DT_KM"``.
        params: Hyperparameters overriding ``DEFAULT_GB_PARAMS``.
        max_iter: Maximum number of trees.
        seed: Random seed.

    Returns:
        Tuple of (fitted model, validation MAE in km, number of trees used).
        Predict with ``predict_gb``, not ``model.predict``.
    """
    settings = {**DEFAULT_GB_PARAMS, **(params or {})}

    def make(n_trees):
        return HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=n_trees, early_stopping=False,
            random_state=seed, **settings,
        )

    scale = horizon_scale(train)
    y_scaled = train[target].to_numpy(dtype=float) / scale

    def fit(n_trees):
        return make(n_trees).fit(train[GB_COLUMNS], y_scaled, sample_weight=scale)

    model = fit(max_iter)
    y_val = val[target].to_numpy(dtype=float)
    val_scale = horizon_scale(val)
    curve = [np.mean(np.abs(y_val - p * val_scale)) for p in model.staged_predict(val[GB_COLUMNS])]
    best_iter = int(np.argmin(curve)) + 1
    if best_iter < max_iter:
        model = fit(best_iter)
    return model, float(curve[best_iter - 1]), best_iter


def search_gb_params(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = "DT_KM",
    n_candidates: int = 8,
    sample_rows: int = 150_000,
    max_iter: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Random search over gradient-boosting hyperparameters.

    To keep the search quick it uses a random subsample of the training rows
    and the along-track target only. The first candidate is always the default
    configuration, so the search can never return something worse than it.

    Args:
        train: Training rows.
        val: Validation rows.
        target: Target column used to score candidates.
        n_candidates: Number of configurations to try.
        sample_rows: Training rows to subsample.
        max_iter: Maximum trees per candidate.
        seed: Random seed.

    Returns:
        One row per candidate with its hyperparameters, validation MAE (km),
        trees used and fit time, sorted best first. ``VAL_MAE_SGP4_KM`` is the
        validation MAE with no correction, for comparison.
    """
    rng = np.random.default_rng(seed)
    sample = train.sample(n=min(sample_rows, len(train)), random_state=seed)
    candidates = [dict(DEFAULT_GB_PARAMS)]
    while len(candidates) < n_candidates:
        candidate = {k: v[rng.integers(len(v))] for k, v in GB_SEARCH_SPACE.items()}
        if candidate not in candidates:
            candidates.append(candidate)

    baseline = float(np.mean(np.abs(val[target])))
    rows = []
    for n, candidate in enumerate(candidates, start=1):
        start = time.perf_counter()
        _, mae, n_trees = fit_gb(sample, val, target, candidate, max_iter, seed)
        rows.append({**candidate, "VAL_MAE_KM": mae, "VAL_MAE_SGP4_KM": baseline, "TREES": n_trees,
                     "FIT_SECONDS": round(time.perf_counter() - start, 1)})
        print(f"Search {n}/{len(candidates)}: val MAE {mae:.4f} km "
              f"(uncorrected {baseline:.4f})  {candidate}")
    return pd.DataFrame(rows).sort_values("VAL_MAE_KM").reset_index(drop=True)


def train_and_predict(
    table: pd.DataFrame,
    n_candidates: int = 8,
    max_iter: int = 300,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run the whole modelling pipeline.

    Args:
        table: Feature table from ``features.py``.
        n_candidates: Hyperparameter configurations to try (1 = defaults only).
        max_iter: Maximum trees for the final models.
        seed: Random seed.

    Returns:
        Tuple of (test rows with ``PRED_<target>_<method>`` columns added,
        dict of fitted models keyed by method then target, search results).

    Raises:
        ValueError: If a split is empty.
    """
    table = add_physics_terms(table)
    split, cutoff = assign_splits(table, seed=seed)
    # Train on unflagged pairs only; validation and test keep everything.
    train = table[(split == "train") & ~table["OUTLIER"]]
    val = table[split == "val"]
    test = table[split == "test"].copy()
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"The {name} split is empty.")
        print(f"{name:>5}: {len(part):>9,} pairs from {part['NORAD_CAT_ID'].nunique():>3} objects")
    print(f"Validation period starts {val['EPOCH_I'].min():%Y-%m-%d}; "
          f"test period starts {cutoff:%Y-%m-%d}.")

    search = search_gb_params(train, val, n_candidates=n_candidates, seed=seed)
    best_params = {k: search.loc[0, k].item() for k in GB_SEARCH_SPACE}
    print(f"Best hyperparameters: {best_params}")

    models: dict = {"RIDGE": {}, "GB": {}}
    for target in TARGET_COLUMNS:
        models["RIDGE"][target], alpha = fit_ridge(train, val, target)
        test[f"PRED_{target}_RIDGE"] = models["RIDGE"][target].predict(test[RIDGE_COLUMNS])
        start = time.perf_counter()
        models["GB"][target], mae, n_trees = fit_gb(train, val, target, best_params, max_iter, seed)
        test[f"PRED_{target}_GB"] = predict_gb(models["GB"][target], test)
        sgp4_mae = np.mean(np.abs(val[target]))
        print(f"{target}: ridge alpha {alpha}; GB {n_trees} trees, val MAE {mae:.4f} km "
              f"vs uncorrected {sgp4_mae:.4f} km "
              f"({time.perf_counter() - start:.0f} s)")
    return test, models, search


def main() -> None:
    """Train all models, save predictions and print the evaluation."""
    parser = argparse.ArgumentParser(description="Train and compare SGP4 corrections.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--candidates", default=8, type=int, help="Hyperparameter candidates.")
    parser.add_argument("--max-iter", default=300, type=int, help="Maximum trees.")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    table = pd.read_parquet(args.data_dir / "processed" / "features.parquet")
    test, models, search = train_and_predict(table, args.candidates, args.max_iter, args.seed)

    out_path = args.data_dir / "processed" / "predictions.parquet"
    test.to_parquet(out_path, index=False)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(models, "models/models.joblib")
    Path("reports").mkdir(exist_ok=True)
    search.to_csv("reports/hyperparameter_search.csv", index=False)
    print(f"Saved {out_path}, models/models.joblib and reports/hyperparameter_search.csv\n")

    from sgp4corr.evaluate import report
    report(test, Path("reports"))


if __name__ == "__main__":
    main()