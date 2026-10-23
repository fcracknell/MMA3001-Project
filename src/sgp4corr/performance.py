"""Performance and interpretation analysis of the trained corrections.

Produces the evidence for the report's optimisation/performance section and
its interpretation of the model:

* Runtime per prediction for SGP4 alone and for each correction, measured
  with ``time.perf_counter`` (best of several repeats).
* Model size and an estimate of the work per prediction (tree comparisons
  for gradient boosting, arithmetic operations for ridge).
* A ``cProfile`` profile of the data pipeline for one object, to locate the
  computational bottleneck.
* Permutation feature importance of the along-track model: how much the test
  error grows when each input is randomly shuffled.

Uses the saved predictions and models, so nothing is retrained.

Usage (from the repository root):
    python src/sgp4corr/performance.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pickle
import pstats
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):                      # allow `python src/sgp4corr/performance.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgp4corr.features import (
    FEATURE_COLUMNS, TARGET_COLUMNS, history_features, history_features_loop,
)
from sgp4corr.models import RIDGE_COLUMNS, add_physics_terms, predict_gb
from sgp4corr.pairs import build_pairs, load_history, make_satrec


def best_time(fn, repeats: int = 5) -> float:
    """Best wall-clock time of ``fn()`` in seconds over several runs, after one warm-up run.

    Args:
        fn: Function with no arguments.
        repeats: Number of timed runs.

    Returns:
        The shortest measured time in seconds.
    """
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def sgp4_timing(history_dir: Path, norad_ids, n_sets: int = 2000, n_epochs: int = 10_000) -> dict:
    """Time SGP4 initialisation and propagation.

    Args:
        history_dir: Directory of per-object JSON histories.
        norad_ids: Objects to draw element sets from.
        n_sets: Number of element sets to time initialisation and single calls on.
        n_epochs: Number of epochs for the vectorised (array) propagation timing.

    Returns:
        Microseconds per element-set initialisation, per single propagation
        call, and per epoch in a vectorised array call.

    Raises:
        ValueError: If no element sets can be loaded.
    """
    rows = []
    for norad_id in norad_ids:
        history, _ = load_history(Path(history_dir) / f"{norad_id}.json")
        rows.extend(history.to_dict("records"))
        if len(rows) >= n_sets:
            break
    rows = rows[:n_sets]
    if not rows:
        raise ValueError("No element sets found for SGP4 timing.")

    sats = [make_satrec(r) for r in rows]
    sat = sats[0]
    jd = np.full(n_epochs, sat.jdsatepoch)
    fr = sat.jdsatepochF + np.linspace(0.0, 7.0, n_epochs)
    return {
        "init_us": 1e6 * best_time(lambda: [make_satrec(r) for r in rows]) / len(rows),
        "single_call_us": 1e6 * best_time(lambda: [s.sgp4_tsince(1440.0) for s in sats]) / len(sats),
        "array_per_epoch_us": 1e6 * best_time(lambda: sat.sgp4_array(jd, fr)) / n_epochs,
    }


def gb_tree_stats(model) -> dict:
    """Size and expected work per prediction of one gradient-boosting model.

    The expected number of comparisons per prediction is the sum over trees of
    the average leaf depth, weighted by how many training samples reached each
    leaf (i.e. the average decision path length).

    Args:
        model: A fitted ``HistGradientBoostingRegressor``.

    Returns:
        Number of trees, total nodes, mean path length per tree and expected
        comparisons per prediction.
    """
    trees = [p for stage in model._predictors for p in stage]
    path_lengths = []
    for tree in trees:
        leaves = tree.nodes[tree.nodes["is_leaf"] == 1]
        weights = leaves["count"].astype(float)
        depth = leaves["depth"].astype(float)
        path_lengths.append(float(np.average(depth, weights=weights)) if weights.sum() else depth.mean())
    return {
        "trees": len(trees),
        "nodes": int(sum(len(t.nodes) for t in trees)),
        "mean_path_length": float(np.mean(path_lengths)),
        "comparisons_per_prediction": float(np.sum(path_lengths)),
    }


def model_timing(models: dict, test: pd.DataFrame, single_repeats: int = 200) -> pd.DataFrame:
    """Time each correction method and measure its size.

    Args:
        models: Dict from ``models.py`` (``{"RIDGE": {...}, "GB": {...}}``).
        test: Test rows with all model inputs.
        single_repeats: Number of single-row predictions for the latency measurement.

    Returns:
        One row per method with batch time per prediction (all three RTN
        components), single-prediction latency, pickled size, and the
        estimated work per prediction.
    """
    one = test.iloc[[0]]

    def predict(method, rows):
        for target in TARGET_COLUMNS:
            model = models[method][target]
            if method == "GB":
                predict_gb(model, rows)
            else:
                model.predict(rows[RIDGE_COLUMNS])

    results = []
    for method in models:
        batch = best_time(lambda: predict(method, test), repeats=3) / len(test)
        single = best_time(lambda: [predict(method, one) for _ in range(single_repeats)], 3) / single_repeats
        size = sum(len(pickle.dumps(models[method][t])) for t in TARGET_COLUMNS)
        row = {"METHOD": method, "BATCH_US_PER_PREDICTION": 1e6 * batch,
               "SINGLE_PREDICTION_MS": 1e3 * single, "SIZE_KB": size / 1024}
        if method == "GB":
            stats = [gb_tree_stats(models["GB"][t]) for t in TARGET_COLUMNS]
            row["TREES"] = sum(s["trees"] for s in stats)
            row["NODES"] = sum(s["nodes"] for s in stats)
            row["WORK_PER_PREDICTION"] = (
                f"~{sum(s['comparisons_per_prediction'] for s in stats):.0f} comparisons "
                f"+ {row['TREES']} additions"
            )
        else:
            n = len(RIDGE_COLUMNS)
            # per component: n imputation checks, n divisions (scaling), n multiply-adds
            row["WORK_PER_PREDICTION"] = f"~{3 * 4 * n} floating-point operations"
        results.append(row)
    return pd.DataFrame(results)


def profile_pipeline(history_dir: Path, norad_id: int, top: int = 15, original: bool = False) -> str:
    """Profile pair generation and history features for one object with ``cProfile``.

    Args:
        history_dir: Directory of per-object JSON histories.
        norad_id: Object to profile.
        top: Number of functions to list.
        original: Profile the original, unvectorised history features instead.

    Returns:
        The profiler report (top functions by cumulative time) as text.
    """
    features_fn = history_features_loop if original else history_features

    def run():
        history, _ = load_history(Path(history_dir) / f"{norad_id}.json")
        build_pairs(history)
        features_fn(history)

    profiler = cProfile.Profile()
    profiler.runcall(run)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(top)
    return stream.getvalue()


def history_features_speedup(history_dir: Path, norad_ids) -> dict:
    """Time the original and vectorised history-feature code on the same objects.

    Args:
        history_dir: Directory of per-object JSON histories.
        norad_ids: Objects to time on.

    Returns:
        Seconds for each version and the speed-up factor.
    """
    histories = [load_history(Path(history_dir) / f"{i}.json")[0] for i in norad_ids]
    slow = best_time(lambda: [history_features_loop(h) for h in histories], repeats=3)
    fast = best_time(lambda: [history_features(h) for h in histories], repeats=3)
    return {"original_s": slow, "vectorised_s": fast, "speedup": slow / fast,
            "element_sets": sum(len(h) for h in histories)}


def permutation_importance_gb(
    model,
    table: pd.DataFrame,
    target: str = "DT_KM",
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation importance of each input feature for one gradient-boosting model.

    Each base feature is shuffled across rows, the physics terms derived from
    it are recomputed, and the increase in mean absolute error (km) is
    recorded. Shuffling the horizon also changes the horizon scaling of the
    prediction, so its importance includes that effect.

    Args:
        model: A model from ``fit_gb`` for ``target``.
        table: Rows with features and the target (e.g. the test set).
        target: Residual column the model predicts.
        n_repeats: Number of shuffles per feature.
        seed: Random seed.

    Returns:
        One row per feature with the mean and standard deviation of the
        increase in MAE, sorted most important first.
    """
    rng = np.random.default_rng(seed)
    base = add_physics_terms(table[[*FEATURE_COLUMNS, target]])
    y = base[target].to_numpy()
    reference = np.mean(np.abs(y - predict_gb(model, base)))
    rows = []
    for feature in FEATURE_COLUMNS:
        increases = []
        for _ in range(n_repeats):
            shuffled = base[FEATURE_COLUMNS].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            shuffled = add_physics_terms(shuffled)
            increases.append(np.mean(np.abs(y - predict_gb(model, shuffled))) - reference)
        rows.append({"FEATURE": feature, "MAE_INCREASE_KM": float(np.mean(increases)),
                     "STD_KM": float(np.std(increases))})
    out = pd.DataFrame(rows).sort_values("MAE_INCREASE_KM", ascending=False).reset_index(drop=True)
    out.attrs["reference_mae_km"] = reference
    return out


def plot_importance(importance: pd.DataFrame, out_path: Path, title: str) -> None:
    """Horizontal bar chart of permutation importance.

    Args:
        importance: Output of ``permutation_importance_gb``.
        out_path: Where to save the PNG.
        title: Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(data["FEATURE"], data["MAE_INCREASE_KM"], xerr=data["STD_KM"], color="C3")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Increase in along-track MAE when shuffled (km)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the performance and importance analysis from the command line."""
    parser = argparse.ArgumentParser(description="Performance and feature-importance analysis.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="reports", type=Path)
    args = parser.parse_args()

    test = add_physics_terms(pd.read_parquet(args.data_dir / "processed" / "predictions.parquet"))
    models = joblib.load("models/models.joblib")
    history_dir = args.data_dir / "raw" / "gp_history"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("== SGP4 runtime ==")
    sgp4 = sgp4_timing(history_dir, test["NORAD_CAT_ID"].unique())
    say(f"Initialise from element set:  {sgp4['init_us']:8.2f} us")
    say(f"Propagate, one call:          {sgp4['single_call_us']:8.2f} us")
    say(f"Propagate, vectorised array:  {sgp4['array_per_epoch_us']:8.2f} us per epoch")

    say("\n== Correction runtime and size (all three RTN components) ==")
    timing = model_timing(models, test)
    say(timing.round(3).to_string(index=False))
    timing.to_csv(out_dir / "performance_models.csv", index=False)

    gb_us = float(timing.loc[timing["METHOD"] == "GB", "BATCH_US_PER_PREDICTION"].iloc[0])
    say(f"\nGB correction costs {gb_us / sgp4['single_call_us']:.1f}x one SGP4 call per prediction "
        f"(batched). History features need one extra SGP4 propagation per element set.")

    norad_id = int(test["NORAD_CAT_ID"].iloc[0])
    say(f"\n== cProfile: pairs + ORIGINAL history features for object {norad_id} ==")
    profile = profile_pipeline(history_dir, norad_id, original=True)
    (out_dir / "profile_pipeline_original.txt").write_text(profile)
    say("\n".join(profile.splitlines()[:22]))
    say(f"\n== cProfile: pairs + VECTORISED history features for object {norad_id} ==")
    profile = profile_pipeline(history_dir, norad_id)
    (out_dir / "profile_pipeline.txt").write_text(profile)
    say("\n".join(profile.splitlines()[:22]))

    ids = test["NORAD_CAT_ID"].unique()[:5]
    speed = history_features_speedup(history_dir, ids)
    say(f"\nHistory features, {speed['element_sets']:,} element sets from {len(ids)} objects: "
        f"original {speed['original_s']:.2f} s, vectorised {speed['vectorised_s']:.2f} s "
        f"({speed['speedup']:.1f}x faster)")

    say("\n== Permutation importance, along-track model (test set) ==")
    importance = permutation_importance_gb(models["GB"]["DT_KM"], test)
    say(f"Reference along-track MAE: {importance.attrs['reference_mae_km']:.3f} km")
    say(importance.round(4).to_string(index=False))
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    plot_importance(importance, out_dir / "figures" / "feature_importance.png",
                    "Permutation importance, along-track correction (test set)")

    (out_dir / "performance_output.txt").write_text("\n".join(lines))
    print(f"\nSaved performance_*.csv, profile_pipeline*.txt, feature_importance.csv, "
          f"figures/feature_importance.png and performance_output.txt in {out_dir}")


if __name__ == "__main__":
    main()