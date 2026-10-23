"""Tests for the performance and importance analysis."""

import numpy as np
import pandas as pd

from sgp4corr.features import FEATURE_COLUMNS
from sgp4corr.models import add_physics_terms, fit_gb
from sgp4corr.performance import best_time, gb_tree_stats, permutation_importance_gb


def make_table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    table["HORIZON_DAYS"] = rng.uniform(0.05, 7.0, n)
    table["DT_KM"] = 3.0 * table["PERIGEE_ALT_KM"] * table["HORIZON_DAYS"] + rng.normal(0, 0.1, n)
    return add_physics_terms(table)


def test_best_time_is_positive_and_small():
    assert 0 < best_time(lambda: sum(range(100)), repeats=3) < 0.1


def test_tree_stats_are_consistent():
    table = make_table()
    model, _, _ = fit_gb(table, table, "DT_KM", {"max_leaf_nodes": 7}, max_iter=10)
    stats = gb_tree_stats(model)
    assert stats["trees"] == model.n_iter_
    assert stats["nodes"] <= stats["trees"] * 13           # at most 7 leaves -> 13 nodes per tree
    assert 0 < stats["mean_path_length"] <= 6
    assert np.isclose(stats["comparisons_per_prediction"], stats["mean_path_length"] * stats["trees"])


def test_permutation_importance_finds_the_informative_feature():
    table = make_table()
    model, _, _ = fit_gb(table, table, "DT_KM", max_iter=40)
    importance = permutation_importance_gb(model, table, n_repeats=2)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert set(importance["FEATURE"].head(2)) == {"PERIGEE_ALT_KM", "HORIZON_DAYS"}
    noise = importance[~importance["FEATURE"].isin(["PERIGEE_ALT_KM", "HORIZON_DAYS"])]
    assert noise["MAE_INCREASE_KM"].abs().max() < 0.1 * importance["MAE_INCREASE_KM"].iloc[0]