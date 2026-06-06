"""
Feature preprocessing that only the Isolation Forest sees. XGBoost and RF
keep getting the raw 59-dim vector; this just reshapes the IF's input.

Two steps:
  1. log1p the value-magnitude columns with very heavy tails (max/p99 over
     the threshold), plus a few columns we pin by hand.
  2. Drop columns that saturate at an Etherscan cap (too many training rows
     sitting on the same max value), again plus a pinned minimum set.

build_recipe() returns a plain dict describing all of this. Save it next to
the trained IF and apply_recipe_*() will reproduce the same transform at
inference time.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

LOG_RATIO_THRESHOLD = 100.0
SATURATION_THRESHOLD = 0.05

# Columns we always log-transform, even if the max/p99 heuristic wouldn't
# pick them up on its own.
FORCE_LOG_FEATURES: tuple[str, ...] = (
    "total Ether sent",
    "total ether received",
    "max value received",
    "max val sent",
    "ERC20 max val sent",
    "ERC20 max val rec",
    "avg val received",
    "avg val sent",
    "ERC20 avg val rec",
    "ERC20 avg val sent",
    "ERC20 total Ether received",
    "ERC20 total ether sent",
)

# Columns we always drop (they're capped by Etherscan, so the "max" is an
# artefact rather than a real value).
FORCE_DROP_FEATURES: tuple[str, ...] = (
    "Total ERC20 tnxs",
)


def build_recipe(
    X_legit: pd.DataFrame,
    log_ratio_threshold: float = LOG_RATIO_THRESHOLD,
    saturation_threshold: float = SATURATION_THRESHOLD,
    force_log: Iterable[str] = FORCE_LOG_FEATURES,
    force_drop: Iterable[str] = FORCE_DROP_FEATURES,
) -> dict:
    """Look at the legit-only training matrix and decide what to log/drop.

    X_legit is the FLAG==0 training rows. A column gets logged if its max/p99
    blows past log_ratio_threshold, dropped if too many rows pile up on the
    column max (> saturation_threshold), plus whatever's pinned in force_log /
    force_drop. Returns a JSON-serialisable dict with the decisions and the
    per-column stats behind them.
    """
    analysis = []
    auto_log: list[str] = []
    auto_drop: list[str] = []

    for col in X_legit.columns:
        series = X_legit[col]
        col_max = float(series.max())
        col_p99 = float(series.quantile(0.99))
        col_p95 = float(series.quantile(0.95))
        col_p50 = float(series.median())
        nunique = int(series.nunique())

        # tail ratio, guarding the p99==0 case
        if col_p99 > 0:
            ratio = col_max / col_p99
        else:
            ratio = float("inf") if col_max > 0 else 0.0

        # how many rows sit exactly on the max (cap detector)
        share_at_max = float((series == col_max).mean()) if col_max > 0 else 0.0

        if col_max > 0 and ratio > log_ratio_threshold:
            auto_log.append(col)
        if share_at_max > saturation_threshold and nunique > 1:
            auto_drop.append(col)

        analysis.append({
            "feature": col,
            "max": col_max,
            "p99": col_p99,
            "p95": col_p95,
            "p50": col_p50,
            "max_over_p99": ratio if ratio != float("inf") else "inf",
            "share_at_max": round(share_at_max, 4),
            "nunique": nunique,
        })

    log_set = sorted(set(auto_log) | {c for c in force_log if c in X_legit.columns})
    drop_set = sorted(set(auto_drop) | {c for c in force_drop if c in X_legit.columns})

    # If a feature appears in BOTH log and drop, dropping wins.
    log_set = [c for c in log_set if c not in drop_set]

    kept_feature_order = [c for c in X_legit.columns if c not in drop_set]

    return {
        "version": "phase2-v1",
        "log_ratio_threshold": log_ratio_threshold,
        "saturation_threshold": saturation_threshold,
        "log_features": log_set,
        "drop_features": drop_set,
        "kept_feature_order": kept_feature_order,
        "n_log": len(log_set),
        "n_drop": len(drop_set),
        "n_kept": len(kept_feature_order),
        "analysis": analysis,
    }


def apply_recipe_to_dataframe(X: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    """Apply the recipe to a whole dataframe (training side)."""
    X = X.copy()
    for feat in recipe["log_features"]:
        if feat in X.columns:
            X[feat] = np.log1p(X[feat].clip(lower=0.0))
    # selecting kept_feature_order also drops the dropped columns
    return X[recipe["kept_feature_order"]]


def build_rank_calibrator(training_raw_scores: np.ndarray) -> dict:
    """Build an ECDF-based calibrator from the IF's raw legit-training scores.

    The idea: map a raw anomaly score to "what fraction of training rows scored
    at or below this". That's monotone, stays in [0, 1] and doesn't saturate -
    an address more extreme than anything in training still lands just under 1.0
    rather than slamming into it. This is what replaced the old (q01, q99)
    clipping, which collapsed everything past the boundary to exactly 1.0.

    training_raw_scores is the 1-D array of -decision_function values on the
    legit-only set (higher = more anomalous). Returns a dict that
    apply_rank_calibrator can use later.
    """
    sorted_scores = np.sort(np.asarray(training_raw_scores, dtype=np.float64))
    return {
        "kind": "rank_ecdf",
        "n_training": int(sorted_scores.shape[0]),
        "sorted_anomaly_scores": sorted_scores.tolist(),
        "min": float(sorted_scores.min()),
        "max": float(sorted_scores.max()),
        "p50": float(np.percentile(sorted_scores, 50)),
        "p95": float(np.percentile(sorted_scores, 95)),
        "p99": float(np.percentile(sorted_scores, 99)),
    }


def apply_rank_calibrator(raw_anomaly: float, calibrator: dict) -> float:
    """Run a raw anomaly score through the training ECDF to get a [0, 1] value.

    Uses the midpoint of the two ranks (count of rows strictly below, count of
    rows at or below) so that ties don't all land on exactly 1.0. The result is
    clamped to [1/(2n), 1 - 1/(2n)], i.e. it never quite hits 0 or 1 - that
    little bit of headroom is deliberate.
    """
    sorted_scores = np.asarray(calibrator["sorted_anomaly_scores"], dtype=np.float64)
    n = int(calibrator["n_training"])
    if n <= 0:
        return 0.5
    x = float(raw_anomaly)
    left = int(np.searchsorted(sorted_scores, x, side="left"))
    right = int(np.searchsorted(sorted_scores, x, side="right"))
    rank = (left + right) / (2.0 * n)
    return float(np.clip(rank, 1.0 / (2.0 * n), 1.0 - 1.0 / (2.0 * n)))


def apply_recipe_to_vector(
    feature_vector: np.ndarray,
    feature_names: list[str],
    recipe: dict,
) -> np.ndarray:
    """Inference-time version of the recipe for a single vector.

    feature_vector is the raw 1-D vector in feature_names order. Returns the
    kept columns (in recipe order), log-transformed where the recipe says so.
    The dataframe version above is the same thing for whole training batches.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    log_set = set(recipe["log_features"])
    out = np.empty(len(recipe["kept_feature_order"]), dtype=np.float64)
    for i, name in enumerate(recipe["kept_feature_order"]):
        if name in name_to_idx:
            val = float(feature_vector[name_to_idx[name]])
        else:
            val = 0.0
        if name in log_set:
            val = float(np.log1p(max(val, 0.0)))
        out[i] = val
    return out
