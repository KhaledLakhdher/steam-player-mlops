"""Drift monitoring via Population Stability Index (PSI).

PSI is the standard industry metric for input drift: it measures how much a feature's
distribution has shifted from a frozen reference. Rule of thumb:
    PSI < 0.1   no significant shift
    0.1 - 0.2   moderate shift
    PSI > 0.2   significant shift  -> investigate / retrain

We compute PSI on the continuous features only (binary/one-hot columns don't bin meaningfully)
and flag drift when the worst feature crosses the threshold. Rolling our own keeps the project
dependency-light and shows the math; Evidently is an easy drop-in later for richer HTML reports.
"""
from __future__ import annotations

import numpy as np

import config


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between a reference and current 1-D sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:                        # (near-)constant feature: no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(ref, edges)[0] / len(ref)
    cur_frac = np.histogram(cur, edges)[0] / len(cur)
    eps = 1e-6
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def psi_report(reference_df, current_df, features=None, threshold=None) -> dict:
    """PSI per feature + the worst score + a drift flag."""
    features = features or config.DRIFT_FEATURES
    threshold = config.PSI_THRESHOLD if threshold is None else threshold
    # Player counts are heavy-tailed and span orders of magnitude across games, so PSI is computed
    # on a log scale — a multiplicative shift (e.g. a popularity surge) then shows up as a clear
    # location shift instead of being swamped by the between-game variance.
    per_feature = {
        f: psi(np.log1p(reference_df[f].values), np.log1p(current_df[f].values)) for f in features
    }
    max_psi = max(per_feature.values()) if per_feature else 0.0
    return {"per_feature": per_feature, "max_psi": max_psi, "drift": max_psi > threshold}
