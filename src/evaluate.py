"""
Evaluation utilities: survival ranking metrics + calibration diagnostics.

These are the numbers we used to make decisions locally *before* spending a
leaderboard submission. Concordance tells us if the ranking is right; the Brier
score and reliability curve tell us if the probabilities are trustworthy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from . import config
from .calibration import binary_target_by_horizon


def concordance_by_horizon(probs, event, time, horizons=config.HORIZONS) -> dict:
    """Harrell's C-index at each horizon (uses that horizon's hit-prob as risk)."""
    from sksurv.metrics import concordance_index_censored

    event = np.asarray(event).astype(bool)
    time = np.asarray(time, dtype=float)
    out = {}
    for j, h in enumerate(horizons):
        c = concordance_index_censored(event, time, probs[:, j])[0]
        out[f"c_index_{h}h"] = round(float(c), 5)
    out["c_index_mean"] = round(float(np.mean(list(out.values()))), 5)
    return out


def brier_by_horizon(probs, event, time, horizons=config.HORIZONS) -> dict:
    """Brier score at each horizon (lower is better). Measures calibration+sharpness."""
    out = {}
    vals = []
    for j, h in enumerate(horizons):
        y = binary_target_by_horizon(event, time, h)
        if len(np.unique(y)) < 2:
            out[f"brier_{h}h"] = None
            continue
        b = brier_score_loss(y, probs[:, j])
        out[f"brier_{h}h"] = round(float(b), 5)
        vals.append(b)
    out["brier_mean"] = round(float(np.mean(vals)), 5) if vals else None
    return out


def reliability_curve(probs_h, y_h, n_bins=10):
    """Return (mean_predicted, observed_rate) per bin for a reliability diagram."""
    probs_h = np.asarray(probs_h, dtype=float)
    y_h = np.asarray(y_h, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs_h, bins) - 1
    mean_pred, obs_rate = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(probs_h[mask].mean())
        obs_rate.append(y_h[mask].mean())
    return np.array(mean_pred), np.array(obs_rate)


def evaluate_all(probs, event, time, horizons=config.HORIZONS) -> dict:
    """Convenience: concordance + Brier in one dict for logging."""
    return {**concordance_by_horizon(probs, event, time, horizons),
            **brier_by_horizon(probs, event, time, horizons)}
