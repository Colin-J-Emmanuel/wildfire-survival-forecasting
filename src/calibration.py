"""
Per-horizon probability calibration — the highest-leverage step in the pipeline.

A survival model can *rank* fires well while still being badly *calibrated*: its
"0.7" may not correspond to a 70% real-world hit rate. That was exactly the
baseline's problem — it was systematically overconfident, especially for distant
fires. The competition rewards trustworthy absolute probabilities, so we
calibrate.

For each of the four horizons independently we:
  1. Turn the survival label into a binary "hit by horizon h?" target
     (event == 1 AND time <= h).
  2. Fit BOTH an isotonic regressor and a Platt (sigmoid) calibrator on
     out-of-fold predictions.
  3. Keep whichever gives the lower Brier score on held-out data.

Different horizons can end up with different calibrators — that's expected and
fine.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from . import config


def binary_target_by_horizon(event, time, horizon) -> np.ndarray:
    """1 if the fire actually hit a zone at or before ``horizon``, else 0.

    Note the censoring subtlety: a censored fire (event==0) is treated as "not
    hit within 72h", which is correct because the horizons never exceed 72h.
    """
    event = np.asarray(event).astype(bool)
    time = np.asarray(time, dtype=float)
    return ((event) & (time <= horizon)).astype(int)


class _PlattCalibrator:
    """Logistic (sigmoid) calibration on a single probability column."""
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, p, y):
        self.lr.fit(np.asarray(p).reshape(-1, 1), y)
        return self

    def transform(self, p):
        return self.lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


class _IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p, y):
        self.iso.fit(np.asarray(p), y)
        return self

    def transform(self, p):
        return self.iso.transform(np.asarray(p))


class PerHorizonCalibrator:
    """Chooses and applies the best calibrator for each horizon."""
    def __init__(self, horizons=config.HORIZONS):
        self.horizons = list(horizons)
        self.calibrators_ = {}          # horizon -> fitted calibrator
        self.method_ = {}               # horizon -> "isotonic" | "platt" | "none"

    def fit(self, oof_probs, event, time):
        """Fit on out-of-fold ensemble probabilities (n_samples, n_horizons)."""
        oof_probs = np.asarray(oof_probs, dtype=float)
        for j, h in enumerate(self.horizons):
            p = oof_probs[:, j]
            y = binary_target_by_horizon(event, time, h)
            self.calibrators_[h], self.method_[h] = self._pick_best(p, y)
        return self

    @staticmethod
    def _pick_best(p, y):
        # Degenerate target (all one class) -> skip calibration for this horizon.
        if len(np.unique(y)) < 2:
            return None, "none"

        p_tr, p_va, y_tr, y_va = train_test_split(
            p, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y
        )
        candidates = {
            "isotonic": _IsotonicCalibrator().fit(p_tr, y_tr),
            "platt": _PlattCalibrator().fit(p_tr, y_tr),
        }
        scored = {
            name: brier_score_loss(y_va, cal.transform(p_va))
            for name, cal in candidates.items()
        }
        best = min(scored, key=scored.get)
        # Refit the winner on all the data before returning.
        refit = (_IsotonicCalibrator() if best == "isotonic" else _PlattCalibrator()).fit(p, y)
        return refit, best

    def transform(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        out = probs.copy()
        for j, h in enumerate(self.horizons):
            cal = self.calibrators_.get(h)
            if cal is not None:
                out[:, j] = cal.transform(probs[:, j])
        return np.clip(out, 0.0, 1.0)

    def summary(self):
        """Which method won at each horizon (for the write-up)."""
        return {f"{h}h": self.method_[h] for h in self.horizons}
