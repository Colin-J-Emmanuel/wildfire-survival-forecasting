"""
End-to-end pipeline: raw data -> engineered features -> ensemble -> calibration
-> monotone, valid submission.csv.

Run it as a module:

    python -m src.pipeline --data-dir data --out submissions/submission.csv

The steps mirror the write-up in docs/methodology.md exactly, so the code and
the documentation never drift apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .calibration import PerHorizonCalibrator
from .ensemble import SurvivalEnsemble
from .evaluate import evaluate_all
from .features import add_engineered_features, engineered_feature_columns
from .models import MODEL_REGISTRY


# --------------------------------------------------------------------------- #
def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / config.TRAIN_FILE)
    test = pd.read_csv(data_dir / config.TEST_FILE)
    return train, test


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    """Engineer features on both splits and align the model-input columns."""
    train_fe = add_engineered_features(train)
    test_fe = add_engineered_features(test)

    feature_cols = [
        c for c in engineered_feature_columns(train_fe)
        if c in test_fe.columns                       # only features present in both
    ]
    X_train = train_fe[feature_cols].fillna(0.0).to_numpy(dtype=float)
    X_test = test_fe[feature_cols].fillna(0.0).to_numpy(dtype=float)
    event = train_fe[config.EVENT_COL].to_numpy()
    time = train_fe[config.TIME_COL].to_numpy(dtype=float)
    return X_train, X_test, event, time, feature_cols


def enforce_monotonicity(probs: np.ndarray) -> np.ndarray:
    """A fire only gets closer over time: prob_12h <= prob_24h <= ... <= prob_72h."""
    return np.maximum.accumulate(probs, axis=1)


def zero_out_impossible_hits(test_df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    """Hard prior from the data: no training fire ever hit from beyond ~5 km.

    For fires that are physically too far to reach a zone within a horizon (even
    at a generous spread speed), clamp that horizon's probability toward zero.
    This encodes the distant-fire insight as a guardrail on top of calibration.
    """
    probs = probs.copy()
    d = test_df.get("distance_to_zone_km")
    v = test_df.get("spread_rate_kmh")
    if d is None or v is None:
        return probs
    d = d.to_numpy(dtype=float)
    v = np.maximum(v.to_numpy(dtype=float), 1e-6)
    for j, h in enumerate(config.HORIZONS):
        # Max reachable distance in h hours, with a 2x safety margin on speed.
        unreachable = (2.0 * v * h) < (d - config.HIT_RADIUS_KM)
        probs[unreachable, j] = 0.0
    return enforce_monotonicity(probs)


# --------------------------------------------------------------------------- #
def run(data_dir: Path, out_path: Path, tune: bool = False, verbose: bool = True):
    train, test = load_data(data_dir)
    X_train, X_test, event, time, feature_cols = prepare_features(train, test)
    if verbose:
        print(f"[data] train={X_train.shape}  test={X_test.shape}  features={len(feature_cols)}")

    # Optional: tune XGBoost Cox with Optuna, feed best params into the ensemble.
    model_params = {}
    if tune:
        from .models import tune_xgb_cox
        best = tune_xgb_cox(X_train, event, time, n_trials=40)
        n_est = best.pop("n_estimators", 400)
        model_params["xgb_cox"] = {"params": {**MODEL_REGISTRY["xgb_cox"]().params, **best},
                                   "n_estimators": n_est}
        if verbose:
            print(f"[optuna] best xgb params: {best}")

    # Fit the cross-validated ensemble.
    ensemble = SurvivalEnsemble(model_params=model_params).fit(X_train, event, time)
    if verbose:
        print("[ensemble] per-horizon weights:")
        print(ensemble.weight_table().round(2).to_string())

    # Fit calibration on out-of-fold ensemble predictions of the TRAIN set.
    oof = ensemble._out_of_fold_predictions(X_train, event, time, config.HORIZONS)
    oof_blend = np.zeros((len(X_train), len(config.HORIZONS)))
    for j in range(len(config.HORIZONS)):
        stack = np.stack([oof[n][:, j] for n in ensemble.model_names], axis=1)
        oof_blend[:, j] = stack @ ensemble.weights_[:, j]

    calibrator = PerHorizonCalibrator().fit(oof_blend, event, time)
    if verbose:
        print(f"[calibration] method per horizon: {calibrator.summary()}")
        print(f"[local] pre-calibration : {evaluate_all(oof_blend, event, time)}")
        print(f"[local] post-calibration: {evaluate_all(calibrator.transform(oof_blend), event, time)}")

    # Predict on test, calibrate, apply guardrails, enforce monotonicity.
    test_probs = ensemble.predict_hit_prob(X_test, config.HORIZONS)
    test_probs = calibrator.transform(test_probs)
    test_probs = zero_out_impossible_hits(test, test_probs)
    test_probs = enforce_monotonicity(test_probs)

    submission = pd.DataFrame({config.ID_COL: test[config.ID_COL].to_numpy()})
    for j, h in enumerate(config.HORIZONS):
        submission[f"prob_{h}h"] = test_probs[:, j]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if verbose:
        print(f"[done] wrote {len(submission)} rows -> {out_path}")
    return submission


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="WiDS 2026 wildfire survival pipeline.")
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--out", type=Path, default=config.SUBMISSIONS_DIR / "submission.csv")
    p.add_argument("--tune", action="store_true", help="run Optuna tuning first")
    args = p.parse_args()
    run(args.data_dir, args.out, tune=args.tune)


if __name__ == "__main__":
    main()
