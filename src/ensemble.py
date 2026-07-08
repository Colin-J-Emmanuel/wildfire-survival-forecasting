"""
Cross-validated ensemble of the survival models.

Rather than guess blend weights (or, worse, tune them on the public
leaderboard), we search for the convex combination of model predictions that
maximizes an *out-of-fold* survival objective. Because no single model was best
at every horizon, weights are learned **per horizon**.

Pipeline:
  1. For each CV fold, fit every base model on train, predict on the holdout.
  2. Concatenate holdout predictions into out-of-fold (OOF) matrices.
  3. Search the simplex for per-horizon weights maximizing OOF concordance.
  4. Refit all base models on the full data for inference.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

from . import config
from .models import MODEL_REGISTRY


class SurvivalEnsemble:
    def __init__(self, model_names=("xgb_cox", "rsf", "gbst"), model_params=None):
        self.model_names = list(model_names)
        self.model_params = model_params or {}
        self.weights_ = None            # (n_models, n_horizons)
        self.fitted_models_ = {}

    # ------------------------------------------------------------------ #
    def _new_model(self, name):
        cls = MODEL_REGISTRY[name]
        return cls(**self.model_params.get(name, {}))

    # ------------------------------------------------------------------ #
    def fit(self, X, event, time, horizons=config.HORIZONS):
        X = np.asarray(X, dtype=float)
        oof = self._out_of_fold_predictions(X, event, time, horizons)
        self.weights_ = self._optimize_weights(oof, event, time, horizons)

        # Refit every base model on all the data for final inference.
        for name in self.model_names:
            self.fitted_models_[name] = self._new_model(name).fit(X, event, time)
        return self

    # ------------------------------------------------------------------ #
    def _out_of_fold_predictions(self, X, event, time, horizons):
        """Return {model_name: OOF array (n_samples, n_horizons)}."""
        cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        oof = {n: np.zeros((len(X), len(horizons))) for n in self.model_names}
        for tr, va in cv.split(X):
            for name in self.model_names:
                m = self._new_model(name).fit(X[tr], event[tr], time[tr])
                oof[name][va] = m.predict_hit_prob(X[va], horizons)
        return oof

    # ------------------------------------------------------------------ #
    def _optimize_weights(self, oof, event, time, horizons):
        """Grid-search the simplex per horizon; maximize OOF concordance."""
        n_models = len(self.model_names)
        weights = np.zeros((n_models, len(horizons)))
        grid = self._simplex_grid(n_models, step=0.1)

        for j, _h in enumerate(horizons):
            preds = np.stack([oof[n][:, j] for n in self.model_names], axis=1)  # (n, n_models)
            best_c, best_w = -np.inf, np.ones(n_models) / n_models
            for w in grid:
                blend = preds @ w
                c = concordance_index_censored(event.astype(bool), time, blend)[0]
                if c > best_c:
                    best_c, best_w = c, w
            weights[:, j] = best_w
        return weights

    @staticmethod
    def _simplex_grid(n_models, step=0.1):
        """All weight vectors on the simplex with the given step (sum to 1)."""
        ticks = int(round(1.0 / step))
        combos = []
        for raw in product(range(ticks + 1), repeat=n_models):
            if sum(raw) == ticks:
                combos.append(np.array(raw, dtype=float) / ticks)
        return combos

    # ------------------------------------------------------------------ #
    def predict_hit_prob(self, X, horizons=config.HORIZONS) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict_hit_prob().")
        X = np.asarray(X, dtype=float)
        per_model = {n: m.predict_hit_prob(X, horizons) for n, m in self.fitted_models_.items()}
        out = np.zeros((len(X), len(horizons)))
        for j in range(len(horizons)):
            blend = np.stack([per_model[n][:, j] for n in self.model_names], axis=1)
            out[:, j] = blend @ self.weights_[:, j]
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def weight_table(self, horizons=config.HORIZONS):
        """Human-readable weights for docs/README (model x horizon)."""
        import pandas as pd
        return pd.DataFrame(
            self.weights_,
            index=self.model_names,
            columns=[f"{h}h" for h in horizons],
        )
