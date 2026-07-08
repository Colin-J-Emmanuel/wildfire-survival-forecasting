"""
Survival models used in the ensemble, each behind a common interface.

Every wrapper exposes:
    .fit(X, event, time)                 -> self
    .predict_hit_prob(X, horizons)       -> ndarray (n_samples, n_horizons)

where each predicted value is P(T <= horizon | x) = 1 - S(horizon | x): the
probability the fire hits an evacuation zone by that horizon. Having a single
interface is what lets `ensemble.py` blend them without special-casing.

Three complementary learners:
  - XGBoostCoxModel        (proportional hazards, strong ranking + interactions)
  - RandomSurvivalForestModel  (non-parametric, robust on small data)
  - GradientBoostedSurvivalModel (boosted survival trees, different bias than RSF)

Requires: xgboost, scikit-survival (sksurv), numpy. Optuna is optional and only
needed for the tuning helper at the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_structured_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build the (bool event, float time) structured array scikit-survival wants."""
    y = np.empty(len(event), dtype=[("event", bool), ("time", float)])
    y["event"] = event.astype(bool)
    y["time"] = time.astype(float)
    return y


def _survfuncs_to_hit_prob(surv_funcs, horizons) -> np.ndarray:
    """Evaluate a list of sksurv step-function survival curves at each horizon.

    Returns 1 - S(h) = P(hit by h). Values are clipped to [0, 1].
    """
    out = np.empty((len(surv_funcs), len(horizons)))
    for i, fn in enumerate(surv_funcs):
        for j, h in enumerate(horizons):
            out[i, j] = 1.0 - float(fn(h))
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. XGBoost Cox Proportional Hazards
# --------------------------------------------------------------------------- #
@dataclass
class XGBoostCoxModel:
    """XGBoost with the Cox partial-likelihood objective.

    XGBoost's `survival:cox` outputs a *relative risk* (exp of the log-hazard),
    not an absolute survival probability. To turn risk into P(hit by h) we fit a
    non-parametric **Breslow** baseline cumulative hazard H0(t) on the training
    data, then use the proportional-hazards form:
        S(t | x) = exp( -H0(t) * risk(x) )
        P(hit by t | x) = 1 - S(t | x)
    """
    params: dict = field(default_factory=lambda: {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "tree_method": "hist",
        "seed": config.RANDOM_STATE,
    })
    n_estimators: int = 400
    _booster: object = field(default=None, repr=False)
    _base_times: np.ndarray = field(default=None, repr=False)
    _base_cumhaz: np.ndarray = field(default=None, repr=False)

    def fit(self, X, event, time):
        import xgboost as xgb

        # XGBoost encodes censoring via the sign of the label: negative time =
        # censored, positive time = observed event.
        signed_time = np.where(event.astype(bool), time, -time)
        dtrain = xgb.DMatrix(np.asarray(X, dtype=float), label=signed_time)
        self._booster = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators)

        # Breslow baseline cumulative hazard from training risk scores.
        risk = np.exp(self._booster.predict(dtrain))
        self._fit_breslow_baseline(time, event, risk)
        return self

    def _fit_breslow_baseline(self, time, event, risk):
        order = np.argsort(time)
        t_sorted, e_sorted, r_sorted = time[order], event[order], risk[order]
        uniq_event_times = np.unique(t_sorted[e_sorted.astype(bool)])
        cumhaz, running = [], 0.0
        for t in uniq_event_times:
            d = np.sum((t_sorted == t) & e_sorted.astype(bool))   # ties
            at_risk = np.sum(t_sorted >= t)
            denom = r_sorted[t_sorted >= t].sum()
            running += d / (denom + 1e-12)
            cumhaz.append(running)
        self._base_times = uniq_event_times
        self._base_cumhaz = np.asarray(cumhaz)

    def _baseline_cumhaz_at(self, h: float) -> float:
        if self._base_times is None or len(self._base_times) == 0:
            return 0.0
        idx = np.searchsorted(self._base_times, h, side="right") - 1
        return 0.0 if idx < 0 else float(self._base_cumhaz[idx])

    def predict_hit_prob(self, X, horizons=config.HORIZONS) -> np.ndarray:
        import xgboost as xgb

        risk = np.exp(self._booster.predict(xgb.DMatrix(np.asarray(X, dtype=float))))
        out = np.empty((len(risk), len(horizons)))
        for j, h in enumerate(horizons):
            H0 = self._baseline_cumhaz_at(h)
            out[:, j] = 1.0 - np.exp(-H0 * risk)
        return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. Random Survival Forest
# --------------------------------------------------------------------------- #
@dataclass
class RandomSurvivalForestModel:
    params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "min_samples_leaf": 8,          # regularize hard on 316 rows
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": config.RANDOM_STATE,
    })
    _model: object = field(default=None, repr=False)

    def fit(self, X, event, time):
        from sksurv.ensemble import RandomSurvivalForest

        self._model = RandomSurvivalForest(**self.params)
        self._model.fit(np.asarray(X, dtype=float), to_structured_y(event, time))
        return self

    def predict_hit_prob(self, X, horizons=config.HORIZONS) -> np.ndarray:
        fns = self._model.predict_survival_function(np.asarray(X, dtype=float))
        return _survfuncs_to_hit_prob(fns, horizons)


# --------------------------------------------------------------------------- #
# 3. Gradient Boosted Survival Trees
# --------------------------------------------------------------------------- #
@dataclass
class GradientBoostedSurvivalModel:
    params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2,
        "subsample": 0.8,
        "random_state": config.RANDOM_STATE,
    })
    _model: object = field(default=None, repr=False)

    def fit(self, X, event, time):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        self._model = GradientBoostingSurvivalAnalysis(**self.params)
        self._model.fit(np.asarray(X, dtype=float), to_structured_y(event, time))
        return self

    def predict_hit_prob(self, X, horizons=config.HORIZONS) -> np.ndarray:
        fns = self._model.predict_survival_function(np.asarray(X, dtype=float))
        return _survfuncs_to_hit_prob(fns, horizons)


MODEL_REGISTRY = {
    "xgb_cox": XGBoostCoxModel,
    "rsf": RandomSurvivalForestModel,
    "gbst": GradientBoostedSurvivalModel,
}


# --------------------------------------------------------------------------- #
# Optuna hyperparameter tuning (optional)
# --------------------------------------------------------------------------- #
def tune_xgb_cox(X, event, time, n_trials: int = 50):
    """Tune XGBoost Cox with Optuna, maximizing cross-validated concordance.

    Returns the best params dict. Requires optuna + scikit-survival.
    """
    import optuna
    from sklearn.model_selection import KFold
    from sksurv.metrics import concordance_index_censored

    X = np.asarray(X, dtype=float)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "tree_method": "hist",
            "seed": config.RANDOM_STATE,
        }
        n_estimators = trial.suggest_int("n_estimators", 200, 600, step=50)

        cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        scores = []
        for tr, va in cv.split(X):
            m = XGBoostCoxModel(params=params, n_estimators=n_estimators)
            m.fit(X[tr], event[tr], time[tr])
            # Rank by 72h hit prob; concordance wants a risk score.
            risk = m.predict_hit_prob(X[va], horizons=(72,))[:, 0]
            c = concordance_index_censored(event[va].astype(bool), time[va], risk)[0]
            scores.append(c)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
