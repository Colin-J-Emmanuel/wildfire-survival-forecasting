# Methodology

This document explains *why* each part of the pipeline exists, not just what it does. The code that
implements it lives in [`../src/`](../src/) and follows this narrative step for step.

---

## 1. Framing: why survival analysis, not classification

The obvious approach is to train four binary classifiers ("will it hit within 12h? 24h? ..."). We didn't,
for two reasons:

1. **The labels are right-censored.** For fires that never reached a zone within 72h, we don't know they'd
   *never* hit — only that they hadn't by the end of the observation window. A classifier treats "censored"
   and "confirmed safe" identically and throws away that distinction. Survival models handle censoring natively.
2. **The horizons share structure.** A fire's 12h, 24h, 48h, and 72h risks come from one underlying process
   (how fast it's closing on a zone). Modeling the survival function once and reading off four horizons keeps
   the predictions **internally consistent and monotone**, which four independent classifiers would not
   guarantee.

So we model the survival function `S(t | x) = P(T > t | x)` and read each horizon as
`P(hit by h) = 1 - S(h | x)`.

---

## 2. The dataset constraint: 316 fires

With only ~316 events, the dominant risk is **overfitting**, not underfitting. That single fact drove most
choices:

- Shallow trees (`max_depth` 2-4) and strong regularization (`min_child_weight`, `min_samples_leaf`, L1/L2).
- **Cross-validation everywhere** — for tuning, for ensemble weights, and for calibration — so nothing is
  fit on the same data it's evaluated on.
- Preferring **robust, boring models blended together** over one clever high-variance model.

---

## 3. The core insight: distant-fire overconfidence

Error analysis on the baseline revealed the decisive problem:

> The baseline assigned real hit probability to fires **7-545 km** from any evacuation zone, even though
> **no fire in training ever hit a zone from beyond ~5 km.**

Mechanically, the baseline's hazard didn't fall off fast enough with distance, so faraway fires leaked
probability mass. This is both a *scoring* problem (wasted probability where the truth is ~0) and an
*operational* problem (crying wolf about fires that can't possibly reach you). We attacked it three ways:
distance-decay features, a physical reachability guardrail, and — most importantly — calibration.

---

## 4. Feature engineering (+20 features)

Implemented in [`../src/features.py`](../src/features.py). Five families:

| Family | Examples | Why |
|---|---|---|
| **Distance-decay** | `dist_exp_decay_5km`, `dist_inverse`, `within_hit_radius`, `is_distant_fire` | Let the model express "risk ≈ 0 beyond a few km" — the direct antidote to overconfidence. |
| **Distance × speed** | `hours_to_zone_naive`, `speed_x_proximity`, `reachable_by_{h}h` | A fast fire *near* a zone ≠ a fast fire *far* from one. Encode the interaction instead of hoping the model finds it. |
| **Directional** | `directed_speed`, `heading_toward_zone` | A fast fire spreading *away* from the zone is not a threat; wind alignment captures this. |
| **Growth dynamics** | `growth_rate_per_hour`, `fire_weather_energy` | Acceleration/energy signals from the 5h window that correlate with sustained spread. |
| **Composite risk** | `risk_score`, `risk_score_directed`, `gated_risk` | Bundle proximity + speed + direction into single interpretable indices. |

Each feature is a pure function of the raw inputs, so they're easy to test and easy to drop if a raw column
is missing in your copy of the data.

---

## 5. The ensemble

Implemented in [`../src/ensemble.py`](../src/ensemble.py). Three complementary survival learners:

| Model | Strength | Bias it contributes |
|---|---|---|
| **XGBoost Cox PH** | Sharp ranking, captures interactions | Proportional-hazards structure |
| **Random Survival Forest** | Very robust on small data, low variance | Non-parametric, bagged |
| **Gradient Boosted Survival Trees** | Boosted survival fit | Different error profile from RSF |

No single model won at every horizon, so **weights are learned per horizon** by searching the simplex for the
convex combination that maximizes **out-of-fold** concordance. Learning weights on OOF predictions (not the
leaderboard) is what keeps the blend honest.

> **Turning models into probabilities.** RSF and GBST give survival curves directly. XGBoost Cox gives only a
> *relative risk*, so we fit a non-parametric **Breslow** baseline cumulative hazard on the training data and
> combine them as `S(t|x) = exp(-H₀(t)·risk(x))`. See `XGBoostCoxModel._fit_breslow_baseline` in
> [`../src/models.py`](../src/models.py).

---

## 6. Hyperparameter optimization

Implemented in `tune_xgb_cox` in [`../src/models.py`](../src/models.py). We used **Optuna** (TPE sampler) to
maximize cross-validated concordance rather than hand-tuning. The search space is deliberately
conservative — small depths, real regularization ranges — because on 316 rows an unconstrained search will
happily overfit the CV folds too.

---

## 7. Per-horizon calibration (the biggest single win)

Implemented in [`../src/calibration.py`](../src/calibration.py). A model can rank fires perfectly and still be
badly *calibrated*. Since the task rewards trustworthy absolute probabilities, we calibrate each horizon
independently:

1. Convert the survival label to a binary "hit by horizon h?" target.
2. Fit **both** isotonic regression and **Platt (sigmoid)** scaling on out-of-fold predictions.
3. Keep whichever has the lower **Brier score** on held-out data — possibly a different choice per horizon.

Isotonic is flexible (any monotone mapping) but can overfit on little data; Platt is a rigid 2-parameter
sigmoid that's safer when data is thin. Letting each horizon pick its winner gets the best of both. This step
is what corrected the distant-fire overconfidence in probability space.

---

## 8. Guardrails on the final submission

Implemented in [`../src/pipeline.py`](../src/pipeline.py):

- **Physical reachability clamp** (`zero_out_impossible_hits`): if a fire can't cover its distance to the
  nearest zone within a horizon — even at 2× its observed speed — that horizon's probability is set to 0. A
  domain prior straight from the "nothing hits from beyond 5 km" observation.
- **Monotonicity** (`enforce_monotonicity`): `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`, because a fire can
  only get closer over time. Enforced with a running max across horizons.

---

## 9. What we'd try next

- Proper survival calibration (e.g. time-dependent / D-calibration) instead of per-horizon binary calibration.
- A Bayesian survival model for principled uncertainty on such a small dataset.
- Spatial features from the evacuation-zone geometry beyond centroid distance (zone shape, road access).
- Leave-one-region-out validation if a region/geography field is available, to test spatial generalization.
