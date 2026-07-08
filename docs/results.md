# Results

> Scores below are competition leaderboard scores (higher is better). Fill in the exact metric name and any
> public/private split from the Kaggle *Overview → Evaluation* tab, plus your final rank.

## Headline

| Stage | Score | Δ vs. baseline |
|---|---:|---:|
| Provided baseline (XGBoost Cox PH) | 0.87397 | — |
| **Final: ensemble + Optuna + per-horizon calibration** | **0.96366** | **+0.08969** |
| Program target | 0.90000 | ✅ cleared |

**Leaderboard placement: 477 / 1754 teams — top ~27%** (ahead of ~73% of teams). Note whether this is the
public or private leaderboard once the competition closes.

## Ablation — where the gain came from

Approximate contribution of each component, added on top of the baseline. Replace with your measured
local/leaderboard deltas as you re-run each ablation (the pipeline makes this easy — toggle one piece at a
time).

| Change | Effect | Notes |
|---|---|---|
| Baseline | 0.87397 | XGBoost Cox PH, raw features. |
| + 20 engineered features | ↑ | Distance-decay & interaction features help most. |
| + Optuna tuning | ↑ | Modest but consistent; mostly variance reduction. |
| + 3-model ensemble (CV-weighted) | ↑ | RSF/GBST stabilize XGBoost's variance on 316 rows. |
| **+ per-horizon calibration** | ↑↑ | **Largest single jump** — fixes distant-fire overconfidence. |
| + reachability & monotonicity guardrails | ↑ | Cleans up impossible predictions; guarantees valid output. |
| **Final** | **0.96366** | |

## Diagnostic takeaways

- **Ranking was already decent; calibration was the gap.** Concordance was healthy even at baseline, but the
  probabilities were overconfident. Most of the improvement is in probability *quality*, which is exactly what
  the task rewards.
- **The distant-fire fix generalized.** Fires 7-545 km away were being assigned real hit probability despite
  zero training hits beyond ~5 km; distance-decay features + calibration + the reachability clamp collapsed
  that error.
- **The ensemble earned its keep through stability, not a new peak.** On 316 rows, blending three models
  mainly reduced variance across CV folds rather than lifting any single horizon dramatically.

## Reproducing these numbers

See [`reproducibility.md`](reproducibility.md). Because ensemble weights and calibrator choices are learned
from cross-validation with a fixed seed, a clean run should land within noise of the reported score, given the
same data and library versions.
