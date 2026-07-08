# Reproducibility

The goal of this file: someone who has never seen the project can clone it, add the data, run one command,
and get our submission. Reproducibility was an explicit responsibility on the team, so the pipeline is
deterministic wherever we can make it (fixed `RANDOM_STATE = 42` in `src/config.py`).

## 1. Environment

```bash
python --version            # 3.10 or newer
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **`scikit-survival` note:** it needs a C/C++ compiler and a matching NumPy. If `pip install scikit-survival`
> fails, install via conda instead (`conda install -c sepandhaghighi scikit-survival` or
> `conda install -c conda-forge scikit-survival`), or see the sksurv install docs. Everything else installs
> cleanly with pip.

## 2. Data

Download `train.csv` and `test.csv` from Kaggle into `data/` — see [`../data/README.md`](../data/README.md).
Then confirm the column names in [`../src/config.py`](../src/config.py) match the real headers (`ID_COL`,
`EVENT_COL`, `TIME_COL`, `RAW_FEATURES`). This is the one manual alignment step.

## 3. Run

```bash
# Reproduce the submission end-to-end (ensemble + calibration + guardrails)
python -m src.pipeline --data-dir data --out submissions/submission.csv

# Same, but re-run Optuna tuning first (slower)
python -m src.pipeline --data-dir data --out submissions/submission.csv --tune
```

The script prints, in order: data shapes, per-horizon ensemble weights, the calibration method chosen at each
horizon, and local pre/post-calibration metrics — so you can watch the improvement happen before you ever
submit.

## 4. Interactive walkthrough

```bash
jupyter lab notebooks/wildfire_survival_walkthrough.ipynb
```

The notebook reproduces the same steps with commentary and plots (EDA, the distant-fire overconfidence chart,
reliability diagrams before/after calibration).

## 5. Determinism & expected variance

- All learners and splits use `RANDOM_STATE = 42`.
- Ensemble weights and calibrator choices are learned from fixed-seed CV, so they're stable run to run.
- Small numeric differences can still arise from **library/BLAS versions** (especially XGBoost and
  scikit-survival). Pin the versions in `requirements.txt` for an exact match; a clean run should otherwise
  land within noise of **0.96366**.

## 6. Our scoring submission

The exact file that produced our score is checked in at
[`../submissions/submission_colin.csv`](../submissions/submission_colin.csv) (95 fires, columns
`event_id, prob_12h, prob_24h, prob_48h, prob_72h`), so results are verifiable even without re-running.
