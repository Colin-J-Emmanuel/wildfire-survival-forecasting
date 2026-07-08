"""
Build a single, self-contained Kaggle notebook from the canonical `src/` modules.

Why this exists: Kaggle reviewers expect one notebook that runs top-to-bottom in
the kernel with nothing to install locally. But we don't want two copies of the
logic. So `src/` stays the single source of truth, and this script *generates* the
standalone notebook by inlining the module bodies (imports consolidated, relative
imports removed, config swapped for Kaggle paths).

Edit the modules in `src/`, re-run this script, and the Kaggle notebook updates.

Usage:  python scripts/build_kaggle_notebook.py
Output: notebooks/wildfire_survival_kaggle.ipynb
"""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "wildfire_survival_kaggle.ipynb"

# Modules to inline, in dependency order. config is handled separately (Kaggle paths).
MODULES = ["features", "models", "ensemble", "calibration", "evaluate", "pipeline"]
# Top-level defs we don't want in a notebook (CLI plumbing from pipeline.py).
DROP_DEFS = {"main", "run"}


def strip_module(path: Path):
    """Return (import_lines, body_source) with docstring, relative imports,
    __future__, argparse, and CLI defs removed."""
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()          # 0-based line indices to blank out
    imports = []

    for node in tree.body:
        # module docstring
        if (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)):
            for ln in range(node.lineno - 1, node.end_lineno):
                drop.add(ln)
            continue
        # imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            is_relative = isinstance(node, ast.ImportFrom) and (node.level or 0) > 0
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            is_argparse = isinstance(node, ast.Import) and any(a.name == "argparse" for a in node.names)
            stmt = "\n".join(lines[node.lineno - 1:node.end_lineno])
            for ln in range(node.lineno - 1, node.end_lineno):
                drop.add(ln)
            if not (is_relative or is_future or is_argparse):
                imports.append(stmt)
            continue
        # CLI defs + __main__ guard
        if isinstance(node, ast.FunctionDef) and node.name in DROP_DEFS:
            for ln in range(node.lineno - 1, node.end_lineno):
                drop.add(ln)
            continue
        if isinstance(node, ast.If):  # `if __name__ == "__main__":`
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"):
                for ln in range(node.lineno - 1, node.end_lineno):
                    drop.add(ln)
                continue

    body = "\n".join(l for i, l in enumerate(lines) if i not in drop).strip("\n")
    # collapse >2 blank lines
    while "\n\n\n\n" in body:
        body = body.replace("\n\n\n\n", "\n\n\n")
    return imports, body


def build():
    all_imports, bodies = [], {}
    for name in MODULES:
        imps, body = strip_module(SRC / f"{name}.py")
        all_imports += imps
        bodies[name] = body

    # de-dupe imports, stable order
    seen, deduped = set(), []
    for imp in all_imports:
        if imp not in seen:
            seen.add(imp); deduped.append(imp)

    cells = []
    md = lambda t: cells.append(new_markdown_cell(t))
    code = lambda t: cells.append(new_code_cell(t))

    md("""# 🔥 Predicting Time-to-Threat for Wildfire Evacuation Zones
**WiDS Global Datathon 2026 · Survival Analysis**

Predicts, from only the first **5 hours** of a wildfire, the probability it threatens an evacuation zone
within **12 / 24 / 48 / 72 hours**. Pipeline: engineered features → survival ensemble
(XGBoost Cox + Random Survival Forest + Gradient Boosted Survival Trees) → **per-horizon calibration** →
physical guardrails.

> **Self-contained Kaggle version.** This notebook is auto-generated from the modular, unit-tested source
> in our GitHub repo (`src/`) by `scripts/build_kaggle_notebook.py` — that repo is the canonical source.
> Baseline **0.87397 → 0.96366** after these enhancements.""")

    md("### Install dependencies (Kaggle kernels don't ship scikit-survival)")
    code("!pip install -q scikit-survival optuna xgboost")

    md("### Imports")
    code("\n".join(deduped))

    md("""### Configuration
Set `DATA_DIR` to the competition's input folder, and confirm the column names match the real dataset
headers (`ID_COL`, `EVENT_COL`, `TIME_COL`, `RAW_FEATURES`).""")
    code('''from types import SimpleNamespace
from pathlib import Path

config = SimpleNamespace(
    # ---- paths (Kaggle) ----
    DATA_DIR=Path("/kaggle/input/WiDSWorldWide_GlobalDathon26"),   # <-- confirm slug
    SUBMISSIONS_DIR=Path("/kaggle/working"),
    TRAIN_FILE="train.csv",
    TEST_FILE="test.csv",
    # ---- problem definition ----
    HORIZONS=(12, 24, 48, 72),
    HIT_RADIUS_KM=5.0,
    FEATURE_WINDOW_HOURS=5,
    # ---- columns (ALIGN WITH REAL DATA) ----
    ID_COL="event_id",
    EVENT_COL="event",
    TIME_COL="time_to_hit_hours",
    RAW_FEATURES=[
        "distance_to_zone_km", "spread_rate_kmh", "perimeter_growth_km2",
        "bearing_to_zone_deg", "wind_speed_kmh", "wind_alignment",
        "fuel_dryness_index", "terrain_slope",
    ],
    RANDOM_STATE=42,
    N_SPLITS=5,
)''')

    section_titles = {
        "features": "### Feature engineering (+20 features)\nDistance-decay, distance×speed interactions, directional, growth-dynamics, composite risk scores.",
        "models": "### Survival models\nXGBoost Cox (with a Breslow baseline), Random Survival Forest, Gradient Boosted Survival Trees — behind one `predict_hit_prob` interface. Includes optional Optuna tuning.",
        "ensemble": "### Cross-validated ensemble\nPer-horizon blend weights learned on out-of-fold predictions.",
        "calibration": "### Per-horizon calibration\nThe biggest single win: isotonic vs. Platt per horizon, chosen by Brier score.",
        "evaluate": "### Evaluation utilities\nConcordance + Brier + reliability curves for local validation before spending a submission.",
        "pipeline": "### Pipeline helpers\nFeature prep, physical reachability guardrail, and monotonicity enforcement.",
    }
    for name in MODULES:
        md(section_titles[name])
        code(bodies[name])

    md("""## Run it: data → submission
Loads the data, fits the ensemble, calibrates on out-of-fold predictions, applies guardrails, and writes a
valid `submission.csv` to `/kaggle/working`.""")
    code('''import numpy as np, pandas as pd

train = pd.read_csv(config.DATA_DIR / config.TRAIN_FILE)
test  = pd.read_csv(config.DATA_DIR / config.TEST_FILE)
print("train:", train.shape, "| test:", test.shape)

X_train, X_test, event, time, feature_cols = prepare_features(train, test)
print(f"{len(feature_cols)} model-input features")

ensemble = SurvivalEnsemble().fit(X_train, event, time)
print("per-horizon weights:")
print(ensemble.weight_table().round(2).to_string())''')

    code('''# Fit calibration on out-of-fold blended predictions (honest, no leakage).
oof = ensemble._out_of_fold_predictions(X_train, event, time, config.HORIZONS)
oof_blend = np.zeros((len(X_train), len(config.HORIZONS)))
for j in range(len(config.HORIZONS)):
    stack = np.stack([oof[n][:, j] for n in ensemble.model_names], axis=1)
    oof_blend[:, j] = stack @ ensemble.weights_[:, j]

calibrator = PerHorizonCalibrator().fit(oof_blend, event, time)
print("calibration per horizon:", calibrator.summary())
print("pre :", evaluate_all(oof_blend, event, time))
print("post:", evaluate_all(calibrator.transform(oof_blend), event, time))''')

    code('''# Predict on test, calibrate, apply guardrails, enforce monotonicity.
test_probs = ensemble.predict_hit_prob(X_test, config.HORIZONS)
test_probs = calibrator.transform(test_probs)
test_probs = zero_out_impossible_hits(test, test_probs)
test_probs = enforce_monotonicity(test_probs)

submission = pd.DataFrame({config.ID_COL: test[config.ID_COL].to_numpy()})
for j, h in enumerate(config.HORIZONS):
    submission[f"prob_{h}h"] = test_probs[:, j]

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("wrote", len(submission), "rows")
submission.head()''')

    md("""## Recap
Survival framing (uses the censored labels), a CV-weighted 3-model ensemble, **per-horizon calibration**
(the decisive fix for distant-fire overconfidence), and physical guardrails took the baseline from
**0.87397 → 0.96366**, clearing the 0.90 target.""")

    nb = new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    OUT.write_text("")  # ensure exists
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    return nb, deduped, bodies


if __name__ == "__main__":
    nb, deduped, bodies = build()
    nbf.validate(nb)
    print(f"Wrote {OUT.relative_to(ROOT)} — {len(nb['cells'])} cells, {len(deduped)} consolidated imports")
