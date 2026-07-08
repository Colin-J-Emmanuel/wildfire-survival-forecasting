"""
Central configuration for the WiDS 2026 wildfire survival pipeline.

Keeping paths, horizons, and column names in one place means the rest of the
codebase never hard-codes a magic number, and adapting to the *actual* Kaggle
column names is a one-file change.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
FIGURES_DIR = ROOT / "figures"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
# The four forecast horizons, in hours. Order matters: predictions must be
# monotonically non-decreasing across these (a fire can only get closer).
HORIZONS = (12, 24, 48, 72)

# A fire "hits" a zone when it comes within this distance of a zone centroid.
HIT_RADIUS_KM = 5.0

# The observation window that generated all input features.
FEATURE_WINDOW_HOURS = 5

# --------------------------------------------------------------------------- #
# Column names — ALIGN THESE WITH THE ACTUAL KAGGLE DATASET
# --------------------------------------------------------------------------- #
# These names reflect the documented data schema (features from the first five
# hours: early spread dynamics + spatial relationship to evacuation zones).
# Rename them to match the real column headers when you plug in the data.
ID_COL = "event_id"

# Survival labels (present in train only)
EVENT_COL = "event"                     # 1 = fire hit a zone within 72h, 0 = censored
TIME_COL = "time_to_hit_hours"          # observed hit time (or censoring time = 72)

# Example raw feature columns the engineered features expect. Adjust to reality.
RAW_FEATURES = [
    "distance_to_zone_km",              # proximity to nearest evacuation-zone centroid
    "spread_rate_kmh",                  # early spread speed
    "perimeter_growth_km2",             # area growth over the 5h window
    "bearing_to_zone_deg",              # direction of the nearest zone
    "wind_speed_kmh",
    "wind_alignment",                   # cos(angle between wind and bearing-to-zone), in [-1, 1]
    "fuel_dryness_index",
    "terrain_slope",
]

RANDOM_STATE = 42
N_SPLITS = 5                            # CV folds (small dataset → modest fold count)
