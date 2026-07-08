"""
Feature engineering for the wildfire survival model.

The design goal of every feature here is to make the *distance / speed /
direction* relationship explicit, so the model doesn't have to discover it from
316 examples. The distance-decay features in particular exist to counter the
baseline's failure mode: assigning real hit probability to fires 7-545 km away,
despite zero training hits beyond ~5 km.

All functions are pure (DataFrame in, DataFrame out) so they compose cleanly and
are trivial to unit-test. Column names come from ``src.config`` — align those
with the real dataset and this module follows automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EPS = 1e-6


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the full engineered feature set appended.

    Grouped by intent so the reasoning is legible. Guards with ``.get`` keep the
    function robust if a raw column is renamed or missing in your copy of the
    data — engineer only what the inputs support.
    """
    df = df.copy()
    df = _add_distance_decay_features(df)
    df = _add_distance_speed_interactions(df)
    df = _add_directional_features(df)
    df = _add_growth_dynamics_features(df)
    df = _add_composite_risk_scores(df)
    return df


# --------------------------------------------------------------------------- #
# 1. Distance-decay — the direct fix for distant-fire overconfidence
# --------------------------------------------------------------------------- #
def _add_distance_decay_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.get("distance_to_zone_km")
    if d is None:
        return df
    # Sharp, monotone decay so the model can express "risk ~ 0 beyond a few km".
    df["dist_inverse"] = 1.0 / (d + 1.0)
    df["dist_exp_decay_2km"] = np.exp(-d / 2.0)
    df["dist_exp_decay_5km"] = np.exp(-d / config.HIT_RADIUS_KM)
    df["dist_log1p"] = np.log1p(d)
    df["dist_sq"] = d ** 2
    # Hard indicators around the 5 km hit radius — a strong, interpretable prior.
    df["within_hit_radius"] = (d <= config.HIT_RADIUS_KM).astype(int)
    df["within_2x_hit_radius"] = (d <= 2 * config.HIT_RADIUS_KM).astype(int)
    df["is_distant_fire"] = (d > 4 * config.HIT_RADIUS_KM).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 2. Distance x speed interactions — "how fast is it closing the gap?"
# --------------------------------------------------------------------------- #
def _add_distance_speed_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.get("distance_to_zone_km")
    v = df.get("spread_rate_kmh")
    if d is None or v is None:
        return df
    # Naive time-to-reach if the fire drove straight at the zone at current speed.
    df["hours_to_zone_naive"] = d / (v + EPS)
    df["speed_per_distance"] = v / (d + 1.0)          # closing "pressure"
    df["speed_x_proximity"] = v * np.exp(-d / config.HIT_RADIUS_KM)
    # Can the fire physically cover the distance within each horizon?
    for h in config.HORIZONS:
        df[f"reachable_by_{h}h"] = (v * h >= d).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 3. Directional features — a fast fire pointed *away* is not a threat
# --------------------------------------------------------------------------- #
def _add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    align = df.get("wind_alignment")           # cos(angle) toward the zone, [-1, 1]
    v = df.get("spread_rate_kmh")
    if align is not None and v is not None:
        # Effective speed *component* aimed at the zone.
        df["directed_speed"] = v * align.clip(lower=0)
        df["heading_toward_zone"] = (align > 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 4. Growth dynamics — acceleration/energy signals from the 5h window
# --------------------------------------------------------------------------- #
def _add_growth_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    area = df.get("perimeter_growth_km2")
    if area is not None:
        df["growth_rate_per_hour"] = area / config.FEATURE_WINDOW_HOURS
        df["log_growth"] = np.log1p(area)
    dry = df.get("fuel_dryness_index")
    wind = df.get("wind_speed_kmh")
    if dry is not None and wind is not None:
        # A simple fire-weather energy proxy: dry fuel + wind => faster spread.
        df["fire_weather_energy"] = dry * wind
    return df


# --------------------------------------------------------------------------- #
# 5. Composite risk scores — bundle the signals into interpretable indices
# --------------------------------------------------------------------------- #
def _add_composite_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.get("distance_to_zone_km")
    v = df.get("spread_rate_kmh")
    if d is None or v is None:
        return df
    proximity = np.exp(-d / config.HIT_RADIUS_KM)                 # 1 when on top of zone -> 0 far away
    speed_norm = (v - v.min()) / (v.max() - v.min() + EPS)        # 0..1
    directed = df.get("directed_speed", v)
    directed_norm = (directed - directed.min()) / (directed.max() - directed.min() + EPS)

    # Overall "threat" index and a distance-gated version (0 for distant fires).
    df["risk_score"] = proximity * (0.5 + 0.5 * speed_norm)
    df["risk_score_directed"] = proximity * (0.5 + 0.5 * directed_norm)
    df["gated_risk"] = df["risk_score"] * df.get("within_2x_hit_radius", 1)
    return df


def engineered_feature_columns(df: pd.DataFrame) -> list[str]:
    """All model-input columns: engineered features plus any raw features present.

    Excludes the id and survival-label columns.
    """
    exclude = {config.ID_COL, config.EVENT_COL, config.TIME_COL}
    return [c for c in df.columns if c not in exclude]
