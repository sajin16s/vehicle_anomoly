"""
feature_extraction.py
------------------------
Converts raw (vehicle_id, timestamp, component, value) readings into ML-ready
features. This is the core of what makes this an ML project rather than a
simple threshold check: instead of just asking "is this value out of range?"
we ask "is this reading inconsistent with this component's own recent
history?" -- which is what lets the model catch sudden spoofed/injected
values even when they happen to land inside a normally-healthy-looking range.

Per-reading engineered features (computed within each vehicle+component
time series, ordered by timestamp):
    value                 - the raw reading
    rolling_mean_5        - mean of the last 5 readings (recent baseline)
    rolling_std_5         - std of the last 5 readings (recent volatility)
    z_score               - (value - rolling_mean_5) / rolling_std_5
    delta                 - change from the previous reading
    pct_change            - percent change from the previous reading
    rolling_mean_20       - longer-window baseline (captures gradual wear)
    deviation_from_long_baseline - value vs the long-window mean
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain: vehicle_id, timestamp, component, value
    (is_anomaly / rule_based_status are optional, kept if present).
    Returns a new dataframe with added feature columns, NaNs from the
    rolling windows' warm-up period dropped.
    """
    df = df.sort_values(["vehicle_id", "component", "timestamp"]).copy()
    group = df.groupby(["vehicle_id", "component"])["value"]

    df["rolling_mean_5"] = group.transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["rolling_std_5"] = group.transform(lambda s: s.rolling(5, min_periods=3).std())
    df["rolling_mean_20"] = group.transform(lambda s: s.rolling(20, min_periods=5).mean())

    df["delta"] = group.transform(lambda s: s.diff())
    df["pct_change"] = group.transform(lambda s: s.pct_change().replace([np.inf, -np.inf], 0))

    # avoid divide-by-zero on z-score
    safe_std = df["rolling_std_5"].replace(0, np.nan)
    df["z_score"] = (df["value"] - df["rolling_mean_5"]) / safe_std
    df["z_score"] = df["z_score"].fillna(0)

    df["deviation_from_long_baseline"] = df["value"] - df["rolling_mean_20"]

    # one-hot encode component (small cardinality, fine for tree models too,
    # but explicit one-hot keeps the feature matrix model-agnostic)
    component_dummies = pd.get_dummies(df["component"], prefix="comp")
    df = pd.concat([df, component_dummies], axis=1)

    df = df.dropna(subset=["rolling_mean_5", "rolling_std_5", "rolling_mean_20"]).reset_index(drop=True)
    return df


FEATURE_COLUMNS_BASE = [
    "value", "rolling_mean_5", "rolling_std_5", "rolling_mean_20",
    "delta", "pct_change", "z_score", "deviation_from_long_baseline",
]


def get_feature_matrix(df: pd.DataFrame):
    """Returns (X, feature_column_names) ready for sklearn, including component one-hots."""
    component_cols = [c for c in df.columns if c.startswith("comp_")]
    feature_cols = FEATURE_COLUMNS_BASE + component_cols
    X = df[feature_cols].fillna(0).values
    return X, feature_cols
