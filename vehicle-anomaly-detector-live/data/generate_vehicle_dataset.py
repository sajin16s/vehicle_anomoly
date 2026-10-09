"""
generate_vehicle_dataset.py
------------------------------
Generates a realistic, CSV-based dataset of vehicle telemetry: time-series
readings for multiple vehicles across several mechanical AND security-relevant
components (e.g. Door Lock Security represents the integrity of lock/command
signals -- relevant for detecting spoofed/injected commands, not just wear).

Each component has:
  - a healthy baseline value
  - slow natural drift (gradual wear over time -- e.g. brakes wearing down)
  - small random sensor noise
  - occasionally injected ANOMALIES: sudden spikes/drops that simulate either
    a sudden component fault OR a spoofed/injected command on the vehicle's
    command bus (the "command-aware" part of this project). This is the
    pattern a real ML model needs to learn to catch -- not just "is the
    value outside a fixed threshold" but "did this change in a way that's
    inconsistent with the vehicle's normal behavior".

Output: data/vehicle_logs.csv with columns:
    vehicle_id, timestamp, component, value, rule_based_status, is_anomaly

`rule_based_status` (HEALTHY/WARNING/CRITICAL) is a simple fixed-threshold
label, kept only as a baseline to compare against the ML model.
`is_anomaly` (0/1) is the ground-truth label used to train/evaluate the ML
model -- it marks injected anomaly EVENTS, which is a harder and more
realistic task than simple thresholding.
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# Component definitions: (baseline value, healthy direction, thresholds)
# direction: "higher_better" or "lower_better" or "band" (must stay within a range)
# ----------------------------------------------------------------------
COMPONENTS = {
    "Fuel Level":          {"baseline": 80, "direction": "higher_better", "warn": 25, "critical": 10, "noise": 1.5},
    "Engine Coolant Temp": {"baseline": 90, "direction": "band", "band": (80, 105), "warn_margin": 8, "noise": 1.0},
    "Battery Charge":      {"baseline": 85, "direction": "higher_better", "warn": 50, "critical": 25, "noise": 1.2},
    "Steering Alignment":  {"baseline": 0.3, "direction": "lower_better", "warn": 1.5, "critical": 3.0, "noise": 0.15, "max_val": 6},
    "Tyre Strength":       {"baseline": 88, "direction": "higher_better", "warn": 50, "critical": 20, "noise": 1.0},
    "Brake Capacity":      {"baseline": 92, "direction": "higher_better", "warn": 70, "critical": 40, "noise": 1.0},
    "Door Lock Security":  {"baseline": 97, "direction": "higher_better", "warn": 80, "critical": 40, "noise": 0.8},
}

WEAR_RATE = {  # how fast each component degrades naturally per timestep (gradual trend)
    "Fuel Level": -0.25,            # consumed over a trip, then "refueled" periodically
    "Engine Coolant Temp": 0.0,     # stays near operating temp, no long-term drift
    "Battery Charge": -0.02,
    "Steering Alignment": 0.002,    # slowly drifts out of alignment
    "Tyre Strength": -0.015,
    "Brake Capacity": -0.01,
    "Door Lock Security": -0.003,
}

N_VEHICLES = 8
N_TIMESTEPS = 220          # readings per component per vehicle
ANOMALY_PROBABILITY = 0.025  # chance of an injected anomaly at any timestep


def rule_based_status(component: str, value: float) -> str:
    cfg = COMPONENTS[component]
    if cfg["direction"] == "higher_better":
        if value < cfg["critical"]:
            return "CRITICAL"
        elif value < cfg["warn"]:
            return "WARNING"
        return "HEALTHY"
    elif cfg["direction"] == "lower_better":
        if value > cfg["critical"]:
            return "CRITICAL"
        elif value > cfg["warn"]:
            return "WARNING"
        return "HEALTHY"
    else:  # band
        low, high = cfg["band"]
        margin = cfg["warn_margin"]
        if value < low - margin or value > high + margin:
            return "CRITICAL"
        elif value < low or value > high:
            return "WARNING"
        return "HEALTHY"


def inject_anomaly(component: str, trend_value: float, kind: str) -> float:
    """Returns a transient anomalous display value for a single reading."""
    cfg = COMPONENTS[component]
    if kind == "spike":
        return trend_value + random.uniform(15, 40)
    elif kind == "drop":
        return max(0, trend_value - random.uniform(15, 40))
    else:  # spoof_extreme -- an obviously out-of-physical-range injected command
        return random.choice([0, cfg.get("max_val", 100) if cfg["direction"] != "higher_better" else 100])


def generate_vehicle_timeseries(vehicle_id: str, component: str) -> list:
    cfg = COMPONENTS[component]
    trend_value = cfg["baseline"] + np.random.normal(0, cfg["noise"])
    rows = []
    fuel_refill_counter = 0
    freeze_remaining = 0
    freeze_value = None

    for t in range(N_TIMESTEPS):
        is_anomaly = 0

        # natural gradual wear / drift -- this is the REAL underlying trend,
        # anomalies below are transient deviations from it, not permanent shifts
        trend_value += WEAR_RATE[component] + np.random.normal(0, cfg["noise"] * 0.3)

        # fuel gets refilled periodically instead of hitting zero
        if component == "Fuel Level":
            fuel_refill_counter += 1
            if trend_value < 15 or fuel_refill_counter > 40:
                trend_value = cfg["baseline"] + np.random.normal(0, cfg["noise"])
                fuel_refill_counter = 0

        trend_value = max(0, trend_value)
        if component == "Steering Alignment":
            trend_value = min(trend_value, 6)
        else:
            trend_value = min(trend_value, 130)

        display_value = trend_value

        if freeze_remaining > 0:
            display_value = freeze_value
            freeze_remaining -= 1
            is_anomaly = 1
        elif random.random() < ANOMALY_PROBABILITY:
            kind = random.choice(["spike", "drop", "freeze", "spoof_extreme"])
            if kind == "freeze":
                freeze_value = cfg["baseline"]
                freeze_remaining = random.randint(2, 4)  # stuck sensor for a short burst
                display_value = freeze_value
            else:
                display_value = inject_anomaly(component, trend_value, kind)
            is_anomaly = 1

        value = max(0, min(display_value, 130 if component != "Steering Alignment" else 10))
        status = rule_based_status(component, value)
        rows.append({
            "vehicle_id": vehicle_id,
            "timestep": t,
            "component": component,
            "value": round(float(value), 2),
            "rule_based_status": status,
            "is_anomaly": is_anomaly,
        })
    return rows


def generate_dataset() -> pd.DataFrame:
    all_rows = []
    for v in range(1, N_VEHICLES + 1):
        vehicle_id = f"VH-{v:03d}"
        for component in COMPONENTS:
            all_rows.extend(generate_vehicle_timeseries(vehicle_id, component))

    df = pd.DataFrame(all_rows)
    # convert timestep -> a real timestamp per vehicle/component series
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df["timestep"] * 5, unit="min")
    df = df[["vehicle_id", "timestamp", "component", "value", "rule_based_status", "is_anomaly"]]
    df = df.sort_values(["vehicle_id", "component", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "vehicle_logs.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} vehicle telemetry rows -> {out_path}")
    print(f"Vehicles: {df['vehicle_id'].nunique()} | Components: {df['component'].nunique()}")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")
    print(df["rule_based_status"].value_counts())
