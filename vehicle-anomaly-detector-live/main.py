"""
main.py
---------
This is the new entry point for the project. Instead of a hardcoded Python
list of readings, the input is now a CSV file (data/vehicle_logs.csv) --
the same approach a real fleet-telemetry pipeline would use (the CSV could
just as easily come from a CAN-bus logger, a telematics API export, etc.).

Pipeline:
    CSV  ->  feature engineering  ->  trained ML model  ->  anomaly graph

Run:
    python main.py                      # uses the first vehicle in the CSV
    python main.py --vehicle_id VH-003  # a specific vehicle
    python main.py --csv path/to/your_own_vehicle_logs.csv
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import engineer_features, get_feature_matrix

HERE = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(HERE, "data", "vehicle_logs.csv")
MODEL_DIR = os.path.join(HERE, "models")
OUTPUT_DIR = os.path.join(HERE, "outputs")

COMPONENT_ORDER = [
    "Fuel Level", "Engine Coolant Temp", "Battery Charge",
    "Steering Alignment", "Tyre Strength", "Brake Capacity", "Door Lock Security",
]


def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    return rf, feature_cols


def run_detection(csv_path: str, vehicle_id: str = None):
    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    if vehicle_id is None:
        vehicle_id = sorted(raw_df["vehicle_id"].unique())[0]
    if vehicle_id not in raw_df["vehicle_id"].unique():
        raise ValueError(f"vehicle_id '{vehicle_id}' not found. "
                          f"Available: {sorted(raw_df['vehicle_id'].unique())}")

    df = engineer_features(raw_df)  # features need full df for correct rolling stats
    rf, feature_cols = load_models()

    X_all = df[feature_cols].fillna(0).values
    df["ml_anomaly_score"] = rf.predict_proba(X_all)[:, 1]
    df["ml_predicted_anomaly"] = rf.predict(X_all)

    vehicle_df = df[df["vehicle_id"] == vehicle_id].copy()
    return vehicle_df, df, vehicle_id


def plot_vehicle_report(vehicle_df: pd.DataFrame, vehicle_id: str, out_path: str):
    components = [c for c in COMPONENT_ORDER if c in vehicle_df["component"].unique()]
    n = len(components)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=False)
    fig.suptitle(f"Vehicle Anomaly Detection Report — {vehicle_id}", fontsize=14, fontweight="bold", y=1.0)

    for ax, comp in zip(axes, components):
        sub = vehicle_df[vehicle_df["component"] == comp].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["value"], color="#3b6fa6", linewidth=1.2, label="value")

        true_anom = sub[sub["is_anomaly"] == 1]
        pred_anom = sub[sub["ml_predicted_anomaly"] == 1]

        ax.scatter(true_anom["timestamp"], true_anom["value"],
                   marker="o", s=55, facecolors="none", edgecolors="orange",
                   linewidths=1.6, label="true injected anomaly", zorder=4)
        ax.scatter(pred_anom["timestamp"], pred_anom["value"],
                   marker="x", s=45, color="red", linewidths=1.6,
                   label="ML-flagged anomaly", zorder=5)

        ax.set_ylabel(comp, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fleet_overview(full_df: pd.DataFrame, out_path: str):
    summary = (
        full_df.groupby("vehicle_id")["ml_predicted_anomaly"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(summary.index, summary.values, color="#c0392b")
    ax.set_title("Fleet Overview — ML-Flagged Anomalies per Vehicle", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of flagged anomaly readings")
    ax.set_xlabel("Vehicle")
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, str(int(val)),
                ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                         help="Path to the input vehicle telemetry CSV")
    parser.add_argument("--vehicle_id", type=str, default=None,
                         help="Vehicle to plot in detail (defaults to the first one in the CSV)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vehicle_df, full_df, vehicle_id = run_detection(args.csv, args.vehicle_id)

    detail_path = os.path.join(OUTPUT_DIR, f"vehicle_report_{vehicle_id}.png")
    fleet_path = os.path.join(OUTPUT_DIR, "fleet_overview.png")

    plot_vehicle_report(vehicle_df, vehicle_id, detail_path)
    plot_fleet_overview(full_df, fleet_path)

    n_flagged = int(vehicle_df["ml_predicted_anomaly"].sum())
    print(f"✅ Success! Processed {len(vehicle_df)} readings for {vehicle_id}.")
    print(f"   ML flagged {n_flagged} anomalous readings.")
    print(f"   Per-component report saved to: {detail_path}")
    print(f"   Fleet overview saved to:       {fleet_path}")


if __name__ == "__main__":
    main()
