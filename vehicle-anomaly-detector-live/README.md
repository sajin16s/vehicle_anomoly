# Vehicle Command-Aware Anomaly Detection System

An ML-based system that monitors vehicle telemetry — mechanical sensors
(fuel, engine temperature, battery, steering, tyres, brakes) **and**
security-relevant command/lock-state signals — and flags anomalous
readings that could indicate a developing fault or a spoofed/injected
command, rather than just checking values against fixed thresholds.

**Tech stack:** Python, scikit-learn, pandas, matplotlib

## What's new vs. a basic threshold checker

The original version of this script hardcoded a single snapshot of
readings as a Python list and labeled each with a fixed-threshold
`HEALTHY/WARNING/CRITICAL` status. This version:

1. **Reads from a CSV** (`data/vehicle_logs.csv`) instead of a hardcoded
   list — a real fleet pipeline would feed this from a telematics API,
   CAN-bus logger, or similar.
2. **Is time-series, multi-vehicle**: each component has a history of
   readings per vehicle, not a single point-in-time value.
3. **Uses real ML, not just thresholds**: features are engineered from
   each value's *recent history* (rolling mean/std, rate of change,
   z-score, deviation from long-term baseline) so the model can catch
   sudden, suspicious changes — even ones that land inside a normally
   "healthy-looking" range — which a fixed threshold check would miss
   entirely. This is what makes it relevant to detecting spoofed/injected
   command values, not just gradual mechanical wear.
4. **Outputs a graph**, not just console text: a per-vehicle, per-component
   time-series report with anomalies highlighted, plus a fleet-wide
   summary chart.

## Project structure

```
vehicle-anomaly-detector/
├── data/
│   ├── generate_vehicle_dataset.py  # builds the CSV (synthetic but realistic time series)
│   └── vehicle_logs.csv             # input data: vehicle_id, timestamp, component, value, ...
├── src/
│   ├── feature_extraction.py        # rolling-window feature engineering
│   └── train_model.py               # trains IsolationForest + RandomForest, evaluates
├── models/                          # saved .joblib model artifacts (generated)
├── outputs/                         # generated graphs (PNG)
├── main.py                          # CSV in -> ML detection -> graph out
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the input CSV (or point at your own real telemetry CSV later)
python data/generate_vehicle_dataset.py

# 2. Train the models
python src/train_model.py

# 3. Run detection + produce the graphs
python main.py --vehicle_id VH-003
```

Outputs land in `outputs/`:
- `vehicle_report_<vehicle_id>.png` — one subplot per component, with
  ML-flagged anomalies (red ✕) and the true injected anomalies (orange ○,
  for validation) marked on the time series.
- `fleet_overview.png` — bar chart of flagged anomaly counts per vehicle.

## Results on this dataset

| Model                  | Accuracy | Precision | Recall | F1   |
|-------------------------|---------|-----------|--------|------|
| RandomForestClassifier  | ~99%    | ~92%      | ~89%   | ~90% |
| IsolationForest         | ~95%    | ~40%      | ~39%   | ~39% |

The supervised RandomForest is the production model. Its top features
by importance are `rolling_std_5`, `delta`, `pct_change`, and `z_score`
— i.e. it's learning to recognize *abnormal change patterns*, not just
memorizing raw sensor values. That's the key evidence this is a genuine
ML anomaly detector rather than a relabeled threshold check.

## Using your own CSV

`main.py --csv path/to/your_file.csv` works with any CSV that has these
columns: `vehicle_id, timestamp, component, value` (an `is_anomaly` column
is only needed for training/evaluation, not for running detection).

## Notes on the dataset

`data/generate_vehicle_dataset.py` simulates realistic telemetry: a slow
natural wear trend per component (e.g. brakes gradually wearing down,
battery slowly depleting) plus occasional **transient injected anomalies**
(sudden spikes, drops, or stuck/frozen-sensor readings) representing either
a sudden fault or a spoofed/injected command value. This is clearly
documented as synthetic in this script — for a portfolio project this is
standard practice; swap it for real telematics/OBD-II/CAN-bus logs to make
it production-real.
