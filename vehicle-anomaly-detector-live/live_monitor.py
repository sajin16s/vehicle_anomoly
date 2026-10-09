"""
live_monitor.py
---------------
Replaces the static PNG output of main.py with a live, animated terminal
dashboard.  Every tick the vehicle "moves":

  • Fuel Level drains slowly (–0.10 to –0.20 % per tick)
  • Engine Coolant Temp drifts upward until it stabilises around 90 °C
  • All other sensors fluctuate realistically with small Gaussian noise
  • The trained Random Forest scores every new reading in real time
  • Anomaly spikes are injected randomly (~2.5 % of ticks per sensor)
  • Each subplot scrolls a rolling 60-tick window so the graph is always live

Run:
    python live_monitor.py                      # first vehicle in CSV
    python live_monitor.py --vehicle_id VH-003
    python live_monitor.py --interval 800       # ms between ticks (default 600)
    python live_monitor.py --window 80          # how many ticks to show (default 60)
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # interactive backend – change to "Qt5Agg" if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import FEATURE_COLUMNS_BASE

HERE        = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(HERE, "models")
DEFAULT_CSV = os.path.join(HERE, "data", "vehicle_logs.csv")

# ── sensor catalogue ──────────────────────────────────────────────────────────
SENSORS = [
    {
        "name":       "Fuel Level",
        "unit":       "%",
        "color":      "#378add",
        "start":      (72, 85),   # (min_start, max_start)
        "clip":       (0, 100),
        "drift":      -0.15,      # per tick mean drift  (negative = draining)
        "noise":      0.25,
        "drain":      True,       # shown as "↓ draining" label
        "warn_lo":    20,
        "danger_lo":  10,
    },
    {
        "name":       "Engine Coolant Temp",
        "unit":       "°C",
        "color":      "#ba7517",
        "start":      (60, 70),
        "clip":       (55, 130),
        "drift":      0.10,
        "noise":      0.60,
        "target":     90,         # asymptote – drift slows as value approaches target
        "warn_hi":    105,
        "danger_hi":  115,
    },
    {
        "name":       "Battery Charge",
        "unit":       "%",
        "color":      "#3b6d11",
        "start":      (75, 90),
        "clip":       (0, 100),
        "drift":      -0.03,
        "noise":      0.35,
        "warn_lo":    30,
        "danger_lo":  15,
    },
    {
        "name":       "Steering Alignment",
        "unit":       "°",
        "color":      "#993c1d",
        "start":      (0.1, 0.5),
        "clip":       (0, 10),
        "drift":      0.008,
        "noise":      0.10,
        "warn_hi":    3,
        "danger_hi":  6,
    },
    {
        "name":       "Tyre Strength",
        "unit":       "%",
        "color":      "#0f6e56",
        "start":      (75, 92),
        "clip":       (0, 100),
        "drift":      -0.025,
        "noise":      0.40,
        "warn_lo":    30,
        "danger_lo":  15,
    },
    {
        "name":       "Brake Capacity",
        "unit":       "%",
        "color":      "#534ab7",
        "start":      (82, 96),
        "clip":       (0, 100),
        "drift":      -0.018,
        "noise":      0.30,
        "warn_lo":    40,
        "danger_lo":  20,
    },
    {
        "name":       "Door Lock Security",
        "unit":       "%",
        "color":      "#3c3489",
        "start":      (95, 100),
        "clip":       (0, 100),
        "drift":      0.0,
        "noise":      0.08,
        "warn_lo":    50,
        "danger_lo":  20,
    },
]

ANOMALY_PROB    = 0.025   # probability of injecting a spike per sensor per tick
ANOMALY_SPIKE   = 0.35    # fraction of (clip_max – clip_min) for spike magnitude

HEALTHY_COLOR   = "#2e7d32"
WARN_COLOR      = "#e65100"
DANGER_COLOR    = "#c62828"
ANOMALY_MARKER  = "#e24b4a"


# ── sensor state ─────────────────────────────────────────────────────────────
class SensorState:
    def __init__(self, cfg, window: int):
        lo, hi    = cfg["start"]
        self.cfg  = cfg
        self.val  = lo + np.random.random() * (hi - lo)
        self.hist = deque([self.val] * window, maxlen=window)
        self.anom = deque([False] * window, maxlen=window)
        # rolling buffers for on-the-fly feature engineering
        self._buf = deque([self.val] * 25, maxlen=25)

    def tick(self) -> tuple[float, bool]:
        cfg  = self.cfg
        cmin, cmax = cfg["clip"]
        rng  = cmax - cmin

        drift = cfg["drift"]
        # asymptote logic for Engine Coolant Temp
        if "target" in cfg:
            drift = drift * max(0, 1 - abs(self.val - cfg["target"]) / 30)

        v = self.val + drift + np.random.normal(0, cfg["noise"])

        is_anomaly = False
        if np.random.random() < ANOMALY_PROB:
            spike = (1 if np.random.random() > 0.5 else -1) * (
                ANOMALY_SPIKE * rng * (0.7 + np.random.random() * 0.6)
            )
            v += spike
            is_anomaly = True

        v = float(np.clip(v, cmin, cmax))
        self.val = v
        self.hist.append(v)
        self.anom.append(is_anomaly)
        self._buf.append(v)
        return v, is_anomaly

    def feature_row(self) -> dict:
        arr  = np.array(self._buf)
        w5   = arr[-5:]
        w20  = arr[-20:]
        rm5  = w5.mean()
        rs5  = w5.std()
        rm20 = w20.mean()
        prev = arr[-2] if len(arr) >= 2 else arr[-1]
        delta = self.val - prev
        pct   = delta / (abs(prev) + 1e-9)
        z     = (self.val - rm5) / (rs5 + 1e-9)
        dev   = self.val - rm20
        return {
            "value":                      self.val,
            "rolling_mean_5":             rm5,
            "rolling_std_5":              rs5,
            "rolling_mean_20":            rm20,
            "delta":                      delta,
            "pct_change":                 pct,
            "z_score":                    z,
            "deviation_from_long_baseline": dev,
        }

    def status(self) -> str:
        v, cfg = self.val, self.cfg
        if cfg.get("danger_lo") and v < cfg["danger_lo"]:  return "danger"
        if cfg.get("danger_hi") and v > cfg["danger_hi"]:  return "danger"
        if cfg.get("warn_lo")   and v < cfg["warn_lo"]:    return "warn"
        if cfg.get("warn_hi")   and v > cfg["warn_hi"]:    return "warn"
        return "ok"


# ── ML scoring ────────────────────────────────────────────────────────────────
def load_model():
    try:
        rf           = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
        return rf, feature_cols
    except FileNotFoundError:
        print("⚠  No trained model found.  Run `python src/train_model.py` first.")
        return None, None


def ml_score(rf, feature_cols: list, sensor: SensorState, sensor_name: str) -> float:
    """Return anomaly probability from the Random Forest for this reading."""
    if rf is None:
        return 0.0
    row = sensor.feature_row()
    # add one-hot columns for components; zero-fill unknowns
    for col in feature_cols:
        if col not in row:
            row[col] = 1.0 if col == f"comp_{sensor_name}" else 0.0
    X = pd.DataFrame([row])[feature_cols].fillna(0).values
    return float(rf.predict_proba(X)[0, 1])


# ── dashboard ─────────────────────────────────────────────────────────────────
def build_dashboard(vehicle_id: str, window: int, interval_ms: int):
    rf, feature_cols = load_model()
    n = len(SENSORS)
    states = [SensorState(s, window) for s in SENSORS]
    xs = list(range(window))
    tick_counter = [0]
    anomaly_log  = []          # list of (tick, sensor_name, value)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    fig.canvas.manager.set_window_title(f"Vehicle Live Monitor — {vehicle_id}")

    # title
    fig.text(0.015, 0.975, f"🚗  Vehicle Live Monitor", fontsize=14,
             fontweight="bold", color="white", va="top")
    fig.text(0.015, 0.955, f"Vehicle: {vehicle_id}   |   ML model: Random Forest",
             fontsize=9, color="#aaaaaa", va="top")
    tick_text  = fig.text(0.98, 0.975, "Tick: 0", fontsize=9,
                          color="#aaaaaa", va="top", ha="right")
    clock_text = fig.text(0.98, 0.958, "", fontsize=9,
                          color="#aaaaaa", va="top", ha="right")

    # grid of subplots  (4 cols: 3 sensor charts + 1 sidebar)
    # left 3 cols → sensor plots,  right col → anomaly log + ML score bar
    gs = fig.add_gridspec(n, 4, left=0.06, right=0.98, top=0.93, bottom=0.04,
                          hspace=0.55, wspace=0.35,
                          width_ratios=[1, 1, 1, 0.85])

    axes   = []
    lines  = []
    scatters = []
    ml_bars  = []
    val_texts = []
    status_texts = []

    for i, (sensor, state) in enumerate(zip(SENSORS, states)):
        col = i % 3
        ax  = fig.add_subplot(gs[i, col])
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.tick_params(colors="#888888", labelsize=7)
        ax.set_xlim(0, window - 1)
        cmin, cmax = sensor["clip"]
        ax.set_ylim(cmin - (cmax - cmin) * 0.05,
                    cmax + (cmax - cmin) * 0.05)
        ax.set_title(sensor["name"], fontsize=8, color="#cccccc",
                     fontweight="bold", pad=3)
        ax.set_ylabel(sensor["unit"], fontsize=7, color="#888888")
        ax.grid(color="#2a2a2a", linewidth=0.5)

        line, = ax.plot(xs, list(state.hist), color=sensor["color"],
                        linewidth=1.4, zorder=2)
        scat  = ax.scatter([], [], color=ANOMALY_MARKER, s=40, zorder=5,
                           marker="x", linewidths=1.5)

        # current value text in corner
        vt = ax.text(0.98, 0.94, "", transform=ax.transAxes,
                     fontsize=8, color="white", ha="right", va="top",
                     fontweight="bold")
        st = ax.text(0.98, 0.75, "", transform=ax.transAxes,
                     fontsize=7, ha="right", va="top")

        axes.append(ax); lines.append(line); scatters.append(scat)
        val_texts.append(vt); status_texts.append(st)

    # ── sidebar: anomaly log + ML score bars ─────────────────────────────────
    ax_log = fig.add_subplot(gs[0:4, 3])
    ax_log.set_facecolor("#111111")
    for sp in ax_log.spines.values():
        sp.set_color("#333333")
    ax_log.set_title("ML Anomaly Scores", fontsize=8, color="#cccccc",
                      fontweight="bold", pad=4)
    ax_log.set_xlim(0, 1)
    ax_log.set_ylim(-0.5, n - 0.5)
    ax_log.set_yticks(range(n))
    ax_log.set_yticklabels([s["name"] for s in SENSORS], fontsize=7,
                            color="#aaaaaa")
    ax_log.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_log.set_xticklabels(["0", ".25", ".50", ".75", "1"], fontsize=7,
                             color="#888888")
    ax_log.xaxis.set_label_position("top")
    ax_log.xaxis.tick_top()
    ax_log.set_xlabel("Anomaly probability", fontsize=7, color="#888888",
                       labelpad=4)
    ax_log.grid(axis="x", color="#222222", linewidth=0.5)
    ax_log.axvline(0.5, color="#444444", linewidth=0.8, linestyle="--")

    for i, sensor in enumerate(SENSORS):
        bar = ax_log.barh(i, 0, color=sensor["color"], height=0.55,
                          alpha=0.85)
        ml_bars.append(bar)

    score_texts = [
        ax_log.text(0.02, i, "0.00", va="center", fontsize=7, color="white")
        for i in range(n)
    ]

    # ── anomaly log panel (bottom of sidebar) ────────────────────────────────
    ax_alog = fig.add_subplot(gs[4:, 3])
    ax_alog.set_facecolor("#111111")
    for sp in ax_alog.spines.values():
        sp.set_color("#333333")
    ax_alog.set_title("Anomaly log (last 5)", fontsize=8, color="#cccccc",
                       fontweight="bold", pad=4)
    ax_alog.axis("off")
    log_texts = [
        ax_alog.text(0.04, 1 - (j + 1) * 0.17, "", transform=ax_alog.transAxes,
                     fontsize=7, color=ANOMALY_MARKER, va="top")
        for j in range(5)
    ]

    # ── animation update ─────────────────────────────────────────────────────
    def update(_frame):
        import datetime
        tick_counter[0] += 1
        t = tick_counter[0]
        tick_text.set_text(f"Tick: {t}")
        clock_text.set_text(datetime.datetime.now().strftime("%H:%M:%S"))

        for i, (sensor, state) in enumerate(zip(SENSORS, states)):
            val, is_anom = state.tick()
            score        = ml_score(rf, feature_cols, state, sensor["name"])
            hist         = list(state.hist)
            anom_flags   = list(state.anom)
            status       = state.status()

            # line
            lines[i].set_ydata(hist)

            # anomaly scatter
            ax_x = [j for j, f in enumerate(anom_flags) if f]
            ax_y = [hist[j] for j in ax_x]
            scatters[i].set_offsets(np.c_[ax_x, ax_y] if ax_x else np.empty((0, 2)))

            # value + status label
            val_texts[i].set_text(f"{val:.1f}{sensor['unit']}")
            val_texts[i].set_color(
                DANGER_COLOR if status == "danger" else
                WARN_COLOR   if status == "warn"   else "white"
            )
            label = ("↓ draining" if sensor.get("drain") and status == "ok"
                     else "⚠ WARNING" if status == "warn"
                     else "🔴 DANGER" if status == "danger"
                     else "● healthy")
            status_texts[i].set_text(label)
            status_texts[i].set_color(
                DANGER_COLOR if status == "danger" else
                WARN_COLOR   if status == "warn"   else
                "#5588aa"    if sensor.get("drain") else HEALTHY_COLOR
            )

            # ML score bar
            bar = ml_bars[i][0]
            bar.set_width(score)
            bar.set_color(
                DANGER_COLOR if score > 0.7 else
                WARN_COLOR   if score > 0.5 else
                sensor["color"]
            )
            score_texts[i].set_text(f"{score:.2f}")

            # anomaly log entry
            if is_anom:
                anomaly_log.insert(0, f"T{t:04d}  {sensor['name'][:18]:<18}  {val:.1f}{sensor['unit']}")
                if len(anomaly_log) > 5:
                    anomaly_log.pop()

        for j, lt in enumerate(log_texts):
            lt.set_text(anomaly_log[j] if j < len(anomaly_log) else "")

        return lines + [s for s in scatters] + val_texts + status_texts + \
               score_texts + log_texts + [tick_text, clock_text]

    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False
    )

    # legend
    legend_elems = [
        Line2D([0], [0], color="#555555", linewidth=1.5, label="sensor reading"),
        Line2D([0], [0], marker="x", color=ANOMALY_MARKER, linestyle="None",
               markersize=6, label="anomaly spike"),
    ]
    fig.legend(handles=legend_elems, loc="lower left", fontsize=7,
               facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white",
               ncol=2, bbox_to_anchor=(0.01, 0.00))

    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Live vehicle anomaly monitoring dashboard"
    )
    parser.add_argument("--csv",        default=DEFAULT_CSV,
                        help="Path to vehicle_logs.csv (used only to pick vehicle IDs)")
    parser.add_argument("--vehicle_id", default=None,
                        help="Vehicle to monitor (default: first in CSV)")
    parser.add_argument("--interval",   type=int, default=600,
                        help="Milliseconds between ticks (default: 600)")
    parser.add_argument("--window",     type=int, default=60,
                        help="Number of ticks shown in each chart (default: 60)")
    args = parser.parse_args()

    # resolve vehicle_id from CSV so we give a helpful error if it's wrong
    try:
        df = pd.read_csv(args.csv, nrows=200)
        ids = sorted(df["vehicle_id"].unique())
    except Exception:
        ids = ["VH-001"]

    vehicle_id = args.vehicle_id or ids[0]
    if vehicle_id not in ids:
        print(f"❌  vehicle_id '{vehicle_id}' not found.  Available: {ids}")
        sys.exit(1)

    print(f"🚗  Starting live monitor for {vehicle_id}")
    print(f"   Tick interval : {args.interval} ms")
    print(f"   Scroll window : {args.window} ticks")
    print(f"   ML model      : {'loaded' if os.path.exists(os.path.join(MODEL_DIR, 'random_forest.joblib')) else 'not found – run train_model.py'}")
    print("   Close the window to stop.\n")

    build_dashboard(vehicle_id, args.window, args.interval)


if __name__ == "__main__":
    main()
