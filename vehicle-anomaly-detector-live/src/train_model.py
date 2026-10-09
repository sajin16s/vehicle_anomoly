"""
train_model.py
----------------
Loads data/vehicle_logs.csv (the CSV is now the single source of truth --
no hardcoded data), engineers features, and trains two models:

1. IsolationForest (unsupervised) -- learns what "normal" looks like per
   component without using the is_anomaly labels, then flags outliers.
2. RandomForestClassifier (supervised) -- trained directly on the
   is_anomaly ground truth; this is the model that gives the headline
   accuracy number, and what the visualization script uses by default.

Run:
    python src/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from feature_extraction import engineer_features, get_feature_matrix

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "vehicle_logs.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))
    return acc


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Load from CSV (this is the "advanced" change: CSV is the input,
    # not a hardcoded Python list) ----
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(raw_df)} readings from {DATA_PATH}")

    df = engineer_features(raw_df)
    print(f"After feature engineering (rolling-window warm-up dropped): {len(df)} rows")

    X, feature_cols = get_feature_matrix(df)
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 1. Unsupervised anomaly detector ----
    contamination = max(y_train.mean(), 0.01)
    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    )
    iso_forest.fit(X_train)
    iso_preds = (iso_forest.predict(X_test) == -1).astype(int)
    evaluate("IsolationForest (unsupervised)", y_test, iso_preds)

    # ---- 2. Supervised classifier ----
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_accuracy = evaluate("RandomForestClassifier (supervised)", y_test, rf_preds)

    # feature importance -- nice to show off in a resume/portfolio writeup
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8))

    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"FINAL HEADLINE METRIC -> RandomForest accuracy: {rf_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
