from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/processed_data")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs/experiment_results")

with open(OUT_DIR / "vm_split_ids.json") as f:
    splits = json.load(f)

test_ids = splits["test"]

features = [
    "cpu_percent", "cpu_lag_1", "cpu_lag_2", "cpu_lag_3",
    "memory_usage_kb", "disk_read_kbps", "disk_write_kbps",
    "network_received_kbps", "network_transmitted_kbps",
    "hour_utc", "minute_utc"
]

model = joblib.load(MODEL_DIR / "random_forest_cpu.joblib")

frames = []
for vm in test_ids:
    path = DATA_DIR / vm
    if not path.exists():
        print(f"Missing test file: {path}")
        continue

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + ["target_cpu_next"])
    frames.append(df)

if not frames:
    raise ValueError("No test VM data loaded.")

test_df = pd.concat(frames, ignore_index=True)

X_test = test_df[features]
y_test = test_df["target_cpu_next"]

# Random Forest evaluation
rf_pred = model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# Persistence baseline: current CPU predicts next CPU
baseline_pred = test_df["cpu_percent"]

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

results = pd.DataFrame([
    {
        "model": "Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "test_vms": len(test_ids),
        "test_rows": len(test_df)
    },
    {
        "model": "Persistence Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "test_vms": len(test_ids),
        "test_rows": len(test_df)
    }
])

results.to_csv(OUT_DIR / "test_metrics.csv", index=False)

print("\nFinal evaluation on unseen test VMs")
print(results.to_string(index=False))
print("\nSaved to:", OUT_DIR / "test_metrics.csv")