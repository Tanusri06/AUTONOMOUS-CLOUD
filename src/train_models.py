from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/processed_data")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs/experiment_results")

MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the saved VM-level split
with open(OUT_DIR / "vm_split_ids.json") as f:
    splits = json.load(f)

train_ids = splits["train"]
val_ids = splits["validation"]
test_ids = splits["test"]

features = [
    "cpu_percent", "cpu_lag_1", "cpu_lag_2", "cpu_lag_3",
    "memory_usage_kb", "disk_read_kbps", "disk_write_kbps",
    "network_received_kbps", "network_transmitted_kbps",
    "hour_utc", "minute_utc"
]

MAX_ROWS_PER_VM = 2000
RANDOM_STATE = 42


def load_vm_data(vm_ids, max_rows=None):
    frames = []

    for vm in vm_ids:
        path = DATA_DIR / vm
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        data = pd.read_csv(path)
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=features + ["target_cpu_next"])

        if max_rows and len(data) > max_rows:
            data = data.sample(max_rows, random_state=RANDOM_STATE)

        frames.append(data)

    if not frames:
        raise ValueError("No VM data was loaded.")

    return pd.concat(frames, ignore_index=True)


print("Loading training VMs...")
train_df = load_vm_data(train_ids, MAX_ROWS_PER_VM)

print("Loading validation VMs...")
val_df = load_vm_data(val_ids, MAX_ROWS_PER_VM)

X_train = train_df[features]
y_train = train_df["target_cpu_next"]

X_val = val_df[features]
y_val = val_df["target_cpu_next"]

# Random Forest CPU prediction model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining Random Forest on training VMs...")
model.fit(X_train, y_train)

val_pred = model.predict(X_val)

rf_mae = mean_absolute_error(y_val, val_pred)
rf_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

# Persistence baseline: predict next CPU using current CPU
baseline_pred = val_df["cpu_percent"]
baseline_mae = mean_absolute_error(y_val, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))

joblib.dump(model, MODEL_DIR / "random_forest_cpu.joblib")

# Isolation Forest trained only on training VMs
print("\nTraining Isolation Forest on training VMs...")

anomaly_df = load_vm_data(train_ids, max_rows=1000)

detector = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.02,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

detector.fit(anomaly_df[features])
joblib.dump(detector, MODEL_DIR / "isolation_forest.joblib")

# Save validation results
results = pd.DataFrame([
    {
        "model": "Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "training_rows": len(X_train),
        "validation_rows": len(X_val)
    },
    {
        "model": "Persistence Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "training_rows": len(X_train),
        "validation_rows": len(X_val)
    }
])

results.to_csv(OUT_DIR / "validation_metrics.csv", index=False)

print("\nTraining complete!")
print(results.to_string(index=False))
print("\nModels saved in:", MODEL_DIR)
print("Validation metrics saved in:", OUT_DIR)