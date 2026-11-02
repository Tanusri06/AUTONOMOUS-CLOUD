from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed_data"
RESULTS_DIR = ROOT_DIR / "outputs" / "experiment_results"
MODELS_DIR = ROOT_DIR / "models"

SPLIT_FILE = RESULTS_DIR / "vm_split_ids.json"

RF_MODEL_FILE = MODELS_DIR / "random_forest_cpu.joblib"
IF_MODEL_FILE = MODELS_DIR / "isolation_forest.joblib"

WORKLOAD_FILE = RESULTS_DIR / "simulation_workloads.csv.gz"
FAILURE_FILE = RESULTS_DIR / "failure_schedule.json"

NUM_VMS = 30
NUM_STEPS = 288
RANDOM_SEED = 42


# ---------------------------------------------------------
# MODEL FEATURES
# Keep this order identical to the model training order.
# ---------------------------------------------------------

FEATURES = [
    "cpu_percent",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
    "hour_utc",
    "minute_utc",
]

RAW_COLUMNS = [
    "vm_id",
    "timestamp",
    "cpu_percent",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
]

OUTPUT_COLUMNS = [
    "vm_id",
    "step",
    "actual_cpu",
    "predicted_cpu",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
    "anomaly_flag",
]


# ---------------------------------------------------------
# DEFAULT SIMULATED CLOUD FAILURES
# ---------------------------------------------------------

DEFAULT_FAILURE_SCHEDULE = {
    "Cloud_C": {"start_step": 60, "end_step": 72},
    "Cloud_A": {"start_step": 150, "end_step": 162},
    "Cloud_B": {"start_step": 225, "end_step": 237},
}


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def normalize_vm_id(value):
    """Normalize VM IDs so filenames and IDs can be compared."""
    value = str(value).strip()

    if value.endswith(".csv.gz"):
        value = value[:-7]
    elif value.endswith(".csv"):
        value = value[:-4]

    return value


def find_processed_file(vm_id):
    """Find the processed CSV file belonging to a VM."""
    vm_id = normalize_vm_id(vm_id)

    candidates = [
        DATA_DIR / f"{vm_id}.csv",
        DATA_DIR / f"{vm_id}.csv.gz",
    ]

    for file_path in candidates:
        if file_path.exists():
            return file_path

    matches = list(DATA_DIR.rglob(f"{vm_id}.csv"))
    matches += list(DATA_DIR.rglob(f"{vm_id}.csv.gz"))

    return matches[0] if matches else None


def check_required_files():
    """Check whether the original Bitbrains workflow files exist."""
    required_files = [
        SPLIT_FILE,
        RF_MODEL_FILE,
        IF_MODEL_FILE,
    ]

    missing = [file for file in required_files if not file.exists()]

    if missing:
        print("\nMissing required files:")

        for file in missing:
            print(f" - {file}")

        raise FileNotFoundError(
            "Some required files are missing. Check the paths above."
        )


def load_models():
    """Load the saved models without retraining them."""
    if not RF_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {RF_MODEL_FILE}"
        )

    if not IF_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: {IF_MODEL_FILE}"
        )

    rf_model = joblib.load(RF_MODEL_FILE)
    if_model = joblib.load(IF_MODEL_FILE)

    return rf_model, if_model


def create_failure_schedule():
    """Return a fresh copy of the default failure schedule."""
    return json.loads(json.dumps(DEFAULT_FAILURE_SCHEDULE))


def save_failure_schedule(failure_schedule, output_path):
    """Save the failure schedule as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(failure_schedule, file, indent=4)


# ---------------------------------------------------------
# UPLOADED CSV VALIDATION AND FEATURE PREPARATION
# ---------------------------------------------------------

def validate_uploaded_csv(upload_df):
    """Validate the standardized uploaded workload CSV."""
    if upload_df is None or upload_df.empty:
        raise ValueError("The uploaded CSV is empty.")

    missing_columns = [
        column for column in RAW_COLUMNS
        if column not in upload_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing these columns: "
            + ", ".join(missing_columns)
        )

    df = upload_df[RAW_COLUMNS].copy()

    df["vm_id"] = df["vm_id"].astype(str).str.strip()

    if (df["vm_id"] == "").any():
        raise ValueError("The CSV contains an empty vm_id.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamp values are invalid. Use a valid date/time format."
        )

    numeric_columns = [
        "cpu_percent",
        "memory_usage_kb",
        "disk_read_kbps",
        "disk_write_kbps",
        "network_received_kbps",
        "network_transmitted_kbps",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "The CSV contains missing or non-numeric workload values."
        )

    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError(
            "The CSV contains infinite or invalid numeric values."
        )

    if ((df["cpu_percent"] < 0) | (df["cpu_percent"] > 100)).any():
        raise ValueError("cpu_percent must be between 0 and 100.")

    df = df.sort_values(
        ["vm_id", "timestamp"]
    ).reset_index(drop=True)

    if df.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError(
            "The CSV contains duplicate timestamps for the same VM."
        )

    vm_ids = sorted(df["vm_id"].unique())

    if len(vm_ids) != NUM_VMS:
        raise ValueError(
            f"The CSV must contain exactly {NUM_VMS} VMs. "
            f"Found {len(vm_ids)}."
        )

    rows_per_vm = df.groupby("vm_id").size()

    incorrect_counts = rows_per_vm[rows_per_vm != NUM_STEPS]

    if not incorrect_counts.empty:
        details = ", ".join(
            f"{vm_id}: {count} rows"
            for vm_id, count in incorrect_counts.items()
        )

        raise ValueError(
            f"Each VM must contain exactly {NUM_STEPS} rows. "
            f"Incorrect counts: {details}"
        )

    return df


def prepare_uploaded_workload(
    upload_df,
    num_vms=NUM_VMS,
    num_steps=NUM_STEPS,
):
    """
    Prepare an uploaded standardized CSV for the simulator.

    Returns:
        workload_df: DataFrame containing simulator-ready workload
        failure_schedule: Default simulated cloud failure schedule
    """

    if num_vms != NUM_VMS or num_steps != NUM_STEPS:
        raise ValueError(
            f"This version expects {NUM_VMS} VMs and "
            f"{NUM_STEPS} steps."
        )

    df = validate_uploaded_csv(upload_df)

    rf_model, if_model = load_models()

    # Generate lag features independently for each VM.
    grouped_cpu = df.groupby("vm_id")["cpu_percent"]

    df["cpu_lag_1"] = grouped_cpu.shift(1)
    df["cpu_lag_2"] = grouped_cpu.shift(2)
    df["cpu_lag_3"] = grouped_cpu.shift(3)

    # For the first few rows, use that VM's earliest available CPU value.
    # This preserves all 288 rows per VM.
    for column in ["cpu_lag_1", "cpu_lag_2", "cpu_lag_3"]:
        df[column] = df[column].fillna(df["cpu_percent"])

    df["hour_utc"] = df["timestamp"].dt.hour
    df["minute_utc"] = df["timestamp"].dt.minute

    # Ensure the model receives the expected feature order.
    X = df[FEATURES].copy()

    predicted_cpu = rf_model.predict(X)
    predicted_cpu = np.clip(predicted_cpu, 0, 100)

    anomaly_predictions = if_model.predict(X)

    # Isolation Forest returns -1 for anomalies and 1 for normal rows.
    anomaly_flag = (anomaly_predictions == -1).astype(int)

    df["predicted_cpu"] = predicted_cpu
    df["anomaly_flag"] = anomaly_flag

    # Assign a sequential step number to each VM.
    df["step"] = df.groupby("vm_id").cumcount()

    workload_df = df.rename(
        columns={"cpu_percent": "actual_cpu"}
    )[OUTPUT_COLUMNS].copy()

    workload_df = workload_df.sort_values(
        ["vm_id", "step"]
    ).reset_index(drop=True)

    expected_rows = num_vms * num_steps

    if len(workload_df) != expected_rows:
        raise ValueError(
            f"Prepared workload has {len(workload_df)} rows; "
            f"expected {expected_rows}."
        )

    failure_schedule = create_failure_schedule()

    return workload_df, failure_schedule


def prepare_uploaded_csv_file(
    csv_path,
    workload_output_path=None,
    failure_output_path=None,
):
    """
    Convenience function for preparing an uploaded CSV file from a path.

    This can be called by the dashboard after the user uploads a CSV.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Uploaded CSV not found: {csv_path}")

    upload_df = pd.read_csv(csv_path)

    workload_df, failure_schedule = prepare_uploaded_workload(upload_df)

    if workload_output_path is not None:
        workload_output_path = Path(workload_output_path)
        workload_output_path.parent.mkdir(parents=True, exist_ok=True)

        workload_df.to_csv(
            workload_output_path,
            index=False,
            compression="gzip" if str(workload_output_path).endswith(".gz") else None,
        )

    if failure_output_path is not None:
        save_failure_schedule(
            failure_schedule,
            failure_output_path
        )

    return workload_df, failure_schedule


# ---------------------------------------------------------
# ORIGINAL BITBRAINS WORKFLOW
# ---------------------------------------------------------

def prepare_bitbrains_workload():
    """Prepare the simulation workload using the existing processed dataset."""
    check_required_files()

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split_ids = json.load(file)

    test_vm_ids = split_ids["test"]

    test_vm_ids = sorted(
        test_vm_ids,
        key=lambda value: normalize_vm_id(value)
    )

    selected_vm_ids = test_vm_ids[:NUM_VMS]

    if len(selected_vm_ids) < NUM_VMS:
        raise ValueError(
            f"Only {len(selected_vm_ids)} test VMs found. "
            f"At least {NUM_VMS} are required."
        )

    rf_model, if_model = load_models()

    all_workloads = []

    for vm_id in selected_vm_ids:
        file_path = find_processed_file(vm_id)

        if file_path is None:
            print(f"Skipping VM {vm_id}: processed file not found.")
            continue

        vm_df = pd.read_csv(file_path)

        missing_features = [
            feature for feature in FEATURES
            if feature not in vm_df.columns
        ]

        if missing_features:
            print(
                f"Skipping VM {vm_id}: missing features "
                f"{missing_features}"
            )
            continue

        vm_df = vm_df.dropna(subset=FEATURES).copy()

        if "target_cpu_next" not in vm_df.columns:
            print(
                f"Skipping VM {vm_id}: target_cpu_next not found."
            )
            continue

        if len(vm_df) < NUM_STEPS:
            print(
                f"Skipping VM {vm_id}: only {len(vm_df)} usable rows."
            )
            continue

        vm_df = vm_df.head(NUM_STEPS).copy()

        X = vm_df[FEATURES]

        predicted_cpu = rf_model.predict(X)
        predicted_cpu = np.clip(predicted_cpu, 0, 100)

        anomaly_predictions = if_model.predict(X)
        anomaly_flag = (anomaly_predictions == -1).astype(int)

        if "cpu_percent" not in vm_df.columns:
            print(
                f"Skipping VM {vm_id}: cpu_percent not found."
            )
            continue

        output_df = pd.DataFrame({
            "vm_id": normalize_vm_id(vm_id),
            "step": range(NUM_STEPS),
            "actual_cpu": vm_df["cpu_percent"].to_numpy(),
            "predicted_cpu": predicted_cpu,
            "memory_usage_kb": vm_df["memory_usage_kb"].to_numpy(),
            "disk_read_kbps": vm_df["disk_read_kbps"].to_numpy(),
            "disk_write_kbps": vm_df["disk_write_kbps"].to_numpy(),
            "network_received_kbps": (
                vm_df["network_received_kbps"].to_numpy()
            ),
            "network_transmitted_kbps": (
                vm_df["network_transmitted_kbps"].to_numpy()
            ),
            "anomaly_flag": anomaly_flag,
        })

        all_workloads.append(output_df)

    if not all_workloads:
        raise ValueError(
            "No valid VM workloads were prepared. "
            "Check the processed data and feature columns."
        )

    workload_df = pd.concat(
        all_workloads,
        ignore_index=True
    )

    actual_vm_count = workload_df["vm_id"].nunique()

    if actual_vm_count != NUM_VMS:
        raise ValueError(
            f"Prepared {actual_vm_count} VMs; expected {NUM_VMS}. "
            "Check whether some processed VM files were skipped."
        )

    expected_rows = NUM_VMS * NUM_STEPS

    if len(workload_df) != expected_rows:
        raise ValueError(
            f"Prepared {len(workload_df)} rows; "
            f"expected {expected_rows}."
        )

    workload_df = workload_df[OUTPUT_COLUMNS]

    return workload_df, create_failure_schedule()


# ---------------------------------------------------------
# SAVE WORKLOAD AND FAILURE SCHEDULE
# ---------------------------------------------------------

def save_prepared_workload(
    workload_df,
    failure_schedule,
    workload_output_path=WORKLOAD_FILE,
    failure_output_path=FAILURE_FILE,
):
    """Save the simulator workload and failure schedule."""
    workload_output_path = Path(workload_output_path)
    workload_output_path.parent.mkdir(parents=True, exist_ok=True)

    workload_df.to_csv(
        workload_output_path,
        index=False,
        compression="gzip" if str(workload_output_path).endswith(".gz") else None,
    )

    save_failure_schedule(
        failure_schedule,
        failure_output_path
    )

    print("\nSimulation workload prepared successfully.")
    print(f"VMs: {workload_df['vm_id'].nunique()}")
    print(f"Steps per VM: {workload_df.groupby('vm_id').size().iloc[0]}")
    print(f"Total rows: {len(workload_df)}")
    print(f"Workload saved to: {workload_output_path}")
    print(f"Failure schedule saved to: {failure_output_path}")


# ---------------------------------------------------------
# MAIN
# Running this file directly keeps the original workflow.
# ---------------------------------------------------------

def main():
    print("Preparing simulation workload from Bitbrains data...")

    workload_df, failure_schedule = prepare_bitbrains_workload()

    save_prepared_workload(
        workload_df,
        failure_schedule,
        WORKLOAD_FILE,
        FAILURE_FILE,
    )


if __name__ == "__main__":
    main()