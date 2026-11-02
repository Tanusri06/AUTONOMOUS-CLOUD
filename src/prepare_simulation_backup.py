from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_data"
RESULTS_DIR = ROOT / "outputs" / "experiment_results"
MODELS_DIR = ROOT / "models"

SPLIT_FILE = RESULTS_DIR / "vm_split_ids.json"
RF_FILE = MODELS_DIR / "random_forest_cpu.joblib"
IF_FILE = MODELS_DIR / "isolation_forest.joblib"

WORKLOAD_FILE = RESULTS_DIR / "simulation_workloads.csv.gz"
FAILURE_FILE = RESULTS_DIR / "failure_schedule.json"

# ============================================================
# SIMULATION SETTINGS
# ============================================================

NUM_VMS = 30
NUM_STEPS = 288
RANDOM_SEED = 42

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


# ============================================================
# HELPERS
# ============================================================

def normalize_vm_id(vm_id):
    """
    Convert IDs such as:
        1007
        '1007'
        '1007.csv'
        '1007.csv.gz'
    into:
        '1007'
    """
    vm_id = str(vm_id).strip()

    if vm_id.endswith(".csv.gz"):
        vm_id = vm_id[:-7]
    elif vm_id.endswith(".csv"):
        vm_id = vm_id[:-4]

    return vm_id


def find_processed_file(vm_id):
    """Find a processed VM file using its normalized ID."""
    vm_name = normalize_vm_id(vm_id)

    candidates = [
        DATA_DIR / f"{vm_name}.csv.gz",
        DATA_DIR / f"{vm_name}.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def check_required_files():
    """Verify that all required input files exist."""
    required_files = [
        SPLIT_FILE,
        RF_FILE,
        IF_FILE,
    ]

    missing = [str(file) for file in required_files if not file.exists()]

    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(
            "The following required files are missing:\n" + message
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed data folder not found: {DATA_DIR}"
        )


# ============================================================
# MAIN PREPARATION
# ============================================================

def main():
    print("=" * 60)
    print("PREPARING SIMULATION WORKLOADS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # 2. Load VM split
    # --------------------------------------------------------

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    if "test" not in split_data:
        raise KeyError(
            "The VM split JSON does not contain a 'test' list."
        )

    test_vm_ids = split_data["test"]

    if not test_vm_ids:
        raise ValueError("The test VM list is empty.")

    # Sort numerically when possible
    test_vm_ids = sorted(
        test_vm_ids,
        key=lambda value: int(normalize_vm_id(value))
        if normalize_vm_id(value).isdigit()
        else normalize_vm_id(value)
    )

    selected_vm_ids = test_vm_ids[:NUM_VMS]

    print(f"Test VMs available: {len(test_vm_ids)}")
    print(f"VMs requested for simulation: {NUM_VMS}")

    # --------------------------------------------------------
    # 3. Load trained models
    # --------------------------------------------------------

    print("\nLoading trained models...")

    rf_model = joblib.load(RF_FILE)
    isolation_model = joblib.load(IF_FILE)

    print("Random Forest model loaded.")
    print("Isolation Forest model loaded.")

    # --------------------------------------------------------
    # 4. Prepare each VM workload
    # --------------------------------------------------------

    workload_frames = []
    skipped_vms = []

    for position, vm_id in enumerate(selected_vm_ids, start=1):
        vm_name = normalize_vm_id(vm_id)
        vm_file = find_processed_file(vm_id)

        if vm_file is None:
            skipped_vms.append(vm_name)
            print(
                f"[{position}/{NUM_VMS}] Missing processed file for VM "
                f"{vm_name}"
            )
            continue

        try:
            df = pd.read_csv(vm_file)

            # Ensure chronological order
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["datetime"],
                    errors="coerce"
                )
                df = df.sort_values("datetime")

            # Check model features
            missing_columns = [
                column for column in FEATURES
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing feature columns: {missing_columns}"
                )

            # Remove rows with missing model inputs
            df = df.dropna(subset=FEATURES).reset_index(drop=True)

            if len(df) < NUM_STEPS:
                raise ValueError(
                    f"Only {len(df)} usable rows; "
                    f"{NUM_STEPS} are required."
                )

            # Use exactly 288 chronological rows
            df = df.iloc[:NUM_STEPS].copy()

            X = df[FEATURES]

            # ------------------------------------------------
            # Random Forest: next-step CPU prediction
            # ------------------------------------------------

            predicted_cpu = rf_model.predict(X)
            predicted_cpu = np.clip(predicted_cpu, 0, 100)

            # ------------------------------------------------
            # Isolation Forest: anomaly detection
            # ------------------------------------------------

            anomaly_labels = isolation_model.predict(X)

            # Isolation Forest returns -1 for anomalies
            anomaly_flags = (anomaly_labels == -1).astype(int)

            # ------------------------------------------------
            # Build workload table
            # ------------------------------------------------

            vm_workload = pd.DataFrame({
                "vm_id": vm_name,
                "step": np.arange(NUM_STEPS),
                "actual_cpu": df["cpu_percent"].to_numpy(),
                "predicted_cpu": predicted_cpu,
                "memory_usage_kb": df["memory_usage_kb"].to_numpy(),
                "disk_read_kbps": df["disk_read_kbps"].to_numpy(),
                "disk_write_kbps": df["disk_write_kbps"].to_numpy(),
                "network_received_kbps": (
                    df["network_received_kbps"].to_numpy()
                ),
                "network_transmitted_kbps": (
                    df["network_transmitted_kbps"].to_numpy()
                ),
                "anomaly_flag": anomaly_flags,
            })

            workload_frames.append(vm_workload)

            print(
                f"[{position}/{NUM_VMS}] Prepared VM {vm_name}"
            )

        except Exception as error:
            skipped_vms.append(vm_name)
            print(
                f"[{position}/{NUM_VMS}] Could not prepare VM "
                f"{vm_name}: {error}"
            )

    # --------------------------------------------------------
    # 5. Validate workload count
    # --------------------------------------------------------

    prepared_vm_count = len(workload_frames)

    if prepared_vm_count != NUM_VMS:
        raise RuntimeError(
            "\nSimulation workload preparation failed.\n"
            f"Required VMs: {NUM_VMS}\n"
            f"Successfully prepared: {prepared_vm_count}\n"
            f"Skipped VMs: {skipped_vms}\n"
            "No workload file was saved. Check the processed data "
            "and VM split before continuing."
        )

    # --------------------------------------------------------
    # 6. Combine and save workloads
    # --------------------------------------------------------

    workloads = pd.concat(
        workload_frames,
        ignore_index=True
    )

    workloads = workloads[OUTPUT_COLUMNS]

    expected_rows = NUM_VMS * NUM_STEPS

    if len(workloads) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} workload rows, "
            f"but generated {len(workloads)}."
        )

    workloads.to_csv(
        WORKLOAD_FILE,
        index=False,
        compression="gzip"
    )

    # --------------------------------------------------------
    # 7. Create cloud failure schedule
    # --------------------------------------------------------

    failure_schedule = {
        "Cloud_C": {
            "start_step": 60,
            "end_step": 72
        },
        "Cloud_A": {
            "start_step": 150,
            "end_step": 162
        },
        "Cloud_B": {
            "start_step": 225,
            "end_step": 237
        },
    }

    with open(
        FAILURE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            failure_schedule,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 8. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SIMULATION INPUTS CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Prepared VMs: {prepared_vm_count}")
    print(f"Steps per VM: {NUM_STEPS}")
    print(f"Total workload rows: {len(workloads)}")

    print(f"\nWorkload file:\n{WORKLOAD_FILE}")
    print(f"\nFailure schedule:\n{FAILURE_FILE}")

    print("\nWorkload preview:")
    print(workloads.head().to_string(index=False))


if __name__ == "__main__":
    main()