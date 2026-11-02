from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "bitbrains" / "fastStorage" / "2013-8"

OUT_DIR = PROJECT_DIR / "data" / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "dashboard_workload_30vms.csv"

NUM_VMS = 30
NUM_STEPS = 288
INTERVAL = "5min"

# Raw Bitbrains column names
RAW_COLUMNS = {
    "Timestamp [ms]": "timestamp",
    "CPU usage [%]": "cpu_percent",
    "Memory usage [KB]": "memory_usage_kb",
    "Disk read throughput [KB/s]": "disk_read_kbps",
    "Disk write throughput [KB/s]": "disk_write_kbps",
    "Network received throughput [KB/s]": "network_received_kbps",
    "Network transmitted throughput [KB/s]": "network_transmitted_kbps",
}

REQUIRED_COLUMNS = list(RAW_COLUMNS.values())


# ============================================================
# VALIDATE DATASET
# ============================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

files = sorted(DATA_DIR.glob("*.csv"))

if len(files) < NUM_VMS:
    raise ValueError(
        f"Expected at least {NUM_VMS} VM CSV files, found {len(files)}"
    )

print(f"Found {len(files)} raw VM CSV files.")
print(f"Selecting the first {NUM_VMS} files.")


# ============================================================
# PROCESS VM FILES
# ============================================================

vm_frames = []

for vm_number, file_path in enumerate(files[:NUM_VMS], start=1):
    print(f"Processing VM {vm_number}/{NUM_VMS}: {file_path.name}")

    # Bitbrains files are semicolon-separated
    df = pd.read_csv(file_path, sep=";")

    # Clean column names
    df.columns = (
        df.columns
        .str.replace("\t", "", regex=False)
        .str.strip()
    )

    # Rename raw columns to standardized names
    df = df.rename(columns=RAW_COLUMNS)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing}"
        )

    # Convert values to numeric
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # The supplied sample timestamps are Unix seconds
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
        errors="coerce"
    )

    # Remove invalid rows and sort chronologically
    df = df.dropna(subset=["datetime"] + REQUIRED_COLUMNS)
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset="datetime")

    # Use datetime as the index for resampling
    df = df.set_index("datetime")

    # Resample to 5-minute intervals.
    # Mean is used for CPU, memory, disk, and network measurements.
    df = df[REQUIRED_COLUMNS].resample(INTERVAL).mean()

    # Fill small gaps using time interpolation, then edge values
    df = df.interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()

    # Keep exactly 288 time steps
    df = df.head(NUM_STEPS)

    if len(df) < NUM_STEPS:
        raise ValueError(
            f"{file_path.name} has only {len(df)} resampled rows; "
            f"expected {NUM_STEPS}. Choose a different VM file or "
            f"adjust the time window."
        )

    # Standardized VM identifier
    df["vm_id"] = f"VM_{vm_number:02d}"

    # Convert timestamp index into a column
    df = df.reset_index()
    df["timestamp"] = df["datetime"].astype(str)

    # Keep only dashboard-required columns in exact order
    df = df[
        [
            "vm_id",
            "timestamp",
            "cpu_percent",
            "memory_usage_kb",
            "disk_read_kbps",
            "disk_write_kbps",
            "network_received_kbps",
            "network_transmitted_kbps",
        ]
    ]

    vm_frames.append(df)


# ============================================================
# COMBINE AND SAVE
# ============================================================

final_df = pd.concat(vm_frames, ignore_index=True)

# Ensure CPU values are within the expected range
final_df["cpu_percent"] = final_df["cpu_percent"].clip(0, 100)

# Check final row count
expected_rows = NUM_VMS * NUM_STEPS

if len(final_df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, generated {len(final_df)}"
    )

# Save dashboard-ready CSV
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n============================================")
print("Dashboard CSV created successfully!")
print("============================================")
print(f"Output file: {OUTPUT_FILE}")
print(f"VMs: {final_df['vm_id'].nunique()}")
print(f"Rows: {len(final_df)}")
print(f"Expected rows: {expected_rows}")
print("\nColumns:")
print(list(final_df.columns))
print("\nFirst 5 rows:")
print(final_df.head())