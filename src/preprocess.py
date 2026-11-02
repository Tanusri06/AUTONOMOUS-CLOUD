from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/bitbrains/fastStorage/2013-8")
OUT_DIR = Path("data/processed_data")
MANIFEST = OUT_DIR / "preprocessing_manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

records = []

for i, path in enumerate(files, 1):
    out = OUT_DIR / f"{path.stem}.csv.gz"

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.replace("\t", "", regex=False).str.strip()

        df = df.rename(columns={
            "Timestamp [ms]": "timestamp",
            "CPU usage [%]": "cpu_percent",
            "Memory usage [KB]": "memory_usage_kb",
            "Disk read throughput [KB/s]": "disk_read_kbps",
            "Disk write throughput [KB/s]": "disk_write_kbps",
            "Network received throughput [KB/s]": "network_received_kbps",
            "Network transmitted throughput [KB/s]": "network_transmitted_kbps",
        })

        required = [
            "timestamp", "cpu_percent", "memory_usage_kb",
            "disk_read_kbps", "disk_write_kbps",
            "network_received_kbps", "network_transmitted_kbps"
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Detect whether timestamps are seconds or milliseconds
        unit = "ms" if df["timestamp"].median() > 1e11 else "s"
        df["datetime"] = pd.to_datetime(
            df["timestamp"], unit=unit, utc=True, errors="coerce"
        )

        df = df.dropna(subset=["datetime", "cpu_percent"])
        df = df.sort_values("datetime").drop_duplicates("datetime")

        # Notebook-compatible CPU history and next-step target
        df["cpu_lag_1"] = df["cpu_percent"].shift(1)
        df["cpu_lag_2"] = df["cpu_percent"].shift(2)
        df["cpu_lag_3"] = df["cpu_percent"].shift(3)
        df["target_cpu_next"] = df["cpu_percent"].shift(-1)

        df["hour_utc"] = df["datetime"].dt.hour
        df["minute_utc"] = df["datetime"].dt.minute

        df = df.rename(columns={"timestamp": "timestamp_original"})

        columns = [
            "datetime",
            "cpu_percent", "cpu_lag_1", "cpu_lag_2", "cpu_lag_3",
            "memory_usage_kb",
            "disk_read_kbps", "disk_write_kbps",
            "network_received_kbps", "network_transmitted_kbps",
            "hour_utc", "minute_utc", "target_cpu_next"
        ]

        result = df[columns].dropna()
        result.to_csv(out, index=False, compression="gzip")

        records.append({
            "file": path.name,
            "status": "success",
            "valid_rows": len(result),
            "output_file": out.name
        })

    except Exception as e:
        records.append({
            "file": path.name,
            "status": "error",
            "valid_rows": 0,
            "output_file": out.name,
            "message": str(e)
        })

    if i % 25 == 0 or i == len(files):
        pd.DataFrame(records).to_csv(MANIFEST, index=False)
        print(f"Processed {i}/{len(files)} files")

pd.DataFrame(records).to_csv(MANIFEST, index=False)

successes = sum(r["status"] == "success" for r in records)
print("\nPreprocessing finished.")
print("Successful VM files:", successes)
print("Failed VM files:", len(records) - successes)
print("Output folder:", OUT_DIR)
print("Manifest:", MANIFEST)