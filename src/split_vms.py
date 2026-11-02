from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/processed_data")
OUT_DIR = Path("outputs/experiment_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = DATA_DIR / "preprocessing_manifest.csv"
manifest = pd.read_csv(manifest_path)

successful = manifest[
    (manifest["status"] == "success") &
    (manifest["valid_rows"] > 0)
]

vm_ids = sorted(successful["output_file"].tolist())

assert len(vm_ids) == 1250, f"Expected 1250 VMs, found {len(vm_ids)}"

# 80% development pool, 20% final test
train_val_ids, test_ids = train_test_split(
    vm_ids, test_size=0.20, random_state=42
)

# Split the 80% pool into 65% train and 15% validation overall
train_ids, val_ids = train_test_split(
    train_val_ids, test_size=0.1875, random_state=42
)

splits = (
    [{"vm_id": vm, "split": "train"} for vm in train_ids] +
    [{"vm_id": vm, "split": "validation"} for vm in val_ids] +
    [{"vm_id": vm, "split": "test"} for vm in test_ids]
)

split_df = pd.DataFrame(splits)
split_df.to_csv(OUT_DIR / "vm_split_manifest.csv", index=False)

with open(OUT_DIR / "vm_split_ids.json", "w") as f:
    json.dump({
        "train": sorted(train_ids),
        "validation": sorted(val_ids),
        "test": sorted(test_ids)
    }, f, indent=2)

print("VM-level split complete")
print(f"Train:      {len(train_ids)} VMs ({len(train_ids)/len(vm_ids)*100:.1f}%)")
print(f"Validation: {len(val_ids)} VMs ({len(val_ids)/len(vm_ids)*100:.1f}%)")
print(f"Test:       {len(test_ids)} VMs ({len(test_ids)/len(vm_ids)*100:.1f}%)")
print("Saved split files in:", OUT_DIR)