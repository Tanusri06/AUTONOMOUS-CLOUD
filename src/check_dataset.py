from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/bitbrains/fastStorage/2013-8")
files = sorted(DATA_DIR.glob("*.csv"))

print("VM files found:", len(files))

df = pd.read_csv(files[0], sep=";")
df.columns = df.columns.str.replace("\t", "", regex=False).str.strip()

print("\nSample VM:", files[0].name)
print("Rows and columns:", df.shape)
print("Clean columns:", df.columns.tolist())
print(df.head(3))