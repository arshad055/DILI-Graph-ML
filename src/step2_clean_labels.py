from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "DILIrank2.xlsx"
output_file = BASE_DIR / "data" / "dili_binary.csv"

# Read sheet with no header first
raw = pd.read_excel(input_file, header=None)

# Find the row that contains both needed column names
header_row = None
for i in range(len(raw)):
    row_values = raw.iloc[i].astype(str).str.strip().tolist()
    if "CompoundName" in row_values and "vDILI-Concern" in row_values:
        header_row = i
        break

print("Detected header row:", header_row)

if header_row is None:
    raise ValueError("Could not find the header row containing CompoundName and vDILI-Concern")

# Read again using the detected header row
df = pd.read_excel(input_file, header=header_row)

print("\nActual columns:")
print(df.columns.tolist())

# Keep only the columns we need
df = df[["CompoundName", "vDILI-Concern"]].copy()

# Clean strings
df["CompoundName"] = df["CompoundName"].astype(str).str.strip()
df["vDILI-Concern"] = df["vDILI-Concern"].astype(str).str.strip()

# Show unique labels before mapping
print("\nUnique vDILI-Concern values found:")
print(df["vDILI-Concern"].dropna().unique())

# Convert labels to binary
def map_label(x):
    x = str(x).strip().lower()
    if "ambiguous" in x:
        return None
    if "less" in x or "most" in x:
        return 1
    if "no-dili" in x or "no dili" in x:
        return 0
    return None

df["label"] = df["vDILI-Concern"].apply(map_label)

# Remove rows with unknown or ambiguous labels
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Rename and keep final columns
df = df.rename(columns={"CompoundName": "drug_name"})
df = df[["drug_name", "label"]]

# Remove empty names
df = df[df["drug_name"].notna()].copy()
df = df[df["drug_name"].astype(str).str.lower() != "nan"].copy()
df = df[df["drug_name"].astype(str).str.strip() != ""].copy()

# Save
df.to_csv(output_file, index=False)

print("\nSaved cleaned dataset to:", output_file)
print("\nPreview:")
print(df.head().to_string())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFinal shape:", df.shape)